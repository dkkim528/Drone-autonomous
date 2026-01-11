#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/float32_multi_array.hpp> 
#include <std_msgs/msg/bool.hpp> // [추가] 후진 신호용 헤더

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp> 

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

using std::placeholders::_1;
using namespace std::chrono_literals;

// A* 노드 구조체
struct LocalNode {
    int x, y, z; 
    double g, h;
    std::shared_ptr<LocalNode> parent;
    double f() const { return g + h; }
};

struct CompareLocalNode {
    bool operator()(const std::shared_ptr<LocalNode>& a, const std::shared_ptr<LocalNode>& b) {
        return a->f() > b->f();
    }
};

class LocalPlanner : public rclcpp::Node
{
public:
    LocalPlanner() : Node("local_planner_node")
    {
        // 파라미터: 안전거리 (장애물 팽창 반경)
        this->declare_parameter("safety_distance", 0.5); 
        safety_distance_ = this->get_parameter("safety_distance").as_double();

        RCLCPP_INFO(this->get_logger(), "Local Planner Started. Safety Dist: %.2f m", safety_distance_);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        global_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/global_path", 10, std::bind(&LocalPlanner::path_callback, this, _1));
        
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar/points", rclcpp::SensorDataQoS(), std::bind(&LocalPlanner::lidar_callback, this, _1));

        local_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/local_path", 10);
        local_map_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/local_grid_debug", 10);

        // [추가] Control Node에게 후진 모드 알림
        reverse_mode_pub_ = this->create_publisher<std_msgs::msg::Bool>("/local_planner/reverse_mode", 10);

        // Global Planner에게 경로 재계산을 요청하는 클라이언트
        replan_client_ = this->create_client<std_srvs::srv::Trigger>("replan_path");

        timer_ = this->create_wall_timer(50ms, std::bind(&LocalPlanner::control_loop, this));
    }

private:
    // 로컬 그리드 설정 (8m x 8m x 2m 범위 커버)
    const double grid_res_ = 0.2;     
    const int grid_dim_x_ = 40;       
    const int grid_dim_y_ = 40;       
    const int grid_dim_z_ = 10;       
    
    double safety_distance_;

    nav_msgs::msg::Path global_path_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_scan_;
    bool has_path_ = false;
    
    // 상태 관리 플래그
    bool is_stuck_ = false;       // 완전 고립 상태
    bool is_escaping_ = false;    // 후진 회피 동작 중
    rclcpp::Time escape_start_time_;

    std::vector<bool> local_occupancy_; 

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr local_map_pub_;
    
    // [추가] 후진 모드 퍼블리셔
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reverse_mode_pub_;

    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr replan_client_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        global_path_ = *msg;
        has_path_ = true;
        is_stuck_ = false; 
        is_escaping_ = false; 
        RCLCPP_INFO(this->get_logger(), "New Global Path Received. Resume Normal Operation.");
    }

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        latest_scan_ = msg;
    }

    // [추가] 후진 모드 신호 발행 헬퍼
    void publish_reverse_signal(bool is_reverse) {
        std_msgs::msg::Bool msg;
        msg.data = is_reverse;
        reverse_mode_pub_->publish(msg);
    }

    // ==================================================================================
    // [핵심] 제어 루프: 상황 판단 -> 행동 결정
    // ==================================================================================
    void control_loop() {
        if (!latest_scan_) return;

        // 1. 맵 업데이트 (Lidar -> Voxel -> Inflation)
        update_local_map(); 
        
        // 2. [회피 모드] 이미 후진 중이라면? (2초간 유지)
        if (is_escaping_) {
            if ((this->now() - escape_start_time_).seconds() > 2.0) {
                is_escaping_ = false; // 후진 끝, 다시 판단 모드로
                RCLCPP_INFO(this->get_logger(), "Escape maneuver finished. Re-evaluating...");
            } else {
                publish_backward_path(); // 계속 후진 경로 발행 (내부에서 reverse_mode=true 발행)
                return;
            }
        }

        // 3. [안전 검사] 드론이 지금 당장 위험한가? (Start Node Check)
        if (is_drone_in_danger()) {
            RCLCPP_WARN(this->get_logger(), "DANGER! Drone is inside safety radius.");
            
            // 3-1. 뒤는 안전한가?
            if (is_rear_safe()) {
                // 뒤가 비었으면 후진 시작
                start_escape_maneuver();
            } else {
                // 앞뒤 꽉 막힘 -> 진짜 갇힘 (Stuck)
                RCLCPP_ERROR(this->get_logger(), "TRAPPED! Cannot move forward or backward.");
                handle_stuck_state();
            }
            return; // 위험 상황이므로 일반 경로 생성 건너뜀
        }

        // 4. [경로 추종] 목표가 없거나 Stuck 상태면 정지
        if (!has_path_ || global_path_.poses.empty()) return;
        if (is_stuck_) {
            publish_stop_signal(); // Global Replan 기다리며 호버링
            return;
        }

        // 5. 로컬 목표 지점 설정 및 경로 생성
        geometry_msgs::msg::TransformStamped tf_map_to_base;
        try {
            tf_map_to_base = tf_buffer_->lookupTransform("base_link", "map", tf2::TimePointZero);
        } catch (tf2::TransformException &ex) { return; }

        // Global Path를 로컬 좌표로 변환하여 잘라내기
        nav_msgs::msg::Path local_segment = extract_local_segment(tf_map_to_base);
        geometry_msgs::msg::Point local_goal = get_smart_local_goal(local_segment);

        // 6. 경로 유효성 검사 및 A* 실행
        if (check_path_collision(local_segment)) {
            // 기존 Global Path가 안전하면 그대로 주행
            local_path_pub_->publish(local_segment);
            
            // [추가] 정상 주행이므로 후진 모드 끄기
            publish_reverse_signal(false);
        } else {
            // 막혔으면 A*로 우회 경로 탐색
            run_local_astar(local_goal);
        }
    }

    // --- Helper Logic ---

    // [중요] 드론 현재 위치(맵 중앙)가 점유되었는지 확인
    bool is_drone_in_danger() {
        int cx = grid_dim_x_ / 2;
        int cy = grid_dim_y_ / 2;
        int cz = grid_dim_z_ / 2;
        // 인덱스가 유효하고, 해당 셀이 True(점유됨)라면 위험
        return is_valid_index(cx, cy, cz) && local_occupancy_[get_index(cx, cy, cz)];
    }

    // [중요] 드론 후방 1.5m 지점이 안전한지 확인
    bool is_rear_safe() {
        // x축 뒤쪽 (-1.5m)
        int back_x = (grid_dim_x_ / 2) - (int)(1.5 / grid_res_);
        int back_y = grid_dim_y_ / 2;
        int back_z = grid_dim_z_ / 2;

        // 맵 밖으로 나가거나, 장애물이 있으면 false
        if (!is_valid_index(back_x, back_y, back_z)) return false; 
        return !local_occupancy_[get_index(back_x, back_y, back_z)];
    }

    void start_escape_maneuver() {
        RCLCPP_WARN(this->get_logger(), "Initiating BACKWARD ESCAPE maneuver!");
        is_escaping_ = true;
        escape_start_time_ = this->now();
        publish_backward_path();
    }

    void publish_backward_path() {
        nav_msgs::msg::Path escape_path;
        escape_path.header.frame_id = "base_link";
        escape_path.header.stamp = this->now();
        
        geometry_msgs::msg::PoseStamped p;
        p.pose.position.x = -1.0; // 뒤로 1m 이동 목표
        p.pose.position.y = 0.0;
        p.pose.position.z = 0.0;
        p.pose.orientation.w = 1.0;
        
        escape_path.poses.push_back(p);
        local_path_pub_->publish(escape_path);

        // [핵심] 후진 모드 ON 신호 전송 -> Control Node가 Yaw를 고정함
        publish_reverse_signal(true);
    }

    void handle_stuck_state() {
        // 처음 갇혔을 때만 Replan 요청
        if (!is_stuck_) {
            is_stuck_ = true;
            if (replan_client_->service_is_ready()) {
                RCLCPP_WARN(this->get_logger(), "Requesting Global Planner for a new path...");
                auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
                replan_client_->async_send_request(request);
            }
        }
        publish_stop_signal();
    }

    void publish_stop_signal() {
        nav_msgs::msg::Path empty_path;
        empty_path.header.frame_id = "base_link";
        empty_path.header.stamp = this->now();
        local_path_pub_->publish(empty_path); // 빈 경로 = 호버링
        
        // [추가] 정지 상태는 후진 모드가 아님 (제자리 회전 가능하게)
        publish_reverse_signal(false);
    }

    // A* 알고리즘
    void run_local_astar(const geometry_msgs::msg::Point& goal) {
        int start_x = grid_dim_x_ / 2;
        int start_y = grid_dim_y_ / 2;
        int start_z = grid_dim_z_ / 2;

        int goal_x = (int)(goal.x / grid_res_) + (grid_dim_x_ / 2);
        int goal_y = (int)(goal.y / grid_res_) + (grid_dim_y_ / 2);
        int goal_z = (int)(goal.z / grid_res_) + (grid_dim_z_ / 2);
        
        goal_x = std::clamp(goal_x, 0, grid_dim_x_ - 1);
        goal_y = std::clamp(goal_y, 0, grid_dim_y_ - 1);
        goal_z = std::clamp(goal_z, 0, grid_dim_z_ - 1);

        if (local_occupancy_[get_index(start_x, start_y, start_z)]) {
             handle_stuck_state();
             return;
        }

        std::priority_queue<std::shared_ptr<LocalNode>, std::vector<std::shared_ptr<LocalNode>>, CompareLocalNode> open_list;
        std::vector<bool> visited(local_occupancy_.size(), false);

        auto start_node = std::make_shared<LocalNode>();
        start_node->x = start_x; start_node->y = start_y; start_node->z = start_z;
        start_node->g = 0; start_node->h = std::hypot(goal_x - start_x, goal_y - start_y, goal_z - start_z);
        open_list.push(start_node);

        std::shared_ptr<LocalNode> final_node = nullptr;
        int max_iter = 5000; 

        while (!open_list.empty() && max_iter-- > 0) {
            auto current = open_list.top(); open_list.pop();
            if (current->x == goal_x && current->y == goal_y && current->z == goal_z) {
                final_node = current; break;
            }
            int idx = get_index(current->x, current->y, current->z);
            if (visited[idx]) continue;
            visited[idx] = true;

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        int nx = current->x + dx; int ny = current->y + dy; int nz = current->z + dz;
                        if (is_valid_index(nx, ny, nz)) {
                            if (local_occupancy_[get_index(nx, ny, nz)]) continue;
                            auto neighbor = std::make_shared<LocalNode>();
                            neighbor->x = nx; neighbor->y = ny; neighbor->z = nz;
                            neighbor->parent = current;
                            neighbor->g = current->g + std::sqrt(dx*dx + dy*dy + dz*dz);
                            neighbor->h = std::hypot(goal_x - nx, goal_y - ny, goal_z - nz);
                            open_list.push(neighbor);
                        }
                    }
                }
            }
        }

        if (final_node) {
            nav_msgs::msg::Path path_msg;
            path_msg.header.frame_id = "base_link"; 
            path_msg.header.stamp = this->now();
            auto trace = final_node;
            std::vector<geometry_msgs::msg::PoseStamped> raw_poses;
            while (trace) {
                geometry_msgs::msg::PoseStamped p;
                p.pose.position.x = (trace->x - start_x) * grid_res_;
                p.pose.position.y = (trace->y - start_y) * grid_res_;
                p.pose.position.z = (trace->z - start_z) * grid_res_;
                p.pose.orientation.w = 1.0; 
                raw_poses.push_back(p);
                trace = trace->parent;
            }
            std::reverse(raw_poses.begin(), raw_poses.end());
            
            // Pruning (너무 가까운 점 제거)
            for (const auto& pose : raw_poses) {
                if (std::hypot(pose.pose.position.x, pose.pose.position.y) > 0.5) 
                    path_msg.poses.push_back(pose);
            }
            if (path_msg.poses.empty() && !raw_poses.empty()) path_msg.poses.push_back(raw_poses.back());
            
            local_path_pub_->publish(path_msg);
            
            // [추가] A* 경로는 전진(Turn & Go)이므로 후진 모드 끄기
            publish_reverse_signal(false);

        } else {
            // A* 실패 -> 경로 없음 -> Global Replan 요청
            RCLCPP_WARN(this->get_logger(), "A* Failed. Requesting Replan.");
            handle_stuck_state();
        }
    }

    // --- Utility Functions ---
    void update_local_map() {
        local_occupancy_.assign(grid_dim_x_ * grid_dim_y_ * grid_dim_z_, false);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*latest_scan_, *cloud);
        
        // Voxel Grid Filter
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*cloud_filtered);

        int inflation = std::ceil(safety_distance_ / grid_res_);
        
        for (const auto& pt : cloud_filtered->points) {
            // PointCloud가 base_link 기준이라고 가정
            int idx_x = (int)(pt.x / grid_res_) + (grid_dim_x_ / 2);
            int idx_y = (int)(pt.y / grid_res_) + (grid_dim_y_ / 2);
            int idx_z = (int)(pt.z / grid_res_) + (grid_dim_z_ / 2);

            for (int ix = -inflation; ix <= inflation; ++ix) {
                for (int iy = -inflation; iy <= inflation; ++iy) {
                    for (int iz = -inflation; iz <= inflation; ++iz) {
                        double d2 = (ix*ix + iy*iy + iz*iz) * (grid_res_*grid_res_);
                        if (d2 > safety_distance_ * safety_distance_) continue;
                        
                        int nx = idx_x + ix; int ny = idx_y + iy; int nz = idx_z + iz;
                        if (is_valid_index(nx, ny, nz)) 
                            local_occupancy_[get_index(nx, ny, nz)] = true;
                    }
                }
            }
        }
        // 디버깅용 맵 발행
        std_msgs::msg::Float32MultiArray debug_msg;
        for(auto v : local_occupancy_) debug_msg.data.push_back(v ? 1.0 : 0.0);
        local_map_pub_->publish(debug_msg);
    }

    nav_msgs::msg::Path extract_local_segment(const geometry_msgs::msg::TransformStamped& tf) {
        nav_msgs::msg::Path local_path;
        local_path.header.frame_id = "base_link";
        local_path.header.stamp = this->now();
        for (const auto& pose : global_path_.poses) {
            geometry_msgs::msg::PoseStamped p_base;
            try { tf2::doTransform(pose, p_base, tf); } catch(...) { continue; }
            if (p_base.pose.position.x < -0.5) continue; // 이미 지나간 점
            
            // 로컬 그리드 범위를 벗어나면 중단
            if (std::abs(p_base.pose.position.x) > (grid_dim_x_ * grid_res_ / 2.0) || 
                std::abs(p_base.pose.position.y) > (grid_dim_y_ * grid_res_ / 2.0)) break;
            
            local_path.poses.push_back(p_base);
        }
        return local_path;
    }

    geometry_msgs::msg::Point get_smart_local_goal(const nav_msgs::msg::Path& local_segment) {
        if (!local_segment.poses.empty()) return local_segment.poses.back().pose.position;
        return geometry_msgs::msg::Point();
    }

    bool check_path_collision(const nav_msgs::msg::Path& path) {
        if (path.poses.empty()) return false;
        for (const auto& pose : path.poses) {
            int ix = (int)(pose.pose.position.x / grid_res_) + (grid_dim_x_ / 2);
            int iy = (int)(pose.pose.position.y / grid_res_) + (grid_dim_y_ / 2);
            int iz = (int)(pose.pose.position.z / grid_res_) + (grid_dim_z_ / 2);
            if (!is_valid_index(ix, iy, iz)) continue;
            if (local_occupancy_[get_index(ix, iy, iz)]) return false; 
        }
        return true;
    }

    inline int get_index(int x, int y, int z) { return x + y * grid_dim_x_ + z * grid_dim_x_ * grid_dim_y_; }
    inline bool is_valid_index(int x, int y, int z) { return x >= 0 && x < grid_dim_x_ && y >= 0 && y < grid_dim_y_ && z >= 0 && z < grid_dim_z_; }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalPlanner>());
    rclcpp::shutdown();
    return 0;
}