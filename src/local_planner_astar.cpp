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

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp> 

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

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
        // 1. 파라미터
        this->declare_parameter("safety_distance", 0.4);
        safety_distance_ = this->get_parameter("safety_distance").as_double();

        RCLCPP_INFO(this->get_logger(), "Local Planner Mode: Path Generation Only (No cmd_vel)");
        RCLCPP_INFO(this->get_logger(), "Grid Size: 40x40, Safety Dist: %.2f m", safety_distance_);

        // 2. TF 초기화
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 3. 구독
        global_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/global_path", 10, std::bind(&LocalPlanner::path_callback, this, _1));
        
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar/points", rclcpp::SensorDataQoS(), std::bind(&LocalPlanner::lidar_callback, this, _1));

        // 4. 발행 (cmd_vel 삭제됨)
        // ★ 중요: Controller 팀은 이 토픽을 구독해야 함
        local_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/local_path", 10);
        
        // 디버깅 및 시각화용
        obstacle_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/local_obstacles", 10);
        local_map_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/local_grid_debug", 10);

        // 5. 서비스 클라이언트 (길 막혔을 때 Global Replan 요청)
        replan_client_ = this->create_client<std_srvs::srv::Trigger>("replan_path");

        // 6. 타이머 (20Hz)
        timer_ = this->create_wall_timer(50ms, std::bind(&LocalPlanner::control_loop, this));
    }

private:
    // 맵 크기 (40x40 -> 약 8m 범위 커버)
    const double grid_res_ = 0.2;     
    const int grid_dim_x_ = 40;       
    const int grid_dim_y_ = 40;       
    const int grid_dim_z_ = 10;       
    
    double safety_distance_;

    nav_msgs::msg::Path global_path_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_scan_;
    bool has_path_ = false;
    bool is_stuck_ = false;

    std::vector<bool> local_occupancy_; 

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    
    // cmd_vel Publisher 제거됨
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr local_map_pub_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr replan_client_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        global_path_ = *msg;
        has_path_ = true;
        is_stuck_ = false; 
    }

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        latest_scan_ = msg;
    }

    // --- 메인 루프 (경로 생성 전용) ---
    void control_loop() {
        if (!latest_scan_) return;

        // 1. 맵 업데이트
        update_local_map();      
        publish_local_map_raw(); 

        if (!has_path_ || global_path_.poses.empty()) return;

        // 2. TF 조회
        geometry_msgs::msg::TransformStamped tf_map_to_base;
        try {
            tf_map_to_base = tf_buffer_->lookupTransform("base_link", "map", tf2::TimePointZero);
        } catch (tf2::TransformException &ex) { return; }

        // 3. Global Path를 Local Frame으로 변환 및 자르기
        nav_msgs::msg::Path local_segment;
        local_segment.header.frame_id = "base_link";
        local_segment.header.stamp = this->now();

        for (const auto& pose : global_path_.poses) {
            geometry_msgs::msg::PoseStamped p_base;
            try { tf2::doTransform(pose, p_base, tf_map_to_base); } catch(...) { continue; }
            
            if (p_base.pose.position.x < -0.5) continue; // 내 뒤는 무시

            // 그리드 범위 밖이면 루프 중단 (여기까지만 Local Segment로 사용)
            if (std::abs(p_base.pose.position.x) > (grid_dim_x_ * grid_res_ / 2.0) || 
                std::abs(p_base.pose.position.y) > (grid_dim_y_ * grid_res_ / 2.0)) {
                break;
            }
            local_segment.poses.push_back(p_base);
        }

        // 4. Smart Goal 선정 (장애물 없는 목표점 찾기)
        geometry_msgs::msg::Point local_goal = get_smart_local_goal(local_segment, tf_map_to_base);

        // 5. 경로 결정 및 발행 (Controller 팀에게 전달)
        if (check_global_path_collision(local_segment)) {
            // Case A: 원래 경로가 안전함 -> Global Path 조각 그대로 발행
            local_path_pub_->publish(local_segment);
            is_stuck_ = false;
        } 
        else {
            // Case B: 장애물 발견 -> A* 회피 경로 계산 및 발행
            run_local_astar(local_goal);
        }
    }

    // --- 장애물 없는 안전한 목표점 찾기 ---
    geometry_msgs::msg::Point get_smart_local_goal(
        const nav_msgs::msg::Path& local_segment, 
        const geometry_msgs::msg::TransformStamped& tf_map_to_base) 
    {
        geometry_msgs::msg::Point valid_goal;
        bool found = false;

        for (const auto& pose : global_path_.poses) {
            geometry_msgs::msg::PoseStamped p_base;
            try { tf2::doTransform(pose, p_base, tf_map_to_base); } catch (...) { continue; }

            double x = p_base.pose.position.x;
            double y = p_base.pose.position.y;
            double z = p_base.pose.position.z;
            double dist = std::hypot(x, y);

            if (dist < 1.0) continue; // 너무 가까운 점 무시

            // 로컬 맵 범위 체크
            if (std::abs(x) > (grid_dim_x_ * grid_res_ / 2.0) - 0.5 ||
                std::abs(y) > (grid_dim_y_ * grid_res_ / 2.0) - 0.5) {
                break; 
            }

            // 장애물 검사
            int idx_x = (int)(x / grid_res_) + (grid_dim_x_ / 2);
            int idx_y = (int)(y / grid_res_) + (grid_dim_y_ / 2);
            int idx_z = (int)(z / grid_res_) + (grid_dim_z_ / 2);

            if (is_valid_index(idx_x, idx_y, idx_z)) {
                // 비어있는(false) 곳이면 목표 후보
                if (!local_occupancy_[get_index(idx_x, idx_y, idx_z)]) {
                    valid_goal = p_base.pose.position;
                    found = true;
                    if (dist > 3.5) break; // 적당히 멀면 확정
                }
            }
        }

        if (!found) {
            if (!local_segment.poses.empty()) return local_segment.poses.back().pose.position;
            geometry_msgs::msg::Point p; p.x=0; p.y=0; p.z=0; return p;
        }
        return valid_goal;
    }

    bool check_global_path_collision(const nav_msgs::msg::Path& path) {
        if (path.poses.empty()) return false;
        for (const auto& pose : path.poses) {
            int idx_x = (int)(pose.pose.position.x / grid_res_) + (grid_dim_x_ / 2);
            int idx_y = (int)(pose.pose.position.y / grid_res_) + (grid_dim_y_ / 2);
            int idx_z = (int)(pose.pose.position.z / grid_res_) + (grid_dim_z_ / 2);

            if (!is_valid_index(idx_x, idx_y, idx_z)) continue;
            if (local_occupancy_[get_index(idx_x, idx_y, idx_z)]) return false; // 충돌
        }
        return true; 
    }

    void update_local_map() {
        local_occupancy_.assign(grid_dim_x_ * grid_dim_y_ * grid_dim_z_, false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*latest_scan_, *cloud);

        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*cloud_filtered);

        int inflation_cells = std::ceil(safety_distance_ / grid_res_);

        for (const auto& pt : cloud_filtered->points) {
            int idx_x = (int)(pt.x / grid_res_) + (grid_dim_x_ / 2);
            int idx_y = (int)(pt.y / grid_res_) + (grid_dim_y_ / 2);
            int idx_z = (int)(pt.z / grid_res_) + (grid_dim_z_ / 2);

            for (int ix = -inflation_cells; ix <= inflation_cells; ++ix) {
                for (int iy = -inflation_cells; iy <= inflation_cells; ++iy) {
                    for (int iz = -inflation_cells; iz <= inflation_cells; ++iz) {
                        double dist_sq = (ix*ix + iy*iy + iz*iz) * (grid_res_ * grid_res_);
                        if (dist_sq > safety_distance_ * safety_distance_) continue;

                        int nx = idx_x + ix;
                        int ny = idx_y + iy;
                        int nz = idx_z + iz;
                        if (is_valid_index(nx, ny, nz)) {
                            local_occupancy_[get_index(nx, ny, nz)] = true;
                        }
                    }
                }
            }
        }
    }

    void publish_local_map_raw() {
        std_msgs::msg::Float32MultiArray msg;
        for (const auto& occupied : local_occupancy_) {
            msg.data.push_back(occupied ? 1.0 : 0.0);
        }
        if (!msg.data.empty()) local_map_pub_->publish(msg);
    }

    // --- A* 실행 및 경로 발행 (반환값 void) ---
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
             handle_stuck_state(); // 시작점이 막힘
             return;
        }

        std::priority_queue<std::shared_ptr<LocalNode>, std::vector<std::shared_ptr<LocalNode>>, CompareLocalNode> open_list;
        std::vector<bool> visited(local_occupancy_.size(), false);

        auto start_node = std::make_shared<LocalNode>();
        start_node->x = start_x; start_node->y = start_y; start_node->z = start_z;
        start_node->g = 0; start_node->h = std::hypot(goal_x - start_x, goal_y - start_y, goal_z - start_z);
        open_list.push(start_node);

        std::shared_ptr<LocalNode> final_node = nullptr;
        int max_iter = 4000; 

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
                            int n_idx = get_index(nx, ny, nz);
                            if (local_occupancy_[n_idx]) continue; 
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
            nav_msgs::msg::Path gui_path;
            gui_path.header.frame_id = "base_link";
            gui_path.header.stamp = this->now();

            auto trace = final_node;
            while (trace) {
                geometry_msgs::msg::PoseStamped pose;
                pose.pose.position.x = (trace->x - start_x) * grid_res_;
                pose.pose.position.y = (trace->y - start_y) * grid_res_;
                pose.pose.position.z = (trace->z - start_z) * grid_res_;
                pose.pose.orientation.w = 1.0;
                gui_path.poses.push_back(pose);
                trace = trace->parent;
            }
            // ★ A* 결과 경로를 발행 (Controller가 따라갈 경로)
            local_path_pub_->publish(gui_path);
            is_stuck_ = false;
        } else {
            handle_stuck_state(); // 경로 찾기 실패
        }
    }

    void handle_stuck_state() {
        if (is_stuck_) return;
        is_stuck_ = true;
        RCLCPP_WARN(this->get_logger(), "STUCK! No valid path found. Requesting Re-plan.");
        
        // Controller에게 "갈 길 없음"을 알리기 위해 빈 경로 발행 (선택 사항)
        // nav_msgs::msg::Path empty_path;
        // empty_path.header.frame_id = "base_link";
        // empty_path.header.stamp = this->now();
        // local_path_pub_->publish(empty_path);

        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        replan_client_->async_send_request(request);
    }

    inline int get_index(int x, int y, int z) {
        return x + y * grid_dim_x_ + z * grid_dim_x_ * grid_dim_y_;
    }
    inline bool is_valid_index(int x, int y, int z) {
        return x >= 0 && x < grid_dim_x_ && y >= 0 && y < grid_dim_y_ && z >= 0 && z < grid_dim_z_;
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalPlanner>());
    rclcpp::shutdown();
    return 0;
}