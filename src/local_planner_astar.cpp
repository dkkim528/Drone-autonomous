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
#include <std_msgs/msg/bool.hpp> 

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp> 

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/transforms.hpp> 

using std::placeholders::_1;
using namespace std::chrono_literals;

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
        reverse_mode_pub_ = this->create_publisher<std_msgs::msg::Bool>("/local_planner/reverse_mode", 10);
        replan_client_ = this->create_client<std_srvs::srv::Trigger>("replan_path");

        timer_ = this->create_wall_timer(50ms, std::bind(&LocalPlanner::control_loop, this));
    }

private:
    const double grid_res_ = 0.2;     
    const int grid_dim_x_ = 40;       
    const int grid_dim_y_ = 40;       
    // [설정] 위로 갈 "수"도 있게 공간만 확보 (강제 아님)
    const int grid_dim_z_ = 50;       
    
    double safety_distance_;

    nav_msgs::msg::Path global_path_;
    nav_msgs::msg::Path last_safe_path_global_; 
    bool has_last_path_ = false;

    sensor_msgs::msg::PointCloud2::SharedPtr latest_scan_;
    bool has_path_ = false;
    
    bool is_stuck_ = false;       
    bool is_escaping_ = false;    
    rclcpp::Time escape_start_time_;

    std::vector<bool> local_occupancy_; 

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr local_map_pub_;
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
        has_last_path_ = false; 
        RCLCPP_INFO(this->get_logger(), "New Global Path Received.");
    }

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        latest_scan_ = msg;
    }

    void publish_reverse_signal(bool is_reverse) {
        std_msgs::msg::Bool msg;
        msg.data = is_reverse;
        reverse_mode_pub_->publish(msg);
    }

    // ==================================================================================
    // Control Loop
    // ==================================================================================
    void control_loop() {
        if (!latest_scan_) return;

        update_local_map(); 
        
        if (is_escaping_) {
            if ((this->now() - escape_start_time_).seconds() > 2.0) {
                is_escaping_ = false; 
                has_last_path_ = false; 
                RCLCPP_INFO(this->get_logger(), "Escape maneuver finished.");
            } else {
                publish_backward_path(); 
                return;
            }
        }

        if (is_drone_in_danger()) {
            RCLCPP_WARN(this->get_logger(), "DANGER! Drone is inside safety radius.");
            if (is_rear_safe()) {
                start_escape_maneuver();
            } else {
                handle_stuck_state();
            }
            return; 
        }

        if (!has_path_ || global_path_.poses.empty()) return;
        if (is_stuck_) {
            publish_stop_signal(); 
            return;
        }

        geometry_msgs::msg::TransformStamped tf_map_to_base;
        try {
            tf_map_to_base = tf_buffer_->lookupTransform("base_link", "map", tf2::TimePointZero);
        } catch (tf2::TransformException &ex) { return; }

        // [Method B] Path Reuse (Hysteresis)
        if (has_last_path_) {
            nav_msgs::msg::Path reused_path_local;
            reused_path_local.header.frame_id = "base_link";
            reused_path_local.header.stamp = this->now();

            bool transform_success = true;
            for (const auto& pose : last_safe_path_global_.poses) {
                geometry_msgs::msg::PoseStamped p_base;
                try {
                    tf2::doTransform(pose, p_base, tf_map_to_base);
                    reused_path_local.poses.push_back(p_base);
                } catch(...) { transform_success = false; break; }
            }

            if (transform_success && !reused_path_local.poses.empty()) {
                nav_msgs::msg::Path pruned_path = prune_passed_waypoints(reused_path_local);
                if (pruned_path.poses.size() > 5 && check_path_collision(pruned_path)) {
                    local_path_pub_->publish(pruned_path);
                    publish_reverse_signal(false);
                    return; 
                }
            }
        }

        nav_msgs::msg::Path local_segment = extract_local_segment(tf_map_to_base);
        geometry_msgs::msg::Point local_goal = get_smart_local_goal(local_segment);

        if (check_path_collision(local_segment)) {
            nav_msgs::msg::Path smoothed = smooth_path(local_segment); 
            local_path_pub_->publish(smoothed);
            publish_reverse_signal(false);
            save_path_for_reuse(smoothed); 
        } else {
            run_local_astar(local_goal);
        }
    }

    // --- A* Algorithm (Standard Metric + Method A Bias) ---

    void run_local_astar(const geometry_msgs::msg::Point& goal) {
        int start_x = grid_dim_x_ / 2;
        int start_y = grid_dim_y_ / 2;
        int start_z = grid_dim_z_ / 2;

        int goal_x = std::clamp((int)(goal.x / grid_res_) + (grid_dim_x_ / 2), 0, grid_dim_x_ - 1);
        int goal_y = std::clamp((int)(goal.y / grid_res_) + (grid_dim_y_ / 2), 0, grid_dim_y_ - 1);
        int goal_z = std::clamp((int)(goal.z / grid_res_) + (grid_dim_z_ / 2), 0, grid_dim_z_ - 1);

        if (local_occupancy_[get_index(start_x, start_y, start_z)]) {
             handle_stuck_state(); return;
        }

        std::priority_queue<std::shared_ptr<LocalNode>, std::vector<std::shared_ptr<LocalNode>>, CompareLocalNode> open_list;
        std::vector<bool> visited(local_occupancy_.size(), false);

        auto start_node = std::make_shared<LocalNode>();
        start_node->x = start_x; start_node->y = start_y; start_node->z = start_z;
        start_node->g = 0; 
        
        // [수정] 순수 유클리드 거리 (X, Y, Z 공평)
        double dist = std::hypot(goal_x - start_x, goal_y - start_y, goal_z - start_z);
        start_node->h = dist * 1.05; // Bias만 유지
        
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
                            
                            // [수정 완료] Z축 인센티브 제거.
                            // 이제 드론은 "수학적으로 가장 짧은 우회로"를 선택합니다.
                            // 옆으로 2m 돌아가기 vs 위로 3m 넘어가기 -> 옆을 선택.
                            // 옆이 꽉 막힘 vs 위로 3m 넘어가기 -> 위를 선택.
                            double h_val = std::hypot(goal_x - nx, goal_y - ny, goal_z - nz);
                            neighbor->h = h_val * 1.05; // [Method A] Bias는 유지

                            open_list.push(neighbor);
                        }
                    }
                }
            }
        }

        if (final_node) {
            std::vector<geometry_msgs::msg::PoseStamped> raw_poses;
            auto trace = final_node;
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
            
            nav_msgs::msg::Path raw_path;
            raw_path.header.frame_id = "base_link";
            raw_path.header.stamp = this->now();
            raw_path.poses = raw_poses;

            nav_msgs::msg::Path smoothed = smooth_path(raw_path);
            
            local_path_pub_->publish(smoothed);
            publish_reverse_signal(false);
            
            save_path_for_reuse(smoothed);

        } else {
            RCLCPP_WARN(this->get_logger(), "A* Failed.");
            handle_stuck_state();
        }
    }

    // --- Helper & Smoothing Functions ---

    nav_msgs::msg::Path smooth_path(const nav_msgs::msg::Path& raw_path) {
        if (raw_path.poses.size() <= 2) return raw_path;
        nav_msgs::msg::Path smooth = raw_path;
        
        for (size_t i = 1; i < raw_path.poses.size() - 1; ++i) {
            double sx = raw_path.poses[i-1].pose.position.x + raw_path.poses[i].pose.position.x + raw_path.poses[i+1].pose.position.x;
            double sy = raw_path.poses[i-1].pose.position.y + raw_path.poses[i].pose.position.y + raw_path.poses[i+1].pose.position.y;
            double sz = raw_path.poses[i-1].pose.position.z + raw_path.poses[i].pose.position.z + raw_path.poses[i+1].pose.position.z;
            
            smooth.poses[i].pose.position.x = sx / 3.0;
            smooth.poses[i].pose.position.y = sy / 3.0;
            smooth.poses[i].pose.position.z = sz / 3.0;
        }
        return smooth;
    }

    void save_path_for_reuse(const nav_msgs::msg::Path& local_path) {
        try {
            geometry_msgs::msg::TransformStamped tf_base_to_map;
            tf_base_to_map = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
            
            last_safe_path_global_.poses.clear();
            last_safe_path_global_.header.frame_id = "map";
            last_safe_path_global_.header.stamp = this->now();

            for(const auto& pose : local_path.poses) {
                geometry_msgs::msg::PoseStamped p_map;
                tf2::doTransform(pose, p_map, tf_base_to_map);
                last_safe_path_global_.poses.push_back(p_map);
            }
            has_last_path_ = true;
        } catch (...) { has_last_path_ = false; }
    }

    nav_msgs::msg::Path prune_passed_waypoints(const nav_msgs::msg::Path& path) {
        nav_msgs::msg::Path new_path;
        new_path.header = path.header;
        bool keep = false;
        for (const auto& pose : path.poses) {
            if (!keep) {
                if (pose.pose.position.x > 0.0 && std::hypot(pose.pose.position.x, pose.pose.position.y) > 0.2) {
                    keep = true;
                    new_path.poses.push_back(pose);
                }
            } else {
                new_path.poses.push_back(pose);
            }
        }
        return new_path;
    }

    bool is_drone_in_danger() {
        int cx = grid_dim_x_ / 2; int cy = grid_dim_y_ / 2; int cz = grid_dim_z_ / 2;
        return is_valid_index(cx, cy, cz) && local_occupancy_[get_index(cx, cy, cz)];
    }
    bool is_rear_safe() {
        int back_x = (grid_dim_x_ / 2) - (int)(1.5 / grid_res_);
        int back_y = grid_dim_y_ / 2; int back_z = grid_dim_z_ / 2;
        if (!is_valid_index(back_x, back_y, back_z)) return false; 
        return !local_occupancy_[get_index(back_x, back_y, back_z)];
    }
    void start_escape_maneuver() {
        is_escaping_ = true; escape_start_time_ = this->now();
        publish_backward_path();
    }
    void publish_backward_path() {
        nav_msgs::msg::Path escape_path;
        escape_path.header.frame_id = "base_link"; escape_path.header.stamp = this->now();
        geometry_msgs::msg::PoseStamped p;
        p.pose.position.x = -1.0; p.pose.orientation.w = 1.0;
        escape_path.poses.push_back(p);
        local_path_pub_->publish(escape_path); publish_reverse_signal(true);
    }
    void handle_stuck_state() {
        if (!is_stuck_) {
            is_stuck_ = true;
            if (replan_client_->service_is_ready()) {
                auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
                replan_client_->async_send_request(request);
            }
        }
        publish_stop_signal();
    }
    void publish_stop_signal() {
        nav_msgs::msg::Path hover_path;
        hover_path.header.frame_id = "base_link"; hover_path.header.stamp = this->now();
        geometry_msgs::msg::PoseStamped p; p.pose.orientation.w = 1.0;
        for(int i=0; i<10; i++) hover_path.poses.push_back(p);
        local_path_pub_->publish(hover_path); publish_reverse_signal(false);
    }
    void update_local_map() {
        local_occupancy_.assign(grid_dim_x_ * grid_dim_y_ * grid_dim_z_, false);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*latest_scan_, *cloud_raw);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        try { pcl_ros::transformPointCloud("base_link", *cloud_raw, *cloud, *tf_buffer_); } 
        catch (...) { return; }
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud); sor.setLeafSize(0.1f, 0.1f, 0.1f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*cloud_filtered);
        int inflation = std::ceil(safety_distance_ / grid_res_);
        for (const auto& pt : cloud_filtered->points) {
            int idx_x = (int)(pt.x / grid_res_) + (grid_dim_x_ / 2);
            int idx_y = (int)(pt.y / grid_res_) + (grid_dim_y_ / 2);
            int idx_z = (int)(pt.z / grid_res_) + (grid_dim_z_ / 2);
            for (int ix = -inflation; ix <= inflation; ++ix) {
                for (int iy = -inflation; iy <= inflation; ++iy) {
                    for (int iz = -inflation; iz <= inflation; ++iz) {
                        double d2 = (ix*ix + iy*iy + iz*iz) * (grid_res_*grid_res_);
                        if (d2 > safety_distance_ * safety_distance_) continue;
                        int nx = idx_x + ix; int ny = idx_y + iy; int nz = idx_z + iz;
                        if (is_valid_index(nx, ny, nz)) local_occupancy_[get_index(nx, ny, nz)] = true;
                    }
                }
            }
        }
        std_msgs::msg::Float32MultiArray debug_msg;
        for(auto v : local_occupancy_) debug_msg.data.push_back(v ? 1.0 : 0.0);
        local_map_pub_->publish(debug_msg);
    }
    nav_msgs::msg::Path extract_local_segment(const geometry_msgs::msg::TransformStamped& tf) {
        nav_msgs::msg::Path local_path;
        local_path.header.frame_id = "base_link"; local_path.header.stamp = this->now();
        for (const auto& pose : global_path_.poses) {
            geometry_msgs::msg::PoseStamped p_base;
            try { tf2::doTransform(pose, p_base, tf); } catch(...) { continue; }
            if (p_base.pose.position.x < -0.5) continue; 
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