#include <memory>
#include <vector>
#include <cmath>
#include <queue>
#include <set>
#include <tuple>
#include <algorithm>
#include <mutex>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <dynamicEDT3D/dynamicEDTOctomap.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <interactive_markers/interactive_marker_server.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>

using std::placeholders::_1;
using std::placeholders::_2;
using namespace visualization_msgs::msg; 

// A* 노드 구조체
struct AStarNode {
    int x, y, z;
    double g_cost, h_cost;
    std::shared_ptr<AStarNode> parent;
    double f_cost() const { return g_cost + h_cost; }
};

struct CompareNode {
    bool operator()(const std::shared_ptr<AStarNode>& a, const std::shared_ptr<AStarNode>& b) {
        return a->f_cost() > b->f_cost();
    }
};

class GlobalPlanner : public rclcpp::Node
{
public:
  GlobalPlanner() : Node("global_planner_node")
  {
    // [파라미터 설정]
    // safe_radius: 로컬 플래너의 회피 거리(0.9m)보다 커야 Stuck 현상을 막음 (권장: 1.0 ~ 1.2m)
    this->declare_parameter("safe_radius", 1.2);    
    this->declare_parameter("resolution", 0.5);     // 맵 해상도
    this->declare_parameter("min_z", 0.5);          // 드론 비행 최소 높이 (바닥 회피)
    this->declare_parameter("max_z", 2.5);          // 드론 비행 최대 높이 (천장 회피)
    this->declare_parameter("map_update_freq", 2.0); // 맵 업데이트(EDT 계산) 빈도 (Hz)

    safe_radius_ = this->get_parameter("safe_radius").as_double();
    resolution_ = this->get_parameter("resolution").as_double();
    min_z_ = this->get_parameter("min_z").as_double();
    max_z_ = this->get_parameter("max_z").as_double();
    double update_freq = this->get_parameter("map_update_freq").as_double();

    // 1. TF Listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 2. 맵 데이터 구독
    rclcpp::QoS map_qos(1);
    map_qos.transient_local(); map_qos.reliable();
    
    // 초기 맵 (Octomap) 구독
    map_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_binary", map_qos, std::bind(&GlobalPlanner::octomap_callback, this, _1));

    // 실시간 장애물 (Local PointCloud) 구독
    obstacle_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/local_obstacles", 10, std::bind(&GlobalPlanner::obstacle_callback, this, _1));

    // 3. 퍼블리셔 & 서비스
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("global_path", 10);
    
    replan_service_ = this->create_service<std_srvs::srv::Trigger>(
        "replan_path", std::bind(&GlobalPlanner::replan_callback, this, _1, _2));

    // 4. Interactive Marker (Rviz 목표 지점 설정용)
    server_ = std::make_unique<interactive_markers::InteractiveMarkerServer>("goal_marker_server", this);

    // [최적화] 맵 업데이트 타이머 (센서 콜백에서 분리하여 CPU 부하 감소)
    int period_ms = static_cast<int>(1000.0 / update_freq);
    map_update_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(period_ms), std::bind(&GlobalPlanner::update_map_event, this));

    RCLCPP_INFO(this->get_logger(), "Global Planner Started. Safe Radius: %.2fm, Height Limit: %.1f ~ %.1f m", safe_radius_, min_z_, max_z_);
  }

private:
  // --- 상태 변수 ---
  bool map_ready_ = false;
  bool map_needs_update_ = false; // 맵 변경 플래그
  bool goal_received_ = false;
  octomap::point3d goal_pos_;

  std::shared_ptr<octomap::OcTree> octree_;
  std::shared_ptr<DynamicEDTOctomap> distmap_;
  std::mutex map_mutex_; // 멀티스레드 충돌 방지

  // --- 맵 경계 변수 ---
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_; 

  // --- 파라미터 변수 ---
  double min_z_, max_z_; // 비행 허용 높이
  double safe_radius_;
  double resolution_;
  float max_dist_ = 2.5; // EDT 계산 최대 거리

  // ---------------------------------------------------------
  // [최적화 핵심 1] 타이머 기반 맵 업데이트
  // ---------------------------------------------------------
  void update_map_event() {
      if (!map_ready_ || !map_needs_update_) return;

      std::lock_guard<std::mutex> lock(map_mutex_);
      distmap_->update(); 
      map_needs_update_ = false;
  }

  // ---------------------------------------------------------
  // [최적화 핵심 2] 장애물 추가
  // ---------------------------------------------------------
  void obstacle_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
      if (!map_ready_) return;

      std::lock_guard<std::mutex> lock(map_mutex_);
      
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

      bool changed = false;
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
          octomap::point3d p(*iter_x, *iter_y, *iter_z);
          
          if(p.x() < x_min_ || p.x() > x_max_ || p.y() < y_min_ || p.y() > y_max_) continue;

          octree_->updateNode(p, true);
          changed = true;
      }

      if (changed) {
          map_needs_update_ = true;
      }
  }

  // ---------------------------------------------------------
  // A* 알고리즘 (수정됨: Goal 높이 보정 추가)
  // ---------------------------------------------------------
  void run_planning_algorithm(const octomap::point3d& start, const octomap::point3d& original_goal)
  {
      std::lock_guard<std::mutex> lock(map_mutex_);

      // ==========================================
      // [수정 포인트] 목표점 높이 강제 보정 (Clamping)
      // ==========================================
      octomap::point3d goal = original_goal;
      
      if (goal.z() > max_z_) goal.z() = max_z_;
      if (goal.z() < min_z_) goal.z() = min_z_;

      RCLCPP_INFO(this->get_logger(), "[Plan] Start A*... Safe Radius: %.2fm, Goal Adjusted Z: %.2f", safe_radius_, goal.z());

      // [안전 체크] 시작점이 비행 금지 구역(높이)일 경우 경고
      if (start.z() > max_z_ + 0.5 || start.z() < min_z_ - 0.5) {
           RCLCPP_WARN(this->get_logger(), "CAUTION: Drone is currently outside flight altitude limits!");
      }

      // 시작점이 장애물인지 체크
      float start_dist = distmap_->getDistance(start);
      if (start_dist < safe_radius_ && start_dist >= 0) {
          RCLCPP_WARN(this->get_logger(), "Start is too close to obstacle (%.2fm). Planning anyway...", start_dist);
      }

      std::priority_queue<std::shared_ptr<AStarNode>, std::vector<std::shared_ptr<AStarNode>>, CompareNode> open_list;
      std::set<std::tuple<int, int, int>> closed_set;

      auto start_node = std::make_shared<AStarNode>();
      start_node->x = (int)(start.x()/resolution_);
      start_node->y = (int)(start.y()/resolution_);
      start_node->z = (int)(start.z()/resolution_);
      start_node->g_cost = 0; 
      start_node->h_cost = (goal - start).norm();
      open_list.push(start_node);

      // 목표점 인덱스 (보정된 goal 사용)
      int gx = (int)(goal.x()/resolution_);
      int gy = (int)(goal.y()/resolution_);
      int gz = (int)(goal.z()/resolution_);
      
      std::shared_ptr<AStarNode> final_node = nullptr;
      int iterations = 0;
      int max_iter = 100000; 

      while(!open_list.empty()) {
          auto current = open_list.top(); open_list.pop();
          
          if(++iterations > max_iter) { 
              RCLCPP_WARN(this->get_logger(), "A* Timeout! Iterations exceeded."); 
              break; 
          }

          if (std::abs(current->x - gx) <= 1 && std::abs(current->y - gy) <= 1 && std::abs(current->z - gz) <= 1) {
              final_node = current; break;
          }

          std::tuple<int,int,int> idx = {current->x, current->y, current->z};
          if(closed_set.count(idx)) continue;
          closed_set.insert(idx);

          for(int dx=-1; dx<=1; dx++) {
             for(int dy=-1; dy<=1; dy++) {
                for(int dz=-1; dz<=1; dz++) {
                   if(dx==0 && dy==0 && dz==0) continue;
                   
                   int nx = current->x+dx;
                   int ny = current->y+dy; 
                   int nz = current->z+dz;
                   
                   octomap::point3d w_pos(nx*resolution_, ny*resolution_, nz*resolution_);

                   // [제약 조건] 비행 높이 제한 (중간 경로도 제한)
                   if (w_pos.z() < min_z_ - 0.01) continue; 
                   if (w_pos.z() > max_z_ + 0.01) continue; 
                   
                   if(w_pos.x()<x_min_ || w_pos.x()>x_max_ || w_pos.y()<y_min_ || w_pos.y()>y_max_) continue;
                   
                   float d = distmap_->getDistance(w_pos);
                   if(d < safe_radius_ && d >= 0) continue; 

                   double step_dist = std::sqrt(dx*dx + dy*dy + dz*dz) * resolution_;

                   auto neighbor = std::make_shared<AStarNode>();
                   neighbor->x = nx; neighbor->y = ny; neighbor->z = nz;
                   neighbor->parent = current;
                   neighbor->g_cost = current->g_cost + step_dist;
                   
                   // Heuristic: 보정된 goal 기준
                   neighbor->h_cost = (w_pos - goal).norm() * 1.5; 
                   
                   open_list.push(neighbor);
                }
             }
          }
      }

      if(final_node) publish_path(final_node);
      else RCLCPP_WARN(this->get_logger(), "Failed to find global path!");
  }

  // --- 서비스 콜백 ---
  void replan_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
      if (!map_ready_ || !goal_received_) {
          response->success = false; response->message = "Not ready."; return;
      }
      octomap::point3d start_pos;
      if(get_drone_position(start_pos)) {
           run_planning_algorithm(start_pos, goal_pos_);
           response->success = true; response->message = "Replanned.";
      } else {
           response->success = false; response->message = "No TF.";
      }
  }

  // --- 초기 맵 로드 콜백 ---
  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (map_ready_) return; 
      
      octomap::AbstractOcTree* tree = octomap_msgs::binaryMsgToMap(*msg);
      if (!tree) return;
      
      octree_ = std::shared_ptr<octomap::OcTree>(dynamic_cast<octomap::OcTree*>(tree));
      if (octree_) {
          octree_->getMetricMin(x_min_, y_min_, z_min_); 
          octree_->getMetricMax(x_max_, y_max_, z_max_);
          
          distmap_ = std::make_shared<DynamicEDTOctomap>(max_dist_, octree_.get(), 
              octomap::point3d(x_min_, y_min_, z_min_), 
              octomap::point3d(x_max_, y_max_, z_max_), false);
          distmap_->update();
          
          map_ready_ = true;
          create_marker(octomap::point3d(0, 0, 1.0));
          RCLCPP_INFO(this->get_logger(), "Initial Map Processed. Bounds: Z[%.2f ~ %.2f]", z_min_, z_max_);
      }
  }

  void publish_path(std::shared_ptr<AStarNode> node)
  {
      nav_msgs::msg::Path path_msg;
      path_msg.header.frame_id = "map";
      path_msg.header.stamp = this->now();

      std::vector<geometry_msgs::msg::PoseStamped> poses;
      while(node) {
          geometry_msgs::msg::PoseStamped p;
          p.pose.position.x = node->x * resolution_;
          p.pose.position.y = node->y * resolution_;
          p.pose.position.z = node->z * resolution_;
          p.pose.orientation.w = 1.0;
          poses.push_back(p);
          node = node->parent;
      }
      std::reverse(poses.begin(), poses.end());
      path_msg.poses = poses;
      path_pub_->publish(path_msg);
  }

  bool get_drone_position(octomap::point3d &pos)
  {
    try {
      geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
      pos.x() = t.transform.translation.x;
      pos.y() = t.transform.translation.y;
      pos.z() = t.transform.translation.z;
      return true;
    } catch (...) { return false; }
  }

  void create_marker(const octomap::point3d& pos) {
      InteractiveMarker int_marker;
      int_marker.header.frame_id = "map";
      int_marker.name = "goal";
      int_marker.scale = 1.0;
      int_marker.pose.position.x = pos.x();
      int_marker.pose.position.y = pos.y();
      int_marker.pose.position.z = pos.z();

      InteractiveMarkerControl control;
      control.orientation.w = 1; control.orientation.x = 1; control.orientation.y = 0; control.orientation.z = 0;
      control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
      int_marker.controls.push_back(control);
      
      control.orientation.w = 1; control.orientation.x = 0; control.orientation.y = 1; control.orientation.z = 0;
      control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
      int_marker.controls.push_back(control);

      control.orientation.w = 1; control.orientation.x = 0; control.orientation.y = 0; control.orientation.z = 1;
      control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
      int_marker.controls.push_back(control);

      server_->insert(int_marker);
      server_->setCallback(int_marker.name, std::bind(&GlobalPlanner::process_feedback, this, _1));
      server_->applyChanges();
  }

  void process_feedback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback) {
      if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) {
          goal_pos_ = octomap::point3d(feedback->pose.position.x, feedback->pose.position.y, feedback->pose.position.z);
          goal_received_ = true;
      }
      if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_UP) {
          octomap::point3d start;
          if(get_drone_position(start)) run_planning_algorithm(start, goal_pos_);
      }
  }

  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr replan_service_;
  rclcpp::TimerBase::SharedPtr map_update_timer_; 

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<interactive_markers::InteractiveMarkerServer> server_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<GlobalPlanner>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}