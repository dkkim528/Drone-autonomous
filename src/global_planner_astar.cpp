#include <memory>
#include <vector>
#include <cmath>
#include <queue>
#include <set>
#include <tuple>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <dynamicEDT3D/dynamicEDTOctomap.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

// [중요] Interactive Marker 관련 헤더 및 메시지
#include <interactive_markers/interactive_marker_server.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>

using std::placeholders::_1;
using namespace visualization_msgs::msg; 

// A* Node 구조체
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
    // 1. TF Listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 2. 맵 데이터 구독
    rclcpp::QoS map_qos(1);
    map_qos.transient_local(); map_qos.reliable();
    map_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_binary", map_qos, std::bind(&GlobalPlanner::octomap_callback, this, _1));

    // 3. 경로 발행
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("global_path", 10);

    // 4. Interactive Marker Server (목표점 설정용)
    server_ = std::make_unique<interactive_markers::InteractiveMarkerServer>("goal_marker_server", this);

    RCLCPP_INFO(this->get_logger(), "=== Planner Ready: Waiting for Map & TF... ===");
  }

private:
  bool map_ready_ = false;
  octomap::point3d goal_pos_;

  std::shared_ptr<octomap::OcTree> octree_;
  std::shared_ptr<DynamicEDTOctomap> distmap_;
  
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  float max_dist_ = 2.0;   
  
  // [설정] 격자 크기 (속도 최적화를 위해 0.4~0.5 추천)
  double resolution_ = 0.5; 
  double safe_radius_ = 0.6;

  // Interactive Marker 생성
  void create_marker(const octomap::point3d& pos) {
      InteractiveMarker int_marker;
      int_marker.header.frame_id = "map";
      int_marker.header.stamp = this->now();
      int_marker.name = "goal_marker";
      int_marker.description = "Goal";
      int_marker.scale = 1.0;
      int_marker.pose.position.x = pos.x();
      int_marker.pose.position.y = pos.y();
      int_marker.pose.position.z = pos.z();

      InteractiveMarkerControl control;
      
      // X축
      control.orientation.w = 1; control.orientation.x = 1; control.orientation.y = 0; control.orientation.z = 0;
      control.name = "move_x"; control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
      int_marker.controls.push_back(control);

      // Z축
      control.orientation.w = 1; control.orientation.x = 0; control.orientation.y = 1; control.orientation.z = 0;
      control.name = "move_z"; control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
      int_marker.controls.push_back(control);

      // Y축
      control.orientation.w = 1; control.orientation.x = 0; control.orientation.y = 0; control.orientation.z = 1;
      control.name = "move_y"; control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
      int_marker.controls.push_back(control);

      server_->insert(int_marker);
      server_->setCallback(int_marker.name, std::bind(&GlobalPlanner::process_feedback, this, _1));
      server_->applyChanges();
  }

  void process_feedback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback)
  {
      if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) {
          goal_pos_ = octomap::point3d(feedback->pose.position.x, feedback->pose.position.y, feedback->pose.position.z);
      }
      if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_UP) {
          octomap::point3d start_pos;
          if(get_drone_position(start_pos)) {
               run_planning_algorithm(start_pos, goal_pos_);
          } else {
               RCLCPP_WARN(this->get_logger(), "Cannot find 'base_link' TF. Is Static TF running?");
          }
      }
  }

  void run_planning_algorithm(const octomap::point3d& start, const octomap::point3d& goal)
  {
      RCLCPP_INFO(this->get_logger(), "[Plan] A* Start: (%.1f, %.1f, %.1f) -> (%.1f, %.1f, %.1f)", 
          start.x(), start.y(), start.z(), goal.x(), goal.y(), goal.z());

      if (distmap_->getDistance(start) < safe_radius_ && distmap_->getDistance(start) >= 0) {
          RCLCPP_WARN(this->get_logger(), "Start point is inside obstacle! (Dist: %.2f)", distmap_->getDistance(start));
          return;
      }
      
      if (distmap_->getDistance(goal) < safe_radius_ && distmap_->getDistance(goal) >= 0) {
           RCLCPP_WARN(this->get_logger(), "Goal point is inside obstacle! (Dist: %.2f)", distmap_->getDistance(goal));
           return;
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

      int gx = (int)(goal.x()/resolution_);
      int gy = (int)(goal.y()/resolution_);
      int gz = (int)(goal.z()/resolution_);
      
      std::shared_ptr<AStarNode> final_node = nullptr;
      int iterations = 0;

      while(!open_list.empty()) {
          auto current = open_list.top(); open_list.pop();
          
          // [설정] 반복 횟수 제한 (넉넉하게 200만)
          if(++iterations > 2000000) { RCLCPP_WARN(this->get_logger(), "Timeout! Path blocked or too long."); return; }

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
                   int nx = current->x+dx, ny = current->y+dy, nz = current->z+dz;
                   octomap::point3d w_pos(nx*resolution_, ny*resolution_, nz*resolution_);

                   // ============================================
                   // [추가됨] 바닥 & 천장 뚫기 방지 코드
                   // ============================================
                   
                   // 1. 바닥 방지: 0.5m 미만은 절대 못 가게 막음 (지면 충돌 방지)
                   if (w_pos.z() < 0.5) continue; 

                   // 2. 천장 방지: 4.0m 초과는 절대 못 가게 막음 (선택 사항)
                   if (w_pos.z() > 4.0) continue; 
                   
                   // ============================================

                   // 맵 전체 범위 체크
                   if(w_pos.x()<x_min_ || w_pos.x()>x_max_ || w_pos.y()<y_min_ || w_pos.y()>y_max_ || w_pos.z()<z_min_ || w_pos.z()>z_max_) continue;
                   
                   // 장애물 거리 체크
                   float d = distmap_->getDistance(w_pos);
                   if(d < safe_radius_ && d >= 0) continue; 

                   auto neighbor = std::make_shared<AStarNode>();
                   neighbor->x = nx; neighbor->y = ny; neighbor->z = nz;
                   neighbor->parent = current;
                   neighbor->g_cost = current->g_cost + std::sqrt(dx*dx+dy*dy+dz*dz);
                   
                   // [수정됨] Heuristic 가중치 (* 2.0) 추가 -> 속도 향상
                   neighbor->h_cost = (w_pos - goal).norm() * 2.0;
                   
                   open_list.push(neighbor);
                }
             }
          }
      }

      if(final_node) publish_path(final_node);
      else RCLCPP_WARN(this->get_logger(), "Failed to find path.");
  }

  void publish_path(std::shared_ptr<AStarNode> node)
  {
      nav_msgs::msg::Path path_msg;
      path_msg.header.frame_id = "map";
      path_msg.header.stamp = this->now();

      std::vector<geometry_msgs::msg::PoseStamped> poses;
      while(node) {
          geometry_msgs::msg::PoseStamped p;
          p.header.frame_id = "map";
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
      RCLCPP_INFO(this->get_logger(), "Path Published! (%zu points)", poses.size());
  }

  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
      if (map_ready_) return; 
      RCLCPP_INFO(this->get_logger(), "[Plan] Processing Map...");
      
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
          RCLCPP_INFO(this->get_logger(), "[Plan] Map Ready! Waiting for Static TF and Marker Move...");
          
          // 기본 마커 생성 (높이를 1.0으로 조금 낮춰둠)
          create_marker(octomap::point3d(0, 0, 1.0));
      }
  }

  bool get_drone_position(octomap::point3d &pos)
  {
    try {
      geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
      pos.x() = t.transform.translation.x;
      pos.y() = t.transform.translation.y;
      pos.z() = t.transform.translation.z;
      return true;
    } catch (const tf2::TransformException & ex) {
      return false;
    }
  }

  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr map_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<interactive_markers::InteractiveMarkerServer> server_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlanner>());
  rclcpp::shutdown();
  return 0;
}