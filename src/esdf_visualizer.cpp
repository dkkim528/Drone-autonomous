// ==========================================
// Node A: ESDF Visualizer (시각화 전용)
// ==========================================
#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <dynamicEDT3D/dynamicEDTOctomap.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

using std::placeholders::_1;

class ESDFVisualizer : public rclcpp::Node
{
public:
  ESDFVisualizer() : Node("esdf_visualizer_node")
  {
    // QoS 설정 (Map 데이터는 중요하므로 Reliable)
    rclcpp::QoS qos(1);
    qos.transient_local();
    qos.reliable();

    // 맵 구독
    sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_binary", qos, std::bind(&ESDFVisualizer::octomap_callback, this, _1));

    // 시각화 데이터 발행 (RViz용)
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("esdf_vis", qos);

    RCLCPP_INFO(this->get_logger(), "=== Node A: ESDF Visualizer Started ===");
  }

private:
  bool map_processed_ = false; // 한 번만 실행하기 위한 플래그

  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    if (map_processed_) return; // 이미 그렸으면 패스

    RCLCPP_INFO(this->get_logger(), "[Vis] Map Received. Generating PointCloud...");

    octomap::AbstractOcTree* tree = octomap_msgs::binaryMsgToMap(*msg);
    if (!tree) return;

    octomap::OcTree* octree = dynamic_cast<octomap::OcTree*>(tree);
    if (octree) {
      double x_min, y_min, z_min, x_max, y_max, z_max;
      octree->getMetricMin(x_min, y_min, z_min);
      octree->getMetricMax(x_max, y_max, z_max);

      float max_dist = 2.0; // 시각화할 최대 거리

      // ESDF 계산 (시각화를 위해 일회성 생성)
      DynamicEDTOctomap distmap(max_dist, octree, 
          octomap::point3d(x_min, y_min, z_min), 
          octomap::point3d(x_max, y_max, z_max), 
          false);
      distmap.update();

      // 시각화 함수 호출
      visualize_esdf(&distmap, x_min, x_max, y_min, y_max, z_min, z_max, max_dist);
      
      map_processed_ = true;
      RCLCPP_INFO(this->get_logger(), "[Vis] Visualization Published!");
    }
    delete tree; // 시각화용은 멤버변수로 저장할 필요 없이 쓰고 바로 해제
  }

  void visualize_esdf(DynamicEDTOctomap* distmap, 
                      double x_min, double x_max, 
                      double y_min, double y_max, 
                      double z_min, double z_max, 
                      float max_dist)
  {
    auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud_msg->header.frame_id = "map";
    cloud_msg->header.stamp = this->now();
    
    sensor_msgs::PointCloud2Modifier modifier(*cloud_msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    
    double res = 0.2; // 시각화 해상도
    long long est_count = (long long)((x_max-x_min)/res)*((y_max-y_min)/res)*((z_max-z_min)/res);
    modifier.resize(est_count);

    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");

    int count = 0;
    for (double x = x_min; x < x_max; x += res) {
      for (double y = y_min; y < y_max; y += res) {
        for (double z = z_min; z < z_max; z += res) {
             float dist = distmap->getDistance(octomap::point3d(x, y, z));
             if (dist < max_dist && dist >= 0) {
                 *iter_x = x; *iter_y = y; *iter_z = z;
                 
                 // 색상: 가까울수록 빨강, 멀수록 초록
                 uint8_t r, g, b;
                 if (dist <= 0) { r=50; g=50; b=50; }
                 else {
                     float ratio = dist / max_dist;
                     r = (1.0 - ratio) * 255; g = ratio * 255; b = 0;
                 }
                 *iter_r = r; *iter_g = g; *iter_b = b;

                 ++iter_x; ++iter_y; ++iter_z;
                 ++iter_r; ++iter_g; ++iter_b;
                 count++;
             }
        }
      }
    }
    modifier.resize(count);
    pub_->publish(*cloud_msg);
  }

  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ESDFVisualizer>());
  rclcpp::shutdown();
  return 0;
}