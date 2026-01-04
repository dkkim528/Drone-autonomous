#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp> // [중요] Planner와 맞춤
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

using std::placeholders::_1;

class LocalCostmapVisualizer : public rclcpp::Node
{
public:
    LocalCostmapVisualizer() : Node("local_costmap_visualizer")
    {
        // 1. Planner가 보내는 Float32MultiArray 구독
        sub_grid_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/local_grid_debug", 10, std::bind(&LocalCostmapVisualizer::grid_callback, this, _1));

        // 2. RViz에 그릴 마커 발행
        pub_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/local_voxel_marker", 10);

        RCLCPP_INFO(this->get_logger(), "Visualizer Started! Listening for Float32Array...");
    }

private:
    void grid_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        if (msg->data.empty()) return;

        // 마커 기본 설정
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link"; // 드론 기준
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "local_grid";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE_LIST; // 큐브 리스트
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Voxel 크기 (Planner 설정: 0.2m)
        marker.scale.x = 0.2; 
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;

        // 색상 (빨간색, 반투명)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;

        // Planner의 Grid 설정과 맞춰야 역계산 가능
        // (Planner 코드에 있는 설정값과 동일해야 함)
        const int dim_x = 20; 
        const int dim_y = 20; 
        const int dim_z = 10;
        const double grid_res = 0.2;

        // 1차원 배열(Index) -> 3차원 좌표(x,y,z) 변환
        for (size_t i = 0; i < msg->data.size(); ++i) {
            // 값이 0.5보다 크면 장애물(1.0)이라고 판단
            if (msg->data[i] > 0.5) {
                // 인덱스 역계산
                int z_idx = i / (dim_x * dim_y);
                int rem = i % (dim_x * dim_y);
                int y_idx = rem / dim_x;
                int x_idx = rem % dim_x;

                // 인덱스 -> 실제 미터 단위 좌표 변환
                // (Center Offset 보정: 인덱스 10이 0.0m가 되도록)
                double px = (x_idx - (dim_x / 2)) * grid_res;
                double py = (y_idx - (dim_y / 2)) * grid_res;
                double pz = (z_idx - (dim_z / 2)) * grid_res;

                geometry_msgs::msg::Point p;
                p.x = px;
                p.y = py;
                p.z = pz;
                marker.points.push_back(p);
            }
        }

        // 포인트가 하나도 없으면 삭제 명령
        if (marker.points.empty()) {
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
        }

        pub_marker_->publish(marker);
    }

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_grid_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalCostmapVisualizer>());
    rclcpp::shutdown();
    return 0;
}