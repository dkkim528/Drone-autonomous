import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'my_3d_planner'
    
    # YAML 파일 경로 가져오기
    config_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'planner_config.yaml'
    )

    return LaunchDescription([
        # Global Planner
        Node(
            package=pkg_name,
            executable='global_planner_astar',
            name='global_planner_astar',
            output='screen',
            parameters=[config_file] # 파라미터 로드
        ),
        # Local Planner
        Node(
            package=pkg_name,
            executable='local_planner_aster',
            name='local_planner_aster',
            output='screen',
            parameters=[config_file] # 파라미터 로드
        )
    ])