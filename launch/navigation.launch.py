import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'my_3d_planner'

    # 파라미터 정의
    enable_vis_arg = DeclareLaunchArgument(
        'enable_global_vis',
        default_value='false',
        description='Enable heavy global costmap visualization (True/False)'
    )

    safety_dist_arg = DeclareLaunchArgument(
        'safety_distance',
        default_value='0.5',
        description='Safety margin for local obstacle avoidance'
    )

    # 1. Global Planner
    global_planner = Node(
        package=package_name,
        executable='global_planner_astar',
        name='global_planner',
        output='screen',
        parameters=[{'resolution': 0.5, 'safe_radius': 0.6}]
    )

    # 2. Global Costmap Visualizer (조건부 실행)
    global_visualizer = Node(
        package=package_name,
        executable='global_costmap_visualizer',
        name='global_costmap_visualizer',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_global_vis'))
    )

    # 3. Local Planner
    local_planner = Node(
        package=package_name,
        executable='local_planner_astar',
        name='local_planner',
        output='screen',
        parameters=[{'safety_distance': LaunchConfiguration('safety_distance')}]
    )

    # 4. Local Costmap Visualizer
    local_visualizer = Node(
        package=package_name,
        executable='local_costmap_visualizer',
        name='local_costmap_visualizer',
        output='screen'
    )

    return LaunchDescription([
        enable_vis_arg,
        safety_dist_arg,
        global_planner,
        global_visualizer,
        local_planner,
        local_visualizer
    ])