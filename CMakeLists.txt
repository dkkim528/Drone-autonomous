cmake_minimum_required(VERSION 3.8)
project(my_3d_planner) # 주의: package.xml의 <name> 태그와 똑같아야 합니다!

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================
# 1. 패키지 찾기 (Dependencies)
# ============================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)  # [필수] Float32MultiArray용
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(interactive_markers REQUIRED)

# TF2 관련
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED) 
find_package(tf2_sensor_msgs REQUIRED)

# Octomap 관련
find_package(octomap_msgs REQUIRED)
find_package(octomap REQUIRED)

# PCL 관련
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED) # ROS2 Humble에서는 보통 pcl_conversions만으로 충분함

# ============================================
# 2. DynamicEDT3D 라이브러리 설정
# ============================================
find_library(DYNAMICEDT3D_LIB 
  NAMES dynamicedt3d
  HINTS /opt/ros/humble/lib /usr/lib /usr/local/lib
)

if(NOT DYNAMICEDT3D_LIB)
  message(FATAL_ERROR "dynamicEDT3D library not found! Please check installation.")
else()
  message(STATUS "Found dynamicEDT3D: ${DYNAMICEDT3D_LIB}")
endif()

# ============================================
# 3. Include 디렉토리 설정
# ============================================
include_directories(
  include
  ${octomap_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# ============================================
# 4. Node A: Global Costmap Visualizer
# ============================================
add_executable(global_costmap_visualizer src/global_costmap_visualizer.cpp)
target_include_directories(global_costmap_visualizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(global_costmap_visualizer 
  rclcpp 
  octomap_msgs
  sensor_msgs
  geometry_msgs
)

target_link_libraries(global_costmap_visualizer 
  ${octomap_LIBRARIES} 
  ${DYNAMICEDT3D_LIB} 
)


# ============================================
# 5. Node B: Global Planner A*
# ============================================
add_executable(global_planner_astar src/global_planner_astar.cpp)
target_include_directories(global_planner_astar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(global_planner_astar 
  rclcpp 
  std_srvs
  nav_msgs
  geometry_msgs 
  sensor_msgs
  octomap_msgs
  visualization_msgs  
  interactive_markers 
  tf2 
  tf2_ros  
  tf2_geometry_msgs        
)

target_link_libraries(global_planner_astar 
  ${octomap_LIBRARIES} 
  ${DYNAMICEDT3D_LIB} 
)


# ============================================
# 6. Node C: Local Planner A*
# ============================================
add_executable(local_planner_astar src/local_planner_astar.cpp)
target_include_directories(local_planner_astar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(local_planner_astar 
  rclcpp 
  std_msgs       # [추가됨] Float32MultiArray 발행용
  std_srvs
  geometry_msgs 
  nav_msgs 
  sensor_msgs
  tf2 
  tf2_ros 
  tf2_geometry_msgs 
  tf2_sensor_msgs
  pcl_conversions
  pcl_ros       
)

target_link_libraries(local_planner_astar 
  ${PCL_LIBRARIES} 
)

# ============================================
# 7. Node D: Local Costmap Visualizer
# ============================================
add_executable(local_costmap_visualizer src/local_costmap_visualizer.cpp)
target_include_directories(local_costmap_visualizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(local_costmap_visualizer 
  rclcpp 
  std_msgs            # [추가됨] Float32MultiArray 구독용
  geometry_msgs
  visualization_msgs
)
# PCL 링크는 불필요하여 제거함 (Visualizer는 PCL 안씀)


# ============================================
# 설치 설정 (Install)
# ============================================
install(TARGETS
  global_costmap_visualizer
  global_planner_astar
  local_planner_astar
  local_costmap_visualizer
  DESTINATION lib/${PROJECT_NAME}
)

# [중요] Launch 파일 설치 규칙 추가
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()