cmake_minimum_required(VERSION 3.8)
project(my_3d_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================
# 1. 패키지 찾기 (Dependencies)
# ============================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED) 
find_package(tf2_sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# [PCL 관련]
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

# [Visualizer 관련]
find_package(interactive_markers REQUIRED)
find_package(visualization_msgs REQUIRED)

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
  /opt/ros/humble/include 
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# ============================================
# 4. Node A: Global Costmap Visualizer (이름 변경됨)
# ============================================
# [변경] 실행파일 이름과 소스파일 이름 모두 변경
add_executable(global_costmap_visualizer src/global_costmap_visualizer.cpp)
target_include_directories(global_costmap_visualizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# [변경] 타겟 이름 변경 (esdf_visualizer -> global_costmap_visualizer)
ament_target_dependencies(global_costmap_visualizer 
  rclcpp 
  octomap_msgs
  tf2                  
  tf2_ros              
  geometry_msgs    
  sensor_msgs
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
  octomap_msgs
  tf2                  
  tf2_ros  
  tf2_geometry_msgs       
  geometry_msgs    
  sensor_msgs
  nav_msgs
  std_srvs
  interactive_markers 
  visualization_msgs  
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
  geometry_msgs 
  nav_msgs 
  sensor_msgs
  std_srvs
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
# 7. Node D: Local Costmap Visualizer (이름 변경됨)
# ============================================
# [변경] voxel_visualizer -> local_costmap_visualizer
add_executable(local_costmap_visualizer src/local_costmap_visualizer.cpp)
target_include_directories(local_costmap_visualizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(local_costmap_visualizer 
  rclcpp 
  sensor_msgs
  visualization_msgs
  pcl_conversions      
  pcl_ros              
)

target_link_libraries(local_costmap_visualizer 
  ${PCL_LIBRARIES} 
)


# ============================================
# 설치 설정 (Install)
# ============================================
install(TARGETS
  global_costmap_visualizer   # [변경]
  global_planner_astar
  local_planner_astar
  local_costmap_visualizer    # [변경]
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()