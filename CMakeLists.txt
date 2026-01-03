cmake_minimum_required(VERSION 3.8)
project(my_3d_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 패키지 찾기 ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED) # [추가] TF 변환용
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# [추가] Interactive Marker 및 시각화 메시지 필수
find_package(interactive_markers REQUIRED)
find_package(visualization_msgs REQUIRED)

# --- [유지] dynamicEDT3D 라이브러리 직접 찾기 ---
find_library(DYNAMICEDT3D_LIB 
  NAMES dynamicedt3d
  HINTS /opt/ros/humble/lib /usr/lib /usr/local/lib
)

if(NOT DYNAMICEDT3D_LIB)
  message(FATAL_ERROR "dynamicEDT3D library not found! Please check installation.")
else()
  message(STATUS "Found dynamicEDT3D: ${DYNAMICEDT3D_LIB}")
endif()

include_directories(
  include
  ${octomap_INCLUDE_DIRS}
  /opt/ros/humble/include 
)

# ============================================
# 1. Node A: ESDF Visualizer (시각화 전용)
# ============================================
add_executable(esdf_visualizer src/esdf_visualizer.cpp)
target_include_directories(esdf_visualizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(esdf_visualizer 
  rclcpp 
  octomap_msgs
  tf2                
  tf2_ros            
  geometry_msgs    
  sensor_msgs
)

# 라이브러리 수동 링크
target_link_libraries(esdf_visualizer 
  ${octomap_LIBRARIES} 
  ${DYNAMICEDT3D_LIB} 
)


# ============================================
# 2. Node B: Global Planner A* (경로 계산 전용)
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
  tf2_geometry_msgs # [추가]          
  geometry_msgs    
  sensor_msgs
  nav_msgs
  interactive_markers # [추가] 마커 서버용
  visualization_msgs  # [추가] 마커 메시지용
)

# 라이브러리 수동 링크
target_link_libraries(global_planner_astar 
  ${octomap_LIBRARIES} 
  ${DYNAMICEDT3D_LIB} 
)

# ============================================
# 설치 설정
# ============================================
install(TARGETS
  esdf_visualizer
  global_planner_astar
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()