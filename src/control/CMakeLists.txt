# 原生 CMake 最低版本
cmake_minimum_required(VERSION 3.10)
# 项目名称
project(ros2_plain_cmake)

# 设置 C++ 标准（ROS2 必须 C++17）
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
# 直接查找 ROS2 核心依赖（纯 CMake 方式）
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pinocchio REQUIRED) # 新增 Pinocchio 依赖
# 编译可执行文件
add_executable(invkinematic src/invkinematic.cpp)

target_link_libraries(invkinematic 
  rclcpp::rclcpp
  pinocchio::pinocchio # 链接 Pinocchio 库
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp # 链接 ROS2 消息库
)   