# 原生 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(ros2_plain_cmake)

# 设置 C++ 标准（ROS2 必须 C++17）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
# 直接查找 ROS2 核心依赖（纯 CMake 方式）
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED) # 新增消息依赖
# 编译可执行文件
add_executable(minimal_node src/minimal_node.cpp)

# 链接 ROS2 库（纯 CMake 链接方式）
target_link_libraries(minimal_node
  rclcpp::rclcpp
)

add_executable(minimal_publisher src/minimal_publisher.cpp)
target_link_libraries(minimal_publisher 
  rclcpp::rclcpp
  std_msgs::std_msgs__rosidl_typesupport_cpp # 链接消息的 C++ 类型支持
)