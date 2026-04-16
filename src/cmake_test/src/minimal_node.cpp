#include "rclcpp/rclcpp.hpp"
#include <rclcpp/utilities.hpp>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("plain_cmake_node");
  RCLCPP_INFO(node->get_logger(), "纯 CMake ROS2 节点运行成功！");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}