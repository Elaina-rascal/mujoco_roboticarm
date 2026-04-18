#pragma once

#include "control/ik_controller.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <memory>

namespace control {

class IKNode : public rclcpp::Node {
public:
  IKNode();

private:
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

  std::unique_ptr<IKController> controller_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
};

} // namespace control
