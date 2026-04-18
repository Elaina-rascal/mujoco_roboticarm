#include "control/ik_node.hpp"

#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Dense>

#include <filesystem>

namespace fs = std::filesystem;

namespace control {

IKNode::IKNode() : Node("simple_ik") {
  const fs::path model_path =
      fs::path(__FILE__).parent_path().parent_path() / "model" / "ur5e.xml";
  const std::string ee_frame_name = "attachment_site";
  const pinocchio::SE3 target_pose(
      Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.35, 0.15, 0.5));

    controller_ = std::make_unique<IKController>(
      model_path.string(), ee_frame_name, target_pose);

  sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 1,
      std::bind(&IKNode::jointStateCallback, this, std::placeholders::_1));
  pub_ = create_publisher<sensor_msgs::msg::JointState>("/ik_joint_target", 10);

  RCLCPP_INFO(get_logger(), "IK Node initialized with layered architecture.");
}

void IKNode::jointStateCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  if (!msg || msg->name.empty() || msg->position.empty()) {
    return;
  }

  if (msg->position.size() < msg->name.size()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "JointState position size is smaller than name size.");
    return;
  }

  auto out_msg = *msg;
  out_msg.position = controller_->solve(msg->name, msg->position);
  pub_->publish(out_msg);
}

} // namespace control
