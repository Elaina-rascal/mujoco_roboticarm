#include <Eigen/Dense>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/mjcf.hpp"

using joint_state_msg = sensor_msgs::msg::JointState;
namespace fs = std::filesystem;

class ReceiveNode : public rclcpp::Node {
public:
  ReceiveNode() : Node("receive_node") {
    declare_parameter<std::vector<double>>("target_position", {0.35, -0.15, 0.35});
    declare_parameter<std::string>("end_effector_frame", "attachment_site");
    declare_parameter<double>("damping", 1e-3);
    declare_parameter<double>("step_size", 0.5);
    declare_parameter<double>("tolerance", 1e-4);
    declare_parameter<int>("max_iterations", 20);
    declare_parameter<std::string>("joint_states_topic", "/joint_states");
    declare_parameter<std::string>("ik_command_topic", "/ik_joint_target");

    target_position_ = get_parameter("target_position").as_double_array();
    if (target_position_.size() != 3) {
      throw std::runtime_error("target_position must contain exactly 3 values");
    }

    end_effector_frame_ = get_parameter("end_effector_frame").as_string();
    damping_ = get_parameter("damping").as_double();
    step_size_ = get_parameter("step_size").as_double();
    tolerance_ = get_parameter("tolerance").as_double();
    max_iterations_ = get_parameter("max_iterations").as_int();
    joint_states_topic_ = get_parameter("joint_states_topic").as_string();
    ik_command_topic_ = get_parameter("ik_command_topic").as_string();

    const fs::path model_path = fs::path(__FILE__).parent_path().parent_path() / "model" / "ur5e.xml";
    RCLCPP_INFO(get_logger(), "Loading Pinocchio model from: %s", model_path.c_str());

    pinocchio::Model model;
    pinocchio::mjcf::buildModelFromXML(model_path.string(), model);
    model_ = std::make_shared<pinocchio::Model>(std::move(model));
    data_ = std::make_unique<pinocchio::Data>(*model_);

    if (!model_->existFrame(end_effector_frame_)) {
      throw std::runtime_error("end_effector_frame not found in Pinocchio model: " + end_effector_frame_);
    }
    end_effector_frame_id_ = model_->getFrameId(end_effector_frame_);

    for (pinocchio::JointIndex joint_id = 1; joint_id < static_cast<pinocchio::JointIndex>(model_->njoints); ++joint_id) {
      const std::string &joint_name = model_->names[joint_id];
      if (!joint_name.empty()) {
        joint_name_to_id_.emplace(joint_name, joint_id);
      }
    }

    joint_sub_ = create_subscription<joint_state_msg>(
      joint_states_topic_, rclcpp::SensorDataQoS(),
      [this](const joint_state_msg::SharedPtr msg) { inverseKinematics(msg); });

    ik_pub_ = create_publisher<joint_state_msg>(ik_command_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "IK target initialized. frame=%s target=[%.4f, %.4f, %.4f] damping=%.3e step=%.3f tol=%.3e max_iter=%d",
      end_effector_frame_.c_str(), target_position_[0], target_position_[1], target_position_[2],
      damping_, step_size_, tolerance_, max_iterations_);
  }

private:
  Eigen::VectorXd jointStateToConfiguration(const joint_state_msg &msg) const {
    Eigen::VectorXd q = pinocchio::neutral(*model_);

    for (std::size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
      const auto it = joint_name_to_id_.find(msg.name[i]);
      if (it == joint_name_to_id_.end()) {
        continue;
      }

      const pinocchio::JointIndex joint_id = it->second;
      const pinocchio::Index q_index = model_->joints[joint_id].idx_q();
      q[q_index] = msg.position[i];
    }

    return q;
  }

  Eigen::VectorXd solveInverseKinematics(const Eigen::VectorXd &q_seed) {
    Eigen::VectorXd q = q_seed;

    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
      pinocchio::forwardKinematics(*model_, *data_, q);
      pinocchio::updateFramePlacements(*model_, *data_);

      const Eigen::Vector3d current_position = data_->oMf[end_effector_frame_id_].translation();
      const Eigen::Vector3d target(target_position_[0], target_position_[1], target_position_[2]);
      const Eigen::Vector3d error = target - current_position;

      if (error.norm() < tolerance_) {
        break;
      }

      pinocchio::computeJointJacobians(*model_, *data_, q);
      Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, model_->nv);
      pinocchio::getFrameJacobian(*model_, *data_, end_effector_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED, jacobian);

      const Eigen::MatrixXd position_jacobian = jacobian.topRows(3);
      const Eigen::Matrix3d system_matrix = position_jacobian * position_jacobian.transpose() +
                                            (damping_ * damping_) * Eigen::Matrix3d::Identity();
      const Eigen::Vector3d correction = step_size_ * system_matrix.ldlt().solve(error);
      const Eigen::VectorXd dq = position_jacobian.transpose() * correction;

      q = pinocchio::integrate(*model_, q, dq);
    }

    return q;
  }

  void publishJointTarget(const joint_state_msg &reference_msg, const Eigen::VectorXd &q_target) {
    joint_state_msg command_msg;
    command_msg.header.stamp = now();
    command_msg.name = reference_msg.name;
    command_msg.position.resize(reference_msg.name.size());
    command_msg.velocity.clear();
    command_msg.effort.clear();

    for (std::size_t i = 0; i < reference_msg.name.size(); ++i) {
      const auto it = joint_name_to_id_.find(reference_msg.name[i]);
      if (it == joint_name_to_id_.end()) {
        command_msg.position[i] = reference_msg.position[i];
        continue;
      }

      const pinocchio::JointIndex joint_id = it->second;
      const pinocchio::Index q_index = model_->joints[joint_id].idx_q();
      command_msg.position[i] = q_target[q_index];
    }

    ik_pub_->publish(command_msg);
  }

  void inverseKinematics(const joint_state_msg::SharedPtr msg) {
    if (msg->name.empty() || msg->position.empty()) {
      return;
    }

    const Eigen::VectorXd q_seed = jointStateToConfiguration(*msg);
    const Eigen::VectorXd q_target = solveInverseKinematics(q_seed);
    publishJointTarget(*msg, q_target);

    pinocchio::forwardKinematics(*model_, *data_, q_target);
    pinocchio::updateFramePlacements(*model_, *data_);
    const Eigen::Vector3d actual = data_->oMf[end_effector_frame_id_].translation();
    const Eigen::Vector3d target(target_position_[0], target_position_[1], target_position_[2]);
    const double error_norm = (target - actual).norm();

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "IK solved. target=[%.4f, %.4f, %.4f] actual=[%.4f, %.4f, %.4f] error=%.6f",
      target[0], target[1], target[2], actual[0], actual[1], actual[2], error_norm);
  }

  std::shared_ptr<pinocchio::Model> model_;
  std::unique_ptr<pinocchio::Data> data_;
  std::unordered_map<std::string, pinocchio::JointIndex> joint_name_to_id_;
  rclcpp::Subscription<joint_state_msg>::SharedPtr joint_sub_;
  rclcpp::Publisher<joint_state_msg>::SharedPtr ik_pub_;

  std::vector<double> target_position_;
  std::string end_effector_frame_;
  std::string joint_states_topic_;
  std::string ik_command_topic_;
  pinocchio::FrameIndex end_effector_frame_id_{};
  double damping_{1e-3};
  double step_size_{0.5};
  double tolerance_{1e-4};
  int max_iterations_{20};
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ReceiveNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}