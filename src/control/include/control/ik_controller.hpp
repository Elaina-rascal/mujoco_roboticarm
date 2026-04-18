#pragma once

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace control {

class IKController {
public:
  IKController(const std::string &model_path,
               const std::string &ee_frame_name,
               const pinocchio::SE3 &target_pose);

  std::vector<double> solve(const std::vector<std::string> &joint_names,
                            const std::vector<double> &joint_positions);

private:
  Eigen::VectorXd buildConfiguration(const std::vector<std::string> &joint_names,
                                     const std::vector<double> &joint_positions) const;
  std::vector<double> extractJointPositions(const Eigen::VectorXd &q,
                                            const std::vector<std::string> &joint_names,
                                            const std::vector<double> &fallback_positions) const;

  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex ee_id_;
  std::unordered_map<std::string, pinocchio::JointIndex> joint_name_to_id_;
  pinocchio::SE3 target_pose_;
};

} // namespace control
