#pragma once

#include <Eigen/Core>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/fwd.hpp>

#include <string>

namespace pinwrapper {

void BuildModelFromMjcf(const std::string &model_path, pinocchio::Model &model);
Eigen::VectorXd Neutral(const pinocchio::Model &model);
void ForwardKinematics(const pinocchio::Model &model, pinocchio::Data &data,
                       const Eigen::VectorXd &q);
void UpdateFramePlacements(const pinocchio::Model &model, pinocchio::Data &data);
void ComputeJointJacobians(const pinocchio::Model &model, pinocchio::Data &data,
                           const Eigen::VectorXd &q);
void GetFrameJacobian(const pinocchio::Model &model, pinocchio::Data &data,
                      pinocchio::FrameIndex frame_id,
                      pinocchio::ReferenceFrame reference_frame,
                      Eigen::Matrix<double, 6, Eigen::Dynamic> &jacobian);
pinocchio::Motion Log6(const pinocchio::SE3 &pose_error);
Eigen::VectorXd Integrate(const pinocchio::Model &model, const Eigen::VectorXd &q,
                          const Eigen::VectorXd &v);

} // namespace pinwrapper
