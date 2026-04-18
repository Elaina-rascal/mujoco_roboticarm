#include "control/pinocchio_wrapper.hpp"

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/se3.hpp>

namespace pinwrapper {

void BuildModelFromMjcf(const std::string &model_path, pinocchio::Model &model) {
    pinocchio::mjcf::buildModel(model_path, model);
}

Eigen::VectorXd Neutral(const pinocchio::Model &model) {
    return pinocchio::neutral(model);
}

void ForwardKinematics(const pinocchio::Model &model, pinocchio::Data &data,
                                             const Eigen::VectorXd &q) {
    pinocchio::forwardKinematics(model, data, q);
}

void UpdateFramePlacements(const pinocchio::Model &model, pinocchio::Data &data) {
    pinocchio::updateFramePlacements(model, data);
}

void ComputeJointJacobians(const pinocchio::Model &model, pinocchio::Data &data,
                                                     const Eigen::VectorXd &q) {
    pinocchio::computeJointJacobians(model, data, q);
}

void GetFrameJacobian(const pinocchio::Model &model, pinocchio::Data &data,
                                            pinocchio::FrameIndex frame_id,
                                            pinocchio::ReferenceFrame reference_frame,
                                            Eigen::Matrix<double, 6, Eigen::Dynamic> &jacobian) {
    pinocchio::getFrameJacobian(model, data, frame_id, reference_frame, jacobian);
}

pinocchio::Motion Log6(const pinocchio::SE3 &pose_error) {
    return pinocchio::log6(pose_error);
}

Eigen::VectorXd Integrate(const pinocchio::Model &model, const Eigen::VectorXd &q,
                                                    const Eigen::VectorXd &v) {
    return pinocchio::integrate(model, q, v);
}

} // namespace pinwrapper
