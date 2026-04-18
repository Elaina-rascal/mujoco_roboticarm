#include "control/pinocchio_wrapper.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Dense>

// template class pinocchio::ModelTpl<double>;
template class pinocchio::DataTpl<double>;

namespace pinocchio {

// 2. 算法函数声明
// 注意：ConfigVectorType 必须与你业务代码传入的 Eigen 类型严格匹配（通常是
// VectorXd）

// forwardKinematics
template void
forwardKinematics<double, 0, JointCollectionDefaultTpl, Eigen::VectorXd>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &);

// updateFramePlacements
template void
updateFramePlacements<double, 0, JointCollectionDefaultTpl>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &);

// computeJointJacobians
template const typename DataTpl<double, 0,JointCollectionDefaultTpl>::Matrix6x &
computeJointJacobians<double, 0, JointCollectionDefaultTpl, Eigen::VectorXd>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &);

// getFrameJacobian
// 注意：业务代码中如果传入的是子矩阵，通常需要实例化 Eigen::Ref 版本
template void getFrameJacobian<double, 0, JointCollectionDefaultTpl,
                                      Eigen::Ref<Eigen::Matrix<double, 6, -1>>>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &, const FrameIndex,
    const ReferenceFrame,
    const Eigen::MatrixBase<Eigen::Ref<Eigen::Matrix<double, 6, -1>>> &);

// log6
template MotionTpl<double, 0> log6<double, 0>(const SE3Tpl<double, 0> &);

// integrate
template Eigen::Matrix<double, -1, 1>
integrate<double, 0, JointCollectionDefaultTpl, Eigen::VectorXd,
          Eigen::VectorXd>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &);

} // namespace pinocchio
