#pragma once
#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

// --- 关键修改：必须包含具体的算法头文件，否则编译器不认 template 声明 ---
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
namespace pinocchio {

// 在头文件中声明 extern template，避免每个编译单元重复实例化这些模板类，
// 由 pinocchio_wrapper.cpp 统一完成显式实例化，减少编译时间和目标文件体积。
// extern template class ModelTpl<double>;
extern template class DataTpl<double>;

// 2. 算法函数声明
// 注意：ConfigVectorType 必须与你业务代码传入的 Eigen 类型严格匹配（通常是
// VectorXd）

// forwardKinematics
extern template void
forwardKinematics<double, 0, JointCollectionDefaultTpl, Eigen::VectorXd>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &);

// updateFramePlacements
extern template void
updateFramePlacements<double, 0, JointCollectionDefaultTpl>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &);

// computeJointJacobians
extern template const typename DataTpl<double, 0,JointCollectionDefaultTpl>::Matrix6x &
computeJointJacobians<double, 0, JointCollectionDefaultTpl, Eigen::VectorXd>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &);

// getFrameJacobian
// 注意：业务代码中如果传入的是子矩阵，通常需要实例化 Eigen::Ref 版本
extern template void getFrameJacobian<double, 0, JointCollectionDefaultTpl,
                                      Eigen::Ref<Eigen::Matrix<double, 6, -1>>>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    DataTpl<double, 0, JointCollectionDefaultTpl> &, const FrameIndex,
    const ReferenceFrame,
    const Eigen::MatrixBase<Eigen::Ref<Eigen::Matrix<double, 6, -1>>> &);

// log6
extern template MotionTpl<double, 0> log6<double, 0>(const SE3Tpl<double, 0> &);

// integrate
extern template Eigen::Matrix<double, -1, 1>
integrate<double, 0, JointCollectionDefaultTpl, Eigen::VectorXd,
          Eigen::VectorXd>(
    const ModelTpl<double, 0, JointCollectionDefaultTpl> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &,
    const Eigen::MatrixBase<Eigen::VectorXd> &);


} // namespace pinocchio
