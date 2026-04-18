#include "control/pinocchio_wrapper.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Dense>

template class pinocchio::ModelTpl<double>;
template class pinocchio::DataTpl<double>;

namespace pinocchio {

void instantiate_used_algorithms_for_control() {
	// 该函数故意不在业务路径中调用。
	// 只要本编译单元参与编译，下面这些调用就会把 IKController 用到的
	// Pinocchio 算法模板集中实例化到这里，避免散落在多个 .cpp 中重复生成。
	Model model;
	Data data(model);

	const Eigen::VectorXd q = neutral(model);
	forwardKinematics(model, data, q);
	updateFramePlacements(model, data);
	computeJointJacobians(model, data, q);

	const SE3 identity_pose = SE3::Identity();
	const Motion motion_err = log6(identity_pose);

	Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian(6, model.nv);
	getFrameJacobian(model, data, FrameIndex(0), LOCAL, jacobian);

	const Eigen::VectorXd q_next = integrate(model, q, q * 0.0);

	(void)motion_err;
	(void)q_next;
}

} // namespace pinocchio
