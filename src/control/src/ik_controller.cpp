#include "control/ik_controller.hpp"
#include "control/pinocchio_wrapper.hpp"

#include <algorithm>
#include <stdexcept>

namespace control {

IKController::IKController(const std::string &model_path,
                           const std::string &ee_frame_name,
                           const pinocchio::SE3 &target_pose)
    : model_(), data_(model_), ee_id_(0), joint_name_to_id_(), target_pose_(target_pose) {
  // 从 MJCF 文件加载机器人模型，并初始化与模型绑定的运行时缓存 data_。
  // data_ 中会保存正运动学结果、雅可比等中间量，后续每次迭代都复用它。
  pinwrapper::BuildModelFromMjcf(model_path, model_);
  data_ = pinocchio::Data(model_);

  // 建立 “关节名 -> JointIndex” 的查找表，避免每次回调都线性搜索。
  // 从 1 开始是因为 0 通常是 universe 虚拟关节，不对应真实驱动关节。
  for (pinocchio::JointIndex i = 1; i < static_cast<pinocchio::JointIndex>(model_.njoints);
       ++i) {
    joint_name_to_id_[model_.names[i]] = i;
  }

  // 末端 frame 必须存在，否则 IK 目标无从定义，直接抛异常。
  if (!model_.existFrame(ee_frame_name)) {
    throw std::runtime_error("Frame not found: " + ee_frame_name);
  }
  ee_id_ = model_.getFrameId(ee_frame_name);
}

Eigen::VectorXd IKController::buildConfiguration(
    const std::vector<std::string> &joint_names,
    const std::vector<double> &joint_positions) const {
  // 先用 neutral 配置兜底，避免 JointState 缺少部分关节时出现未初始化数据。
  Eigen::VectorXd q = pinwrapper::Neutral(model_);
  const size_t count = std::min(joint_names.size(), joint_positions.size());

  // 按名字把 ROS JointState 的位置写入 Pinocchio 配置向量 q。
  // 若某个名字在模型中不存在，跳过即可，不影响其他关节。
  for (size_t i = 0; i < count; ++i) {
    const auto it = joint_name_to_id_.find(joint_names[i]);
    if (it == joint_name_to_id_.end()) {
      continue;
    }
    const int idx_q = model_.joints[it->second].idx_q();
    q[idx_q] = joint_positions[i];
  }

  return q;
}

std::vector<double> IKController::extractJointPositions(
    const Eigen::VectorXd &q,
    const std::vector<std::string> &joint_names,
    const std::vector<double> &fallback_positions) const {
  // 先复制输入位置作为回退值，保证未知关节或缺失关节保持原值。
  std::vector<double> out = fallback_positions;
  if (out.size() < joint_names.size()) {
    // 理论上 name/position 应该等长，这里做防御性扩容。
    out.resize(joint_names.size(), 0.0);
  }

  // 将 Pinocchio 解出的 q 按 JointState 的名字顺序回写到输出数组。
  for (size_t i = 0; i < joint_names.size(); ++i) {
    const auto it = joint_name_to_id_.find(joint_names[i]);
    if (it == joint_name_to_id_.end()) {
      continue;
    }
    const int idx_q = model_.joints[it->second].idx_q();
    out[i] = q[idx_q];
  }
  return out;
}

std::vector<double> IKController::solve(const std::vector<std::string> &joint_names,
                                        const std::vector<double> &joint_positions) {
  // 阻尼最小二乘（DLS）IK 参数：
  // kDamping: 抑制奇异位形附近的数值发散
  // kMaxIterations: 单次求解最大迭代轮数
  // kStepSize: 每次更新步长
  // kTolerance: 任务空间误差范数收敛阈值
  constexpr double kDamping = 1e-4;
  constexpr int kMaxIterations = 15;
  constexpr double kStepSize = 0.5;
  constexpr double kTolerance = 1e-4;

  // 将 ROS 输入转换为 Pinocchio 配置向量。
  auto q = buildConfiguration(joint_names, joint_positions);

  for (int i = 0; i < kMaxIterations; ++i) {
    // 1) 在当前 q 下更新运动学、frame 位姿和关节雅可比缓存。
    pinwrapper::ForwardKinematics(model_, data_, q);
    pinwrapper::UpdateFramePlacements(model_, data_);
    pinwrapper::ComputeJointJacobians(model_, data_, q);

    // 2) 计算末端位姿误差（当前位姿 -> 目标位姿），并映射到 se(3) 6 维向量。
    //    error 前 3 维通常对应角速度误差，后 3 维对应线速度误差。
    const pinocchio::SE3 current_pose = data_.oMf[ee_id_];
    const pinocchio::Motion error_motion = pinwrapper::Log6(
      current_pose.inverse() * target_pose_);
    const Eigen::Matrix<double, 6, 1> error = error_motion.toVector();

    // 3) 达到阈值则提前收敛。
    if (error.norm() < kTolerance) {
      break;
    }

    // 4) 取末端 frame 在 LOCAL 坐标系下的 6 x nv 雅可比矩阵。
    Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian(6, model_.nv);
    pinwrapper::GetFrameJacobian(model_, data_, ee_id_, pinocchio::LOCAL,
                   jacobian);

    // 5) 构造 DLS 正规方程的稳定项 (J J^T + λI)。
    Eigen::Matrix<double, 6, 6> jj_t = jacobian * jacobian.transpose();
    jj_t.diagonal().array() += kDamping;

    // 6) DLS 更新：dq = J^T (J J^T + λI)^-1 e。
    //    最后通过 integrate 在流形上更新 q，避免直接相加带来的表示错误。
    const Eigen::VectorXd dq = jacobian.transpose() * jj_t.ldlt().solve(error);
    q = pinwrapper::Integrate(model_, q, dq * kStepSize);
  }

  // 将最终 q 按输入关节顺序导出，作为 ROS 输出消息位置数组。
  return extractJointPositions(q, joint_names, joint_positions);
}

} // namespace control
