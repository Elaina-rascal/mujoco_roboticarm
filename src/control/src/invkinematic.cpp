#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <unordered_map>

// 包含必要的 Pinocchio 头文件
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/mjcf.hpp>

using namespace pinocchio;
namespace fs = std::filesystem;

class SimpleIK : public rclcpp::Node {
public:
  SimpleIK() : Node("simple_ik") {
    // 1. 初始化路径与目标位姿
    fs::path model_path =
        fs::path(__FILE__).parent_path().parent_path() / "model" / "ur5e.xml";

    // 设定目标位姿
    target_M_ =
        SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.35, 0.15, 0.5));
    ee_frame_name_ = "attachment_site";

    // 2. 模型加载
    model_ = std::make_shared<Model>();
    mjcf::buildModel(model_path.string(), *model_);
    data_ = std::make_unique<Data>(*model_);

    // --- 新增：建立关节名称到 ID 的映射 ---
    // i 从 1 开始，因为 0 通常是 universe
    for (JointIndex i = 1; i < (JointIndex)model_->njoints; ++i) {
      joint_name_to_id_[model_->names[i]] = i;
    }

    if (!model_->existFrame(ee_frame_name_)) {
      throw std::runtime_error("Frame not found: " + ee_frame_name_);
    }
    ee_id_ = model_->getFrameId(ee_frame_name_);

    // 3. 通信
    sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 1,
        std::bind(&SimpleIK::jointStateCallback, this, std::placeholders::_1));
    pub_ =
        create_publisher<sensor_msgs::msg::JointState>("/ik_joint_target", 10);

    RCLCPP_INFO(get_logger(), "IK Node initialized with Joint Mapping.");
  }

private:
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (msg->name.empty() || msg->position.empty())
      return;

    // --- 核心修改：根据映射填充 q ---
    // 使用 neutral 位置初始化，防止某些关节缺失
    Eigen::VectorXd q = neutral(*model_);

    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto it = joint_name_to_id_.find(msg->name[i]);
      if (it != joint_name_to_id_.end()) {
        JointIndex joint_id = it->second;
        // 获取该关节在 q 中的起始索引 (对于单自由度关节，nq 通常是 1)
        int idx = model_->joints[joint_id].idx_q();
        q[idx] = msg->position[i];
      }
    }

    // --- 核心 IK 循环 ---
    const double damping = 1e-4;
    const int max_iter = 15;
    const double step_size = 0.5;

    for (int i = 0; i < max_iter; i++) {
      forwardKinematics(*model_, *data_, q);
      updateFramePlacements(*model_, *data_);
      computeJointJacobians(*model_, *data_, q);

      const SE3 &current_M = data_->oMf[ee_id_];

      // 这里建议切换到 LOCAL_WORLD_ALIGNED 以获得更好的稳定性
      const Motion err_motion = log6(current_M.inverse() * target_M_);
      Eigen::Matrix<double, 6, 1> err = err_motion.toVector();

      if (err.norm() < 1e-4)
        break;

      Data::Matrix6x J(6, model_->nv);
      getFrameJacobian(*model_, *data_, ee_id_, LOCAL, J);

      Data::Matrix6x JJt = J * J.transpose();
      JJt.diagonal().array() += damping;
      Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(err);

      q = integrate(*model_, q, dq * step_size);
    }

    // --- 发布指令：写回对应位置 ---
    auto out_msg = *msg;
    for (size_t i = 0; i < out_msg.name.size(); ++i) {
      auto it = joint_name_to_id_.find(out_msg.name[i]);
      if (it != joint_name_to_id_.end()) {
        out_msg.position[i] = q[model_->joints[it->second].idx_q()];
      }
    }
    pub_->publish(out_msg);
  }

  std::shared_ptr<Model> model_;
  std::unique_ptr<Data> data_;
  std::unordered_map<std::string, JointIndex> joint_name_to_id_; // 映射表
  FrameIndex ee_id_;
  SE3 target_M_;
  std::string ee_frame_name_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleIK>());
  rclcpp::shutdown();
  return 0;
}