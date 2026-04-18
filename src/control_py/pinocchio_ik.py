from pathlib import Path
from typing import Dict, List

import numpy as np
import pinocchio as pin


class PinocchioIKSolver:
    """Pinocchio 数值逆解核心层。"""

    def __init__(self, model_path: Path) -> None:
        # 设定目标位姿 (SE3: 旋转矩阵, 平移向量)
        self.target_M = pin.SE3(np.eye(3), np.array([0.35, 0.15, 0.5]))
        self.ee_frame_name = "attachment_site"

        # 模型加载 (使用 MJCF 解析器)
        self.model = pin.buildModelFromMJCF(str(model_path))
        self.data = self.model.createData()
        #断言model和model.names,model.joints等属性存在，确保模型加载成功
        assert isinstance(self.model, pin.Model), "模型加载失败，类型不匹配"
        assert self.model.names is not None, "模型加载失败，缺少 names 属性"
        # 检查末端执行器 Frame 是否存在
        if not self.model.existFrame(self.ee_frame_name):
            raise ValueError(f"未找到 Frame: {self.ee_frame_name}")
        self.ee_id = self.model.getFrameId(self.ee_frame_name)

        # 建立关节名称到 ID 的映射
        self.joint_name_to_id: Dict[str, int] = {}
        for i, name in enumerate(self.model.names):
            # 跳过 universe (index 0)
            if i == 0:
                continue
            self.joint_name_to_id[name] = i

    def solve(self, joint_names: List[str], joint_positions: List[float]) -> np.ndarray:
        """根据输入 JointState 解 IK，返回解出的 q。"""
        if not joint_names or not joint_positions:
            return pin.neutral(self.model)

        # 核心：根据映射填充当前配置 q
        q = pin.neutral(self.model)

        # 将传入的 JointState 位置映射到 Pinocchio 的 q 向量中
        for i, name in enumerate(joint_names):
            if name in self.joint_name_to_id:
                joint_id = self.joint_name_to_id[name]
                idx_q = self.model.joints[joint_id].idx_q #type: ignore
                q[idx_q] = joint_positions[i]

        # --- 核心 IK 循环 (数值解法) ---
        damping = 1e-4
        max_iter = 15
        step_size = 0.5

        for _ in range(max_iter):
            # 更新运动学数据
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q)
            # 计算当前末端位姿
            current_M = self.data.oMf[self.ee_id]  #type: ignore

            # 计算误差运动 (SE3 空间中的 log 映射)
            err_motion = pin.log6(current_M.inverse() * self.target_M)
            err = err_motion.vector
            assert isinstance(err, np.ndarray), "误差向量类型不匹配"
            if np.linalg.norm(err) < 1e-4: 
                break

            # 计算雅可比矩阵 (Body Frame / LOCAL)
            J = pin.getFrameJacobian(
                self.model,
                self.data,
                self.ee_id,
                pin.ReferenceFrame.LOCAL,
            )

            # 阻尼最小二乘法解: dq = J^T * (J * J^T + λI)^-1 * err
            jj_t = J @ J.T
            jj_t += np.eye(6) * damping
            dq = J.T @ np.linalg.solve(jj_t, err)

            # 更新配置 q
            q = pin.integrate(self.model, q, dq * step_size)

        return q

    def q_to_joint_positions(
        self,
        q: np.ndarray,
        joint_names: List[str],
        fallback_positions: List[float],
    ) -> List[float]:
        """把 Pinocchio 的 q 按 JointState 名字顺序回写为位置数组。"""
        out = list(fallback_positions)
        if len(out) < len(joint_names):
            out.extend([0.0] * (len(joint_names) - len(out)))

        for i, name in enumerate(joint_names):
            if name in self.joint_name_to_id:
                joint_id = self.joint_name_to_id[name]
                out[i] = float(q[self.model.joints[joint_id].idx_q]) #type: ignore
            else:
                out[i] = fallback_positions[i]

        return out
