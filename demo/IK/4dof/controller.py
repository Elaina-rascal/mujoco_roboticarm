"""四自由度位置 + Pitch 逆解。"""

from pathlib import Path

import pinocchio as pin
import numpy as np


class Controller:
    def __init__(self, target_pitch=0.5):
        self.model = pin.buildModelFromMJCF(str(Path(__file__).with_name("arm.xml")))
        self.data = self.model.createData()
        if self.model.nq != 4 or self.model.nv != 4:
            raise ValueError("需要四个有界转动关节")
        if not self.model.existFrame("ee"):
            raise ValueError("模型中缺少末端 ee")
        self.ee_id = self.model.getFrameId("ee")
        self.target_pitch = target_pitch  # 目标 ZYX Pitch，rad。

    def compute(self, q, dq, ee_position, target_position, dt):
        """返回四个目标关节角，任务为世界系位置与 ZYX Pitch。"""
        error_threshold = 1e-3
        q_target = q.copy()
        for _ in range(100):
            pin.forwardKinematics(self.model, self.data, q_target)
            pin.updateFramePlacements(self.model, self.data)
            current_M = self.data.oMf[self.ee_id]
            R, p = current_M.rotation, current_M.translation
            pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
            yaw = np.arctan2(R[1, 0], R[0, 0])

            # 任务误差：[位置误差, Pitch 误差]。
            pitch_error = (self.target_pitch - pitch + np.pi) % (2 * np.pi) - np.pi
            error = np.r_[target_position - p, pitch_error]
            if np.linalg.norm(error) < error_threshold:
                break

            # Pinocchio WORLD 旋量顺序是 [v, w]。
            A = np.zeros((4, 6))
            A[:3, :3] = np.eye(3)
            A[:3, 3:] = -pin.skew(p)  # p_dot = v + w × p。
            A[3, 3:] = [-np.sin(yaw), np.cos(yaw), 0]  # 世界系角速度 -> Pitch 速度。
            J = pin.computeFrameJacobian(
                self.model, self.data, q_target, self.ee_id, pin.ReferenceFrame.WORLD
            )
            J_task = A @ J
            dq = np.linalg.pinv(J_task) @ error
            q_target += dq * 0.1
            q_target = np.clip(q_target, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        return q_target
