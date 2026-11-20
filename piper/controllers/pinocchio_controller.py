"""Piper FK / IK、重力补偿和 VMC 空间力到关节力矩映射。"""

from pathlib import Path

import numpy as np
import pinocchio as pin

from piper.models import ARM_MODEL
from piper.interface.base import vector

EE_TO_LEVEL = pin.rpy.rpyToMatrix(0.0, -np.pi / 2, 0.0) #type:ignore


class PinocchioControl:
    def __init__(self, damping=0.01, model_path=ARM_MODEL):
        self.model_path = Path(model_path).resolve()
        self.damping = damping
        self.min_singular_value = 1e-3
        self.max_task_error = 1e-3  # H 约束后的位姿误差，包含 m 和 rad。
        self.max_joint_change = np.deg2rad(50)
        self.ee_to_level = EE_TO_LEVEL.copy()
        self.H = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        self.task_dimension = 5
        full_model = pin.buildModelFromMJCF(str(self.model_path))
        fingers = [full_model.getJointId(name) for name in ("joint7", "joint8")]
        self.model = pin.buildReducedModel(full_model, fingers, pin.neutral(full_model)) #type:ignore
        self.data = self.model.createData()
        if not self.model.existFrame("ee"):
            raise ValueError("模型中缺少末端 ee")
        self.ee_id = self.model.getFrameId("ee")

    def fk(self, joint_positions):
        """六轴角度 rad → base 下的工具位姿 SE3，平移单位 m。

        原点沿用模型 ee；旋转为 R_base_ee @ ee_to_level。
        """
        q = np.asarray(joint_positions, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_id] #type:ignore
        return pin.SE3(pose.rotation @ self.ee_to_level, pose.translation.copy())

    def gravity(self, joint_positions):
        """返回六轴重力补偿力矩（N·m）；夹爪沿用六轴模型中的固定姿态。"""
        q = np.asarray(joint_positions, dtype=float)
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    def force_to_torque(self, joint_positions, force):
        """世界系末端 XYZ 力（N）→ 六轴 VMC 力矩（N·m），不含重力。"""
        q = vector(joint_positions, 6)
        force = vector(force, 3)
        jacobian = pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]
        return (jacobian.T @ force).copy()

    def ik(self, joint_positions, target_position):
        """求目标位置的六轴角度；工具保持水平，世界 yaw 自由，失败则返回初始 q。"""
        q_initial = np.asarray(joint_positions, dtype=float).copy()
        q_target = q_initial.copy()
        target = np.asarray(target_position, dtype=float)
        for iteration in range(101):
            current = self.fk(q_target)
            # 沿用仿真的 h=log3(R_tool) 约束，仅约束世界 x/y 分量。
            orientation_error = -pin.log3(current.rotation) #type:ignore
            error = self.H @ np.r_[target - current.translation, orientation_error]
            jac_lwa = pin.computeFrameJacobian(
                self.model, self.data, q_target, self.ee_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            error_rate_map = np.eye(6)
            error_rate_map[3:, 3:] = pin.Jlog3(current.rotation.T) #type:ignore
            jac = self.H @ error_rate_map @ jac_lwa
            u, singular_values, vt = np.linalg.svd(jac, full_matrices=False)

            if iteration == 100 or np.linalg.norm(error) <= self.max_task_error:
                if self.check(singular_values, error, q_initial, q_target):
                    return q_target
                return q_initial

            gains = singular_values / (singular_values**2 + self.damping**2)
            delta_q = vt.T @ (gains * (u.T @ error))
            q_target = pin.integrate(self.model, q_target, delta_q * 0.1)
            q_target = np.clip(
                q_target, self.model.lowerPositionLimit, self.model.upperPositionLimit
            )
    def check(self, singular_values, error, q_initial, q_final):
        """逐项检查最终位姿误差、有效奇异值和关节变化。"""
        error_norm = np.linalg.norm(self.H @ error)
        if not np.isfinite(error_norm):
            print("IK 拒绝：最终位姿误差包含非有限数值")
            return False
        if error_norm > self.max_task_error:
            print(f"IK 拒绝：最终位姿误差 {error_norm:.6f} > {self.max_task_error:.6f}")
            return False

        # H 不约束 yaw，会产生一个固定零奇异值。
        min_singular_value = singular_values[self.task_dimension - 1]
        if not np.all(np.isfinite(singular_values)):
            print("IK 拒绝：最终奇异值包含非有限数值")
            return False
        if min_singular_value < self.min_singular_value:
            print(f"IK 拒绝：最小有效奇异值 {min_singular_value:.6f} < {self.min_singular_value:.6f}")
            return False

        q_error = pin.difference(self.model, q_initial, q_final)
        if not np.all(np.isfinite(q_error)):
            print("IK 拒绝：关节变化包含非有限数值")
            return False
        joint_index = np.argmax(np.abs(q_error))
        max_change = abs(q_error[joint_index])
        if max_change > self.max_joint_change:
            print(
                f"IK 拒绝：joint{joint_index + 1} 变化 {max_change:.6f} rad"
                f" > {self.max_joint_change:.6f} rad"
            )
            return False
        return True
if __name__ == "__main__":
    HOLD_POSITIONS = np.array([0.114214, 0.586248, -0.464362, 0.019862, 0.192911, -0.019862])
    control = PinocchioControl()
    pose = control.fk(HOLD_POSITIONS)
    print("末端位姿：\n", pose)