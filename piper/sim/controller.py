"""xyz IK：可选欧拉角 pitch 约束或 SO(3) h 约束。"""

import numpy as np
import pinocchio as pin
if __package__:
    from .simulation import MODEL_PATH, vector
else:
    from simulation import MODEL_PATH, vector

# 在这里切换模式，不需要修改 run.py。
CONSTRAINT_MODE = "h"  # "euler": roll/yaw 自由；"h": 仅世界系 yaw 自由。
TARGET_PITCH = 0.0  # rad，夹爪参考系目标 pitch。

class Controller:
    def __init__(self, damping=0.01, target_pitch=TARGET_PITCH, constraint_mode=None):
        """damping 为 DLS 的 λ，正规方程中的阻尼项为 λ²I。"""
        self.damping = float(damping)
        self.min_singular_value = 1e-3
        self.max_task_error = 1e-3  # H 约束后的位姿误差范数，包含 m 和 rad。
        self.max_joint_change = np.deg2rad(30)  # 初始 q 到最终 q 的单轴最大变化。
        
        self.target_pitch = float(target_pitch)
        
        self.constraint_mode = CONSTRAINT_MODE if constraint_mode is None else constraint_mode
        
        self.target_rpy = np.array([0.0, self.target_pitch, 0.0])
        self.target_rotation = pin.rpy.rpyToMatrix(self.target_rpy)
        full_model = pin.buildModelFromMJCF(str(MODEL_PATH))
        # 夹爪不参与 IK，锁定两个滑动关节，配置只保留六轴角度。
        finger_ids = [full_model.getJointId(name) for name in ("joint7", "joint8")]
        self.model = pin.buildReducedModel(full_model, finger_ids, pin.neutral(full_model)) #type:ignore
        self.data = self.model.createData()
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError("模型应为六个单自由度转动关节")
        if not self.model.existFrame("ee"):
            raise ValueError("模型中缺少末端 ee")
        self.ee_id = self.model.getFrameId("ee")
        # 模型 ee 的 +Z 沿夹爪前向，+X 在图示水平姿态下朝下。
        # 构造同原点参考系：X_level=Z_ee, Y_level=Y_ee, Z_level=-X_ee。
        self.ee_to_level = pin.rpy.rpyToMatrix(0.0, -np.pi / 2, 0.0)
        self.H = np.diag(
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0] if self.constraint_mode == "euler"
            else [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        )
        self.task_dimension = np.count_nonzero(np.diag(self.H))

    def _orientation_constraint(self, level_rotation):
        """返回姿态误差和使 error_dot=-A@omega_world 的映射 A。"""
        if self.constraint_mode == "euler":
            # 只选 pitch 行，保留 roll/yaw；欧拉角在 pitch=+/-pi/2 时奇异。
            rpy = pin.rpy.matrixToRpy(level_rotation)
            error = (self.target_rpy - rpy + np.pi) % (2 * np.pi) - np.pi
            rate_map = pin.rpy.computeRpyJacobianInverse(
                rpy, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
        else:
            # h=log3(R_level R_d.T) 的 x/y 分量为零。
            # 因此 R_level=Rz(yaw) R_d，仅世界 Z 轴 yaw 自由。
            E = level_rotation @ self.target_rotation.T
            error = -pin.log3(E)
            # E_new=Exp(dw_world) E；左导数由内置右导数 Jlog3(E.T) 给出。
            rate_map = pin.Jlog3(E.T)
        return error, rate_map

    def check(self, singular_values, error, q_initial, q_final):
        """检查最终雅可比、任务残差和关节变化，决定是否接受 IK 结果。"""
        error_norm = np.linalg.norm(self.H @ error)
        if not np.isfinite(error_norm):
            print("IK 拒绝：最终位姿误差包含非有限数值")
            return False
        if error_norm > self.max_task_error:
            print(f"IK 拒绝：最终位姿误差 {error_norm:.6f} > {self.max_task_error:.6f}")
            return False

        # H 的未约束行会产生固定零奇异值，只检查任务维度内的最小值。
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

    def compute(self, q, dq, ee_position, target_position, dt):
        """根据初始化选定的模式构造姿态约束，并与 xyz 共同求解。"""
        q_initial = vector(q, 6).copy()
        q_target = q_initial.copy()
        target = vector(target_position, 3)
        # 最多更新 100 次；_=100 时重新计算最终 q 对应的误差和奇异值。
        for _ in range(101):
            pin.forwardKinematics(self.model, self.data, q_target)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_id] #type:ignore
            level_rotation = current.rotation @ self.ee_to_level
            orientation_error, orientation_rate_map = self._orientation_constraint(level_rotation)
            error = self.H @ np.r_[target - current.translation, orientation_error]
            # LWA 的线速度是末端原点速度，角速度对应 R_new=Exp(dw)@R。
            jac_lwa = pin.computeFrameJacobian(
                self.model, self.data, q_target, self.ee_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            # 固定参考系与 ee 具有相同世界系角速度，因此可沿用 LWA 雅可比。
            error_rate_map = np.eye(6)
            error_rate_map[3:, 3:] = orientation_rate_map
            jac = self.H @ error_rate_map @ jac_lwa
            # DLS: dq = V diag(σ / (σ² + λ²)) Uᵀ error。
            # 小奇异值对应的增益趋向 0，避免普通伪逆的 1/σ 放大。
            u, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
            if _ == 100 or np.linalg.norm(error) <= self.max_task_error:
                if self.check(singular_values, error, q_initial, q_target):
                    return q_target
                return q_initial

            gains = singular_values / (singular_values**2 + self.damping**2)
            dq = vt.T @ (gains * (u.T @ error))
            q_target = pin.integrate(self.model, q_target, dq * 0.1)
            q_target = np.clip(
                q_target, self.model.lowerPositionLimit, self.model.upperPositionLimit
            )
