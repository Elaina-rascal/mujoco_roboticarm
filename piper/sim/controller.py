"""xyz IK：可选欧拉角 pitch 约束或 SO(3) h 约束。"""

import numpy as np
import pinocchio as pin
from itertools import count
if __package__:
    from .simulation import MODEL_PATH, vector
else:
    from simulation import MODEL_PATH, vector

# 在这里切换模式，不需要修改 run.py。
CONSTRAINT_MODE = "h"  # "euler": roll/yaw 自由；"h": 仅世界系 yaw 自由。
TARGET_PITCH = 0.0  # rad，夹爪参考系目标 pitch。

counter = count(1)
class Controller:
    def __init__(self, damping=0.01, target_pitch=TARGET_PITCH, constraint_mode=None):
        """damping 为 DLS 的 λ，正规方程中的阻尼项为 λ²I。"""
        self.damping = float(damping)
        if not np.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("damping 必须为有限正数")
        self.target_pitch = float(target_pitch)
        if not np.isfinite(self.target_pitch):
            raise ValueError("target_pitch 必须为有限数值，单位 rad")
        self.constraint_mode = CONSTRAINT_MODE if constraint_mode is None else constraint_mode
        if self.constraint_mode not in ("euler", "h"):
            raise ValueError('constraint_mode 必须为 "euler" 或 "h"')
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

    def compute(self, q, dq, ee_position, target_position, dt):
        """根据初始化选定的模式构造姿态约束，并与 xyz 共同求解。"""
        q_target = vector(q, 6).copy()
        target = vector(target_position, 3)
        error_threshold = 1e-3
        for _ in range(100):
            pin.forwardKinematics(self.model, self.data, q_target)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_id] #type:ignore
            level_rotation = current.rotation @ self.ee_to_level
            orientation_error, orientation_rate_map = self._orientation_constraint(level_rotation)
            error = self.H @ np.r_[target - current.translation, orientation_error]
            if np.linalg.norm(error) < error_threshold:
                break
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
            gains = singular_values / (singular_values**2 + self.damping**2)
            dq = vt.T @ (gains * (u.T @ error))
            print(f"奇异值{singular_values}") if next(counter) % 10000 == 0 else None
            q_target = pin.integrate(self.model, q_target, dq * 0.1)
            
        return np.clip(
            q_target, self.model.lowerPositionLimit, self.model.upperPositionLimit
        )
