"""Pinocchio 六轴位姿 IK，基于 SVD 的阻尼最小二乘。"""

import numpy as np
import pinocchio as pin
from itertools import count
if __package__:
    from .simulation import MODEL_PATH, vector
else:
    from simulation import MODEL_PATH, vector

counter = count(1)
class Controller:
    def __init__(self, damping=0.01):
        """damping 为 DLS 的 λ，正规方程中的阻尼项为 λ²I。"""
        self.damping = float(damping)
        if not np.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("damping 必须为有限正数")
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
        self.target_rotation = None  # 首次控制时记录初始朝向，也可直接设置 3×3 目标旋转矩阵。

    def compute(self, q, dq, ee_position, target_position, dt):
        """返回六轴目标角；目标位置为世界系，朝向由 target_rotation 指定。"""
        q_target = vector(q, 6).copy()
        if self.target_rotation is None:
            pin.forwardKinematics(self.model, self.data, q_target)
            pin.updateFramePlacements(self.model, self.data)
            self.target_rotation = self.data.oMf[self.ee_id].rotation.copy() #type:ignore
        
        target = pin.SE3( np.eye(3), vector(target_position, 3))
        error_threshold = 1e-3
        for _ in range(100):
            pin.forwardKinematics(self.model, self.data, q_target)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_id] #type:ignore
            error = pin.log6(target * current.inverse()).vector
            if np.linalg.norm(error) < error_threshold:
                break
            # 与世界系旋量误差配套，每次在当前迭代角度处计算雅可比。
            jac = pin.computeFrameJacobian(
                self.model, self.data, q_target, self.ee_id, pin.ReferenceFrame.WORLD
            )
            # DLS: dq = V diag(σ / (σ² + λ²)) Uᵀ error。
            # 小奇异值对应的增益趋向 0，避免普通伪逆的 1/σ 放大。
            u, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
            gains = singular_values / (singular_values**2 + self.damping**2)
            dq = vt.T @ (gains * (u.T @ error))
            print(f"奇异值{singular_values}") if next(counter) % 10000 == 0 else None
            q_target = pin.integrate(self.model, q_target, dq * 0.1)
            
        q_clip = np.clip(
                        q_target, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        dq=q_target-q_clip
        for q in zip(q_clip,q_target):
            pin.forwardKinematics(self.model, self.data, q_clip)
            pin.updateFramePlacements(self.model, self.data)
            error=pin.log6(target * self.data.oMf[self.ee_id].inverse()).vector
        
        return q_clip
