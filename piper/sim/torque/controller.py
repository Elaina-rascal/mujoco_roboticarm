"""Pinocchio 纯重力补偿：tau = g(q)，不附加位置或速度反馈。"""

import pinocchio as pin

try:
    from .position_simulation import MODEL_PATH
except ImportError:
    from position_simulation import MODEL_PATH


class Controller:
    def __init__(self):
        # 保留夹爪关节，按当前开度计算完整机械臂的重力。
        self.model = pin.buildModelFromMJCF(str(MODEL_PATH))
        self.data = self.model.createData()
        self.v_ids = [
            self.model.joints[self.model.getJointId(f"joint{i}")].idx_v
            for i in range(1, 7)
        ]

    def compute(self, qpos):
        """输入模型全部关节位置，返回 joint1～joint6 的重力力矩（N·m）。"""
        gravity = pin.computeGeneralizedGravity(self.model, self.data, qpos)
        return gravity[self.v_ids].copy()
