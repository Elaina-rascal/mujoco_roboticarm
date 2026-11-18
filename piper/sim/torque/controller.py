"""Pinocchio 纯重力补偿：tau = g(q)，不附加位置或速度反馈。"""

from pathlib import Path

import pinocchio as pin


class Controller:
    def __init__(self):
        model_path = Path(__file__).resolve().parents[1] / "arm.xml"
        # 保留夹爪关节，按当前开度计算完整机械臂的重力。
        self.model = pin.buildModelFromMJCF(str(model_path))
        self.data = self.model.createData()
        self.v_ids = [
            self.model.joints[self.model.getJointId(f"joint{i}")].idx_v
            for i in range(1, 7)
        ]

    def compute(self, qpos):
        """输入模型全部关节位置，返回 joint1～joint6 的重力力矩（N·m）。"""
        gravity = pin.computeGeneralizedGravity(self.model, self.data, qpos)
        return gravity[self.v_ids].copy()
