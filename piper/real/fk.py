"""六轴关节角到夹爪中心相对于 base_link 的位姿。"""

from pathlib import Path

import numpy as np
import pinocchio as pin


MODEL_PATH = Path(__file__).resolve().parents[1] / "sim" / "arm.xml"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))


class ForwardKinematics:
    def __init__(self):
        full_model = pin.buildModelFromMJCF(str(MODEL_PATH))
        fingers = [full_model.getJointId(name) for name in ("joint7", "joint8")]
        self.model = pin.buildReducedModel(full_model, fingers, pin.neutral(full_model)) #type:ignore
        self.data = self.model.createData()
        if not self.model.existFrame("ee"):
            raise ValueError("模型中缺少末端 ee")
        self.ee_id = self.model.getFrameId("ee")
        self.joint_ids = [self.model.getJointId(name) for name in JOINT_NAMES]
        self.q_ids = [self.model.joints[joint_id].idx_q for joint_id in self.joint_ids] #type:ignore

    def compute(self, joint_positions):
        """输入 joint1～joint6 的角度（rad），返回 4×4 位姿（平移单位 m）。

        ee 位于 link6 局部 [0, 0, 0.13503]，与仿真使用相同工具中心。
        """
        positions = np.asarray(joint_positions, dtype=float)
        if positions.shape != (6,) or not np.all(np.isfinite(positions)):
            raise ValueError("需要六个有限的关节角，单位 rad")
        q = pin.neutral(self.model)
        q[self.q_ids] = positions
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_id] #type:ignore
