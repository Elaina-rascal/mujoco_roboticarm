"""复用位置仿真场景，将六轴执行器改为直接力矩输入。"""

import mujoco
import numpy as np

try:
    from .position_simulation import ArmSimulation as PositionSimulation, vector
except ImportError:
    from position_simulation import ArmSimulation as PositionSimulation, vector


class ArmSimulation(PositionSimulation):
    def __init__(self):
        super().__init__()
        # 关闭 XML 中已有的自动重力补偿，由控制器计算并施加 g(q)。
        self.model.body_gravcomp[:] = 0
        ids = self.actuator_ids
        self.model.actuator_gaintype[ids] = mujoco.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_NONE
        self.model.actuator_gainprm[ids] = 0
        self.model.actuator_gainprm[ids, 0] = 1
        self.model.actuator_biasprm[ids] = 0
        # 取消原来的角度输入限幅，保留执行器原有的力矩范围。
        self.model.actuator_ctrllimited[ids] = False
        self.data.ctrl[ids] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_command(self, torque):
        """按 joint1～joint6 设置力矩，单位 N·m。"""
        limits = self.model.actuator_forcerange[self.actuator_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(
            vector(torque, 6), limits[:, 0], limits[:, 1]
        )
