"""双轴摇杆的力矩输入与状态接口；不包含控制算法。"""

from pathlib import Path

import mujoco
import numpy as np


class JoystickSimulation:
    """输入顺序固定为 stick_x、stick_y，使用减速器输出端力矩。"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).with_name("joystick.xml"))
        )
        self.data = mujoco.MjData(self.model)
        self.grip_id = self.model.site("grip").id
        mujoco.mj_forward(self.model, self.data)

    def set_torque(self, torque):
        """设置两个关节的输出力矩（N·m），限幅至 ±17 N·m。"""
        values = np.asarray(torque, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("torque 必须为两个有限数值，单位 N·m")
        bounds = self.model.actuator_ctrlrange
        self.data.ctrl[:] = np.clip(values, bounds[:, 0], bounds[:, 1])

    def step(self):
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        """关节角 rad、角速度 rad/s、执行器力矩 N·m、世界系握持点 m。"""
        return {
            "time": float(self.data.time),
            "joint_names": ["stick_x", "stick_y"],
            "position": self.data.qpos.copy(),
            "velocity": self.data.qvel.copy(),
            "actuator_torque": self.data.qfrc_actuator.copy(),
            "grip_position": self.data.site_xpos[self.grip_id].copy(),
        }
