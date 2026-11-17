"""Piper 六轴机械臂接口，关节角统一使用 rad。"""

from copy import deepcopy

import numpy as np
from piper_sdk import C_PiperInterface_V2


class Piper:
    def __init__(self, can_name="can_piper"):
        self.sdk = C_PiperInterface_V2(can_name=can_name, judge_flag=True)
        self.sdk.ConnectPort(piper_init=False)

    def send_joint_positions(self, positions, speed_percent=10):
        """发送六轴目标角（rad），速度为百分比；电机需已使能。"""
        targets = np.rint(np.rad2deg(positions) * 1000).astype(int)
        self.sdk.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
        self.sdk.JointCtrl(*targets.tolist())

    def read_joint_positions(self):
        """读取 SDK 缓存的 joint1～joint6 角度（rad）。"""
        joints = deepcopy(self.sdk.GetArmJointMsgs()).joint_state
        angles = [getattr(joints, f"joint_{i}") for i in range(1, 7)]
        return np.deg2rad(np.array(angles) * 0.001)

    def close(self):
        self.sdk.DisconnectPort()
