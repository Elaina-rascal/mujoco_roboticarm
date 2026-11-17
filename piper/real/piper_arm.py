"""Piper 六轴接口：位置使用 rad，力矩使用输出端 N·m。"""

from copy import deepcopy

import numpy as np
from piper_sdk import C_PiperInterface_V2

KP = 10.0
KD = 0.8
class Piper:
    def __init__(self, can_name="can_piper", mode="position"):
        if mode not in ("position", "mit"):
            raise ValueError('mode 需要为 "position" 或 "mit"')
        # 官方 V2 FAQ：电机执行力矩为 MIT t_ref 的 4 倍。
        self.torque_scale = 4.0
        self.sdk = C_PiperInterface_V2(can_name=can_name, judge_flag=True)
        self.sdk.ConnectPort(piper_init=False)
        if mode == "position":
            self.sdk.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        else:
            self.sdk.MotionCtrl_2(0x01, 0x04, 0, 0xAD)

    def send_joint_positions(self, positions, speed_percent=10):
        """发送六轴目标角（rad），速度为百分比。"""
        targets = np.rint(np.rad2deg(positions) * 1000).astype(int)
        self.sdk.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
        self.sdk.JointCtrl(*targets.tolist())

    def send_mit(self, positions,torques,kp=KP, kd=KD):
        """发送六轴 MIT 指令：目标角 rad、PD 增益、前馈力矩 N·m，目标速度为零。"""
        # 当前 SDK 的 MIT t_ref 编码范围为 [-8, 8]。
        targets = np.clip(np.asarray(torques) / self.torque_scale, -8.0, 8.0)
        kp = np.broadcast_to(kp, (6,))
        kd = np.broadcast_to(kd, (6,))
        self.sdk.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
        for i in range(6):
            self.sdk.JointMitCtrl(
                i + 1, float(positions[i]), 0.0,
                float(kp[i]), float(kd[i]), float(targets[i]),
            )

    def read_joint_positions(self):
        """读取 SDK 缓存的 joint1～joint6 角度（rad）。"""
        joints = deepcopy(self.sdk.GetArmJointMsgs()).joint_state
        angles = [getattr(joints, f"joint_{i}") for i in range(1, 7)]
        return np.deg2rad(np.array(angles) * 0.001)

    def close(self):
        self.sdk.DisconnectPort()
