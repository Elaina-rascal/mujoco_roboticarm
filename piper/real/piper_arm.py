"""Piper 六轴接口：位置使用 rad，力矩使用输出端 N·m。"""

from copy import deepcopy
import re
import time

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
        self.sdk.ConnectPort()
        try:
            self.firmware_version, version = self._read_firmware_version()
        except Exception:
            self.sdk.DisconnectPort()
            raise
        print(f"\033[38;2;200;162;255m[Piper] 当前主控固件：{self.firmware_version}\033[0m", flush=True)
        # Reimagine-Robotics/piper_control: 1.7-3 起取消旧 MIT 方向补偿。
        # https://github.com/Reimagine-Robotics/piper_control/blob/main/src/piper_control/piper_control.py
        self._mit_sign = np.array([-1, -1, 1, -1, 1, -1] if version < (1, 7, 3) else [1] * 6)
        if mode == "position":
            self.sdk.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        else:
            self.sdk.MotionCtrl_2(0x01, 0x04, 0, 0xAD)

    def _read_firmware_version(self, timeout=3.0):
        """等待完整的主控版本；未知版本时不猜测 MIT 方向。"""
        self.sdk.SearchPiperFirmwareVersion()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            firmware = self.sdk.GetPiperFirmwareVersion()
            match = re.fullmatch(r"S-V(\d+)\.(\d+)-(\d+)", firmware) if isinstance(firmware, str) else None
            if match:
                return firmware, tuple(map(int, match.groups()))
            time.sleep(0.05)
        raise TimeoutError("读取 Piper 主控固件版本超时，无法确定 MIT 方向补偿")

    def send_joint_positions(self, positions, speed_percent=10):
        """发送六轴目标角（rad），速度为百分比。"""
        targets = np.rint(np.rad2deg(positions) * 1000).astype(int)
        self.sdk.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
        self.sdk.JointCtrl(*targets.tolist())

    def send_mit(self, positions,torques,kp=KP, kd=KD):
        """发送六轴 MIT 指令：目标角 rad、PD 增益、前馈力矩 N·m，目标速度为零。"""
        # 当前 SDK 的 MIT t_ref 编码范围为 [-8, 8]。
        positions = np.asarray(positions) * self._mit_sign
        targets = np.clip(np.asarray(torques) / self.torque_scale, -8.0, 8.0) * self._mit_sign
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
