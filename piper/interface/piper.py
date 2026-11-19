"""Piper 六轴接口：位置使用 rad，力矩使用输出端 N·m。"""

from copy import deepcopy
import re
import time

import numpy as np

from piper.interface.base import ArmInterface, vector


class PiperInterface(ArmInterface):
    def __init__(self, can_name="can_piper", mode="position"):
        from piper_sdk import C_PiperInterface_V2

        if mode not in ("position", "torque"):
            raise ValueError('mode 需要为 "position" 或 "torque"')
        self.sdk = C_PiperInterface_V2(can_name=can_name, judge_flag=True)
        self.sdk.ConnectPort()
        try:
            self.firmware_version, _ = self._read_firmware_version()
        except Exception:
            self.sdk.DisconnectPort()
            raise
        print(f"\033[38;2;200;162;255m[Piper] 当前主控固件：{self.firmware_version}\033[0m", flush=True)
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
        targets = np.rint(np.rad2deg(vector(positions, 6)) * 1000).astype(int)
        self.sdk.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
        self.sdk.JointCtrl(*targets.tolist())

    def send_joint_torques(self, torques, positions=None, kp=0.0, kd=0.0):
        """MIT 发送六轴力矩；真机可额外传入目标角与 PD 增益，默认纯前馈。"""
        if self.firmware_version != "S-V1.5-8":
            raise NotImplementedError(f"MIT 换算仅适配 S-V1.5-8，当前为 {self.firmware_version}")
        # 旧固件 J1/J2/J4/J6 反号；保留当前 1.5-8 前馈除以 4 的换算。
        mit_sign = np.array([-1, -1, 1, -1, 1, -1])
        # 当前 SDK 的 MIT t_ref 编码范围为 [-8, 8]。
        positions = np.zeros(6) if positions is None else vector(positions, 6)
        positions = positions * mit_sign
        targets = np.clip(vector(torques, 6) / 4.0, -8.0, 8.0) * mit_sign
        kp = vector(np.broadcast_to(kp, (6,)), 6)
        kd = vector(np.broadcast_to(kd, (6,)), 6)
        self.sdk.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
        for i in range(6):
            self.sdk.JointMitCtrl(
                i + 1, float(positions[i]), 0.0,
                float(kp[i]), float(kd[i]), float(targets[i]),
            )
        self.last_mit_t_ref = targets.copy()

    def enable(self):
        """真机使能，返回 SDK 的使能结果。"""
        return self.sdk.EnablePiper()

    def has_feedback(self):
        return self.sdk.GetArmJointMsgs().time_stamp != 0

    def read_joint_positions(self):
        """读取 SDK 缓存的 joint1～joint6 角度（rad）。"""
        joints = deepcopy(self.sdk.GetArmJointMsgs()).joint_state
        angles = [getattr(joints, f"joint_{i}") for i in range(1, 7)]
        return np.deg2rad(np.array(angles) * 0.001)

    def read_joint_torques(self):
        """标准版六轴电流估算力矩（N·m），保留 SDK 符号；缺失轴为 NaN。

        仅适配 S-V1.5-8。沿用现行官方 k*b*c 系数，尚未做真机负载
        和方向标定；不使用旧 SDK effort，也不套用 MIT 发送反号。
        """
        if self.firmware_version != "S-V1.5-8":
            raise NotImplementedError(f"力矩反馈换算仅适配 S-V1.5-8，当前为 {self.firmware_version}")
        feedback = deepcopy(self.sdk.GetArmHighSpdInfoMsgs())
        current_ma = np.full(6, np.nan)
        for i in range(6):
            motor = getattr(feedback, f"motor_{i + 1}")
            if motor.can_id == 0x251 + i:
                current_ma[i] = float(motor.current)
        # 标准版 PIPER 专用，不能用于 PIPER-H/L/X。
        # https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/api/constants.py
        k = np.array([1.18125, 1.18125, 1.18125, 0.9576, 0.9576, 0.9576])
        b = np.array([4., 4., 4., 1., 1., 1.])
        c = np.array([1., 1., 1., 0.813252, 0.813252, 0.813252])
        return current_ma * 0.001 * k * b * c

    def close(self):
        """以当前位置作为普通位置模式目标，确认模式反馈后断开连接。"""
        try:
            if self.sdk.GetArmJointMsgs().time_stamp == 0:
                raise RuntimeError("没有关节反馈，无法设置退出保持位置")
            hold_positions = self.read_joint_positions()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                self.send_joint_positions(hold_positions, speed_percent=10)
                time.sleep(0.02)
                if self.sdk.GetArmStatus().arm_status.mode_feed == 0x01:
                    print("\033[38;2;200;162;255m[Piper] 已切换到位置模式，保持当前位置\033[0m", flush=True)
                    return

            raise TimeoutError("退出时未确认切换到位置模式；旧固件可能不支持直接从 MIT 切换")

        finally:
            self.sdk.DisconnectPort()
