"""使用 pyAgxArm 的 Piper 六轴接口：角度 rad，输出端力矩 N·m。"""

from copy import deepcopy
import re
import time

import numpy as np

from piper.interface.base import ArmInterface, vector


class PiperInterface(ArmInterface):
    """标准版 Piper：仅允许固件严格高于 1.8.8 使用新 SDK。"""

    MIT_SIGN = np.ones(6)

    def __init__(self, can_name="can_piper", mode="position"):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile

        if mode not in ("position", "torque"):
            raise ValueError('mode 需要为 "position" 或 "torque"')
        self.sdk_profile = "v189"
        config = create_agx_arm_config(
            robot="piper", firmeware_version=self.sdk_profile, channel=can_name,
            interface="socketcan", bitrate=1_000_000, auto_connect=False,
            local_loopback=True,
        )
        self.sdk = AgxArmFactory.create_arm(config)
        self._closed = False
        self.torque_limits = 16 * np.asarray(config["joint_torque_c"])
        try:
            self.sdk.connect()
            self.firmware_version = self._read_firmware_version()
            match = re.fullmatch(r"S-V(\d+)\.(\d+)-(\d+)", self.firmware_version)
            if match is None:
                raise ValueError(f"无法识别主控固件版本：{self.firmware_version!r}")
            if tuple(map(int, match.groups())) <= (1, 8, 8):
                raise NotImplementedError(
                    f"新 API 仅允许固件 > 1.8.8，当前为 {self.firmware_version}"
                )
            self.sdk_profile = resolve_firmware_profile("piper", self.firmware_version)
            # 每组六轴命令只发送一次模式，避免 move_mit 各自重复切换模式。
            self.sdk.set_auto_set_motion_mode_enabled(False)
            self.sdk.set_speed_percent(10)
            self.sdk.set_motion_mode("j" if mode == "position" else "mit")
        except BaseException:
            self.sdk.disconnect()
            self._closed = True
            raise
        print(f"[Piper/pyAgxArm] 主控固件：{self.firmware_version}，SDK 配置：{self.sdk_profile}", flush=True)

    def _read_firmware_version(self, timeout=3.0):
        """由 SDK 查询并解析版本，本层仅读取返回的软件版本字段。"""
        firmware = self.sdk.get_firmware(timeout=timeout)
        if not firmware:
            raise TimeoutError("SDK 读取主控固件版本超时，未进入控制模式")
        version = firmware.get("software_version")
        if not isinstance(version, str):
            raise ValueError(f"SDK 未返回有效软件版本：{version!r}")
        return version.strip("\x00\r\n ")

    def send_joint_positions(self, positions, speed_percent=10):
        targets = vector(positions, 6)
        if not isinstance(speed_percent, int) or not 0 <= speed_percent <= 100:
            raise ValueError("speed_percent 需要为 0～100 的整数")
        self.sdk.set_speed_percent(speed_percent)
        self.sdk.set_motion_mode("j")
        self.sdk.move_j(targets.tolist())

    def send_joint_torques(self, torques, positions=None, kp=0.0, kd=0.0):
        requested = vector(torques, 6)
        positions = np.zeros(6) if positions is None else vector(positions, 6)
        kp = vector(np.broadcast_to(kp, (6,)), 6)
        kd = vector(np.broadcast_to(kd, (6,)), 6)
        if np.any(np.abs(positions) > 12.5):
            raise ValueError("MIT 目标角需要在 [-12.5, 12.5] rad 内")
        if np.any((kp < 0) | (kp > 500)) or np.any((kd < -5) | (kd > 5)):
            raise ValueError("MIT kp 需要在 [0, 500]，kd 在 [-5, 5] 内")
        targets = np.clip(requested, -self.torque_limits, self.torque_limits)
        p_des = positions * self.MIT_SIGN
        t_ff = targets * self.MIT_SIGN
        self.sdk.set_motion_mode("mit")
        for i in range(6):
            # t_ff 已是输出端 N·m。v189 SDK 内部执行 /c 和编码，不能再 /4。
            self.sdk.move_mit(
                i + 1, p_des=float(p_des[i]), v_des=0.0,
                kp=float(kp[i]), kd=float(kd[i]), t_ff=float(t_ff[i]),
            )

    def enable(self):
        return self.sdk.enable()

    def has_feedback(self):
        sample = self.sdk.get_joint_angles()
        return sample is not None and sample.timestamp > 0

    def read_joint_positions(self):
        sample = deepcopy(self.sdk.get_joint_angles())
        if sample is None or sample.timestamp <= 0:
            raise RuntimeError("尚未收到 pyAgxArm 关节角反馈")
        return vector(sample.msg, 6).copy()

    def read_joint_torques(self):
        torques = np.full(6, np.nan)
        for i in range(6):
            sample = deepcopy(self.sdk.get_motor_states(i + 1))
            if sample is not None:
                torques[i] = sample.msg.torque  # SDK 已换算 N·m；不重复乘系数。
        return torques

    def read_status_sample(self):
        sample = deepcopy(self.sdk.get_arm_status())
        if sample is None:
            return {"status": None, "status_timestamp": None}
        return {
            "status": {
                "ctrl_mode": int(sample.msg.ctrl_mode),
                "mode_feed": int(sample.msg.mode_feedback),
                "arm_status": int(sample.msg.arm_status),
                "err_code": int(sample.msg.err_code),
            },
            "status_timestamp": sample.timestamp,
        }

    def close(self):
        if self._closed:
            return
        try:
            hold_positions = self.read_joint_positions()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                self.send_joint_positions(hold_positions, speed_percent=10)
                time.sleep(0.02)
                sample = self.read_status_sample()
                if sample["status"] is not None and sample["status"]["mode_feed"] == 0x01:
                    print("[Piper/pyAgxArm] 已切换到位置模式，保持当前位置", flush=True)
                    return
            raise TimeoutError("退出时未确认切换到位置模式")
        finally:
            self.sdk.disconnect()
            self._closed = True
