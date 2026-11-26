"""重力补偿后逐轴测量双向启动摩擦，不读取反馈力矩。"""

import time

import numpy as np

from piper.interface.base import vector


class FrictionEstimator:
    def __init__(self, arm, control):
        self.arm = arm
        self.clock = time.monotonic
        self.sleep = time.sleep
        self.control = control
        self.torque_step = 0.02  # N·m
        self.max_torque = 1.0  # N·m，重力之外的附加力矩上限
        self.ramp_interval = 0.2  # s，每档等待时间
        self.speed = np.deg2rad(0.5)  # rad/s，启动后的目标速度
        self.max_speed = np.deg2rad(200)  # rad/s，超速停止
        self.travel = np.deg2rad(3)  # rad，启动后继续转动的行程
        self.period = 0.005  # s
        self.timeout = 60.0  # s，单向测量或回位最长时间

    def _check(self, q, dq):
        if np.max(np.abs(dq)) > self.max_speed:
            print("当前速度(rad/s):", dq)
            raise RuntimeError("关节超速，停止辨识")
        lower = self.control.model.lowerPositionLimit
        upper = self.control.model.upperPositionLimit
        margin = np.deg2rad(2)
        if np.any(q < lower + margin) or np.any(q > upper - margin):
            print("当前速度(rad/s):", dq)
            raise RuntimeError("关节接近限位，停止辨识")

    def _direction(self, reference, joint, direction):
        start = vector(self.arm.read_joint_positions(), 6).copy()
        previous_q = start.copy()
        begin = self.clock()
        previous_time = begin
        next_ramp = begin
        extra = 0.0
        threshold = None
        start_angle = start[joint]

        while True:
            self.sleep(self.period)
            now = self.clock()
            dt = now - previous_time
            if dt <= 0 or dt > 0.1:
                raise RuntimeError("控制周期中断，停止辨识")
            q = vector(self.arm.read_joint_positions(), 6).copy()
            dq = (q - previous_q) / dt
            previous_q = q.copy()
            previous_time = now
            self._check(q, dq)
            if now - begin > self.timeout:
                raise TimeoutError("单向辨识超时")
            distance = direction * (q[joint] - start[joint])
            if distance < -np.deg2rad(0.3):
                raise RuntimeError("测试关节向相反方向移动，停止辨识")
            for i in range(6):
                if i != joint and abs(q[i] - reference[i]) > np.deg2rad(2):
                    raise RuntimeError("其他关节偏离保持位置，停止辨识")

            if threshold is None:
                # 同时看到角度变化和运动速度，才认为克服了静摩擦。
                if distance > np.deg2rad(0.1) and direction * dq[joint] > np.deg2rad(0.1):
                    threshold = direction * extra
                    start_angle = q[joint]
                elif abs(dq[joint]) < np.deg2rad(0.1) and now >= next_ramp:
                    extra += self.torque_step
                    if extra > self.max_torque + 1e-9:
                        raise RuntimeError("达到附加力矩上限仍未启动")
                    next_ramp = now + self.ramp_interval

            if threshold is not None:
                if direction * (q[joint] - start_angle) >= self.travel:
                    return threshold

            # 每周期更新重力，其他五轴用 PD 保持，测试轴不加位置弹簧。
            gravity = vector(self.control.gravity(q), 6).copy()
            torque = gravity.copy()
            torque += 15.0 * (reference - q) - 1.0 * dq
            drive = direction * extra
            if threshold is not None:
                drive += 20.0 * (direction * self.speed - dq[joint])
            elif direction * dq[joint] > self.speed:
                # 尚未确认启动时也防止越转越快，但不额外助推启动。
                drive -= 20.0 * (dq[joint] - direction * self.speed)
            if abs(drive) > self.max_torque + 1e-9:
                raise RuntimeError("调速力矩超过附加力矩上限")
            torque[joint] = gravity[joint] + drive
            if np.any(np.abs(torque) > self.arm.torque_limits):
                raise RuntimeError("总力矩超过接口范围，停止辨识")
            self.arm.send_joint_torques(torque, positions=reference, kp=0.0, kd=0.0)

    def _return(self, reference):
        """位置模式下缓慢回到同一起点，确保双向测量构型一致。"""
        command = vector(self.arm.read_joint_positions(), 6).copy()
        previous_q = command.copy()
        begin = previous_time = self.clock()
        settled = 0.0
        while True:
            step = self.speed * self.period
            command += np.clip(reference - command, -step, step)
            self.arm.send_joint_positions(command, speed_percent=5)
            self.sleep(self.period)
            now = self.clock()
            dt = now - previous_time
            if dt <= 0 or dt > 0.1:
                raise RuntimeError("回位控制周期中断")
            q = vector(self.arm.read_joint_positions(), 6).copy()
            dq = (q - previous_q) / dt
            previous_q = q.copy()
            previous_time = now
            self._check(q, dq)
            if now - begin > self.timeout:
                raise TimeoutError("回位超时")
            if np.max(abs(q - reference)) < np.deg2rad(0.1) and np.max(abs(dq)) < np.deg2rad(0.1):
                settled += dt
                if settled >= 0.5:
                    return
            else:
                settled = 0.0

    def measure(self, joint):
        """joint 为 1～6，返回有符号双向启动阈值及摩擦估计，单位 N·m。"""
        if type(joint) is not int or joint < 1 or joint > 6:
            raise ValueError("joint 需要为 1～6")
        index = joint - 1
        reference = vector(self.arm.read_joint_positions(), 6).copy()
        self._check(reference, np.zeros(6))
        margin = self.travel + np.deg2rad(4)
        lower = self.control.model.lowerPositionLimit[index]
        upper = self.control.model.upperPositionLimit[index]
        if reference[index] - margin <= lower or reference[index] + margin >= upper:
            raise ValueError("当前姿态没有足够的双向测试行程")
        try:
            self._return(reference)
            positive = self._direction(reference, index, 1)
            self._return(reference)
            negative = self._direction(reference, index, -1)
            self._return(reference)
            return {"positive": positive, "negative": negative,
                    "friction": (positive - negative) / 2,
                    "bias": (positive + negative) / 2}
        finally:
            self.arm.send_joint_positions(self.arm.read_joint_positions(), speed_percent=5)
