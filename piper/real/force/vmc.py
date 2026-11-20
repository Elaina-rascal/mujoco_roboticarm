"""运行：python3 -m piper.real.force.vmc，拖动目标球控制末端 XYZ 阻抗。"""

from dataclasses import dataclass
import time

import numpy as np
import pinocchio as pin

from piper.interface.piper import PiperInterface
from piper.models import OFFICIAL_MODEL
from piper.real.digital_twin import DigitalTwin
from piper.real.force.diagnostics import diagnostic_path, record_torques
from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.base import vector


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.2  # s
VELOCITY_CUTOFF = 10.0  # Hz，实际关节角差分速度的一阶低通


KP = np.array([60.0, 60.0, 60.0])  # N/m
KD = np.array([3.0, 3.0, 3.0])  # N·s/m
MAX_FORCE = np.array([20.0, 20.0, 20.0])  # N，逐轴限幅
INITIAL_POSITION = np.array([0.249335, 0.0308932, 0.266156])


@dataclass
class ImpedanceCommand:
    torque: np.ndarray
    force: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    gravity: np.ndarray
    force_torque: np.ndarray


def compute_impedance(control, q, dq, target):
    """计算 XYZ 弹簧阻尼力并限幅，映射后在此叠加重力补偿。"""
    q, dq, target = vector(q, 6), vector(dq, 6), vector(target, 3)
    position = control.fk(q).translation.copy()
    jacobian = pin.computeFrameJacobian(
        control.model, control.data, q, control.ee_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )[:3, :].copy()
    velocity = jacobian @ dq
    force = np.clip(KP * (target - position) - KD * velocity, -MAX_FORCE, MAX_FORCE)
    gravity = control.gravity(q)
    force_torque = control.force_to_torque(q, force)
    torque = force_torque + gravity
    return ImpedanceCommand(torque, force, position, velocity, gravity,force_torque)


STARTUP_TIMEOUT = 30.0
STARTUP_TOLERANCE = 0.02  # rad


def move_to_initial_pose(piper, control):
    """调用现有 IK，以 10% 位置速度就位；工具水平，yaw 自由。"""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not piper.has_feedback():
        if time.monotonic() >= deadline:
            raise TimeoutError("等待真机关节反馈超时")
        time.sleep(CONTROL_PERIOD)
    q = piper.read_joint_positions()
    target = control.ik(q, INITIAL_POSITION,check=False)
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while True:
        # 就位完成前一直使用普通位置模式，不施加空间弹簧力。
        piper.send_joint_positions(target, speed_percent=10)
        enabled = piper.enable()
        q = piper.read_joint_positions()
        if enabled and np.all(np.abs(q - target) <= STARTUP_TOLERANCE):
            return target
        if time.monotonic() >= deadline:
            raise TimeoutError("30 秒内未到达初始末端位姿，停止进入力控制")
        time.sleep(CONTROL_PERIOD)


def main():
    log_path = diagnostic_path()
    control = PinocchioControl(model_path=OFFICIAL_MODEL)
    piper = PiperInterface(can_name="can_piper", mode="position")
    twin = None
    np.set_printoptions(precision=6, suppress=True)
    try:
        move_to_initial_pose(piper, control)
        previous_q = piper.read_joint_positions()
        twin = DigitalTwin(previous_q, INITIAL_POSITION.copy(), model_path=OFFICIAL_MODEL)
        dq = np.zeros(6)
        previous_time = time.perf_counter()
        next_tick = previous_time
        next_print = previous_time
        print("世界系 XYZ 阻抗；姿态自由；启动时先到达给定位置，工具水平、yaw 自由。")
        print("Kp(N/m):", KP, "Kd(N·s/m):", KD, "逐轴力限幅(N):", MAX_FORCE)
        print("双击目标球，Ctrl + 鼠标右键拖动；关闭窗口或 Ctrl+C 退出并保持当前位置。")
        while twin.process.is_alive() and not twin.stop.is_set():
            q = piper.read_joint_positions()
            feedback_torque = piper.read_joint_torques()
            now = time.perf_counter()
            elapsed = now - previous_time
            if elapsed > 0:
                raw_dq = (q - previous_q) / elapsed
                alpha = -np.expm1(-2 * np.pi * VELOCITY_CUTOFF * elapsed)
                dq += alpha * (raw_dq - dq)
            previous_q, previous_time = q.copy(), now
            target = twin.read_target_position()
            command = compute_impedance(control, q, dq, target)
            # 弹簧和阻尼已在 XYZ 空间实现，不再叠加关节 MIT PD 锁住姿态。
            piper.send_joint_torques(command.torque, kp=0.0, kd=0.0)
            twin.update(q)
            if now >= next_print:
                record_torques(log_path, piper, q, command.gravity, command.force_torque)
                print("目标 XYZ(m):", np.asarray(target))
                print("实际 XYZ(m):", command.position)
                print("重力力矩:",command.gravity)
                print("阻抗力矩:", command.force_torque)
                print("总力矩居:", command.torque)
                print("反馈力矩:", feedback_torque)
                print("阻抗指令 Fx/Fy/Fz(N，逐轴限幅):", command.force)
                # print("关节指令(N·m，含重力):", command.torque)
                next_print = now + PRINT_PERIOD
            next_tick += CONTROL_PERIOD
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                # 超时后从当前时间重排，不连续补发过期控制周期。
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if twin is not None:
                twin.close()
        finally:
            piper.close()


if __name__ == "__main__":
    main()
