"""运行：python3 -m piper.real.force.Mq_pd；IK + M(q) PID，积分力矩限幅。"""

import time

import numpy as np
import pinocchio as pin

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.base import vector
from piper.interface.piper_new import PiperInterface
from piper.models import OFFICIAL_MODEL
from piper.real.digital_twin import DigitalTwin
from piper.real.force.vmc import move_to_initial_pose


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.2  # s
VELOCITY_CUTOFF = 10.0  # Hz，差分关节速度的一阶低通
KP = np.full(6, 50)  # s^-2
KD = np.full(6, 5.0)  # s^-1
KI = np.full(6, 10)  # s^-3，积分项与 PD 一起乘 M(q)
INTEGRAL_TORQUE_LIMIT = 0.5  # N·m，每个关节的积分力矩上限


def main():
    control = PinocchioControl(model_path=OFFICIAL_MODEL)
    piper = PiperInterface(can_name="can_piper", mode="position")
    twin = None
    np.set_printoptions(precision=6, suppress=True)
    try:
        q_target = move_to_initial_pose(piper, control)
        previous_q = piper.read_joint_positions()
        target_position = control.fk(q_target).translation.copy()
        twin = DigitalTwin(previous_q, target_position, model_path=OFFICIAL_MODEL)
        dq = np.zeros(6)
        integral_acceleration = np.zeros(6)
        previous_time = time.perf_counter()
        next_tick = previous_time
        next_print = previous_time
        print("IK → 关节误差 → M(q) PID：tau=M(q)@(Kp*e-Kd*dq+I)+C*dq+g。")
        print("Kp(s^-2):", KP, "Kd(s^-1):", KD)
        print("Ki(s^-3):", KI, "积分力矩限幅(N·m): ±", INTEGRAL_TORQUE_LIMIT)
        print("双击目标球，Ctrl+右键拖动；关闭窗口或 Ctrl+C 退出并保持当前位置。")
        while twin.process.is_alive() and not twin.stop.is_set():
            q = piper.read_joint_positions()
            now = time.perf_counter()
            elapsed = now - previous_time
            if elapsed > 0:
                raw_dq = (q - previous_q) / elapsed
                alpha = -np.expm1(-2 * np.pi * VELOCITY_CUTOFF * elapsed)
                dq += alpha * (raw_dq - dq)
            previous_q, previous_time = q.copy(), now

            target = vector(twin.read_target_position(), 3)
            if not np.array_equal(target, target_position):
                candidate = control.ik(q, target)
                # IK 拒绝时返回输入 q，继续保持上一次接受的关节目标。
                if not np.array_equal(candidate, q):
                    q_target = candidate.copy()
                    integral_acceleration.fill(0.0)  # 新目标不继承旧目标的积分补偿。
                target_position = target.copy()
            error = pin.difference(control.model, q, q_target)
            components = control.computed_torque_components(q, dq, error, kp=KP, kd=KD)
            mass = components["mass"]
            # integral_acceleration += KI * error * max(elapsed, 0.0)
            # # 在力矩空间限幅，再反解积分状态，避免 M(q) 耦合放大和积分累积。
            # integral_torque = np.clip(
            #     mass @ integral_acceleration,
            #     -INTEGRAL_TORQUE_LIMIT, INTEGRAL_TORQUE_LIMIT,
            # )
            # integral_acceleration = np.linalg.solve(mass, integral_torque)
            torque = (
                mass @ (KP * error - KD * dq )
                + components["coriolis"] + components["gravity"]
            )
            # 保留现有 MIT 增益；分项日志仅统计上位机前馈，不含电机内部 PD。
            piper.send_joint_torques(torque,q_target, kp=0.1, kd=0.01)
            twin.update(q)
            if now >= next_print:
                print("目标 XYZ(m):", target_position)
                print("关节误差(rad):", error)
                print("M(q)·PD 力矩(N·m):", components["inertia"])
                print("g(q) 重力力矩(N·m):", components["gravity"])
                print("给定力矩(N·m，接口):", torque)
                next_print = now + PRINT_PERIOD

            next_tick += CONTROL_PERIOD
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
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
