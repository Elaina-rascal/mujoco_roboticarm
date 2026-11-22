"""运行：python3 -m piper.real.force.Mq_pd；IK + M(q) 关节空间 PD。"""

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
KP = np.full(6, 60.0)  # s^-2
KD = np.full(6, 5.0)  # s^-1


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
        previous_time = time.perf_counter()
        next_tick = previous_time
        next_print = previous_time
        print("IK → 关节误差 → M(q) PD：tau=M(q)@(Kp*e-Kd*dq)+C*dq+g。")
        print("Kp(s^-2):", KP, "Kd(s^-1):", KD)
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
                target_position = target.copy()
            error = pin.difference(control.model, q, q_target)
            torque = control.computed_torque(q, dq, error, kp=KP, kd=KD)
            # PD 已在上位机通过 M(q) 计算，MIT 只发送前馈，不重复叠加 PD。
            piper.send_joint_torques(torque,q_target, kp=0.1, kd=0.01)
            twin.update(q)
            if now >= next_print:
                print("目标 XYZ(m):", target_position)
                print("目标关节角(rad):", q_target)
                print("关节误差(rad):", error)
                print("估计关节速度(rad/s):", dq)
                print("给定力矩(N·m，接口限幅前):", torque)
                print("反馈力矩(N·m):", piper.read_joint_torques())
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
