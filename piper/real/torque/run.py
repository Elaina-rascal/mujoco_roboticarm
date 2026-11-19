"""运行：python3 -m piper.real.torque.run，MIT 位置保持并补偿重力。"""

import time
import numpy as np

from piper.controllers.pinocchio_control import PinocchioControl
from piper.interface.piper import PiperInterface
from piper.models import OFFICIAL_MODEL
from piper.real.digital_twin import DigitalTwin


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.1  # s
HOLD_POSITIONS = np.array([0.114214, 0.586248, -0.464362, 0.019862, 0.192911, -0.019862])
KP = 5.0
KD = 0.01


def main():
    piper = PiperInterface(can_name="can_piper", mode="torque")
    control = PinocchioControl(model_path=OFFICIAL_MODEL)
    while not piper.has_feedback():
        time.sleep(CONTROL_PERIOD)
    q = piper.read_joint_positions()
    twin = DigitalTwin(q, control.fk(q).translation, model_path=OFFICIAL_MODEL)
    np.set_printoptions(precision=6, suppress=True)

    try:
        while not piper.enable():
            time.sleep(0.01)

        # 沿用原入口的固定保持目标与 PD 增益。
        print("保持目标(rad):", HOLD_POSITIONS)
        print("力矩反馈为电流估算幅值；包含 PD 输出，幅值差不等于力矩跟踪误差。")
        next_tick = time.perf_counter()
        next_print = next_tick
        while True:
            q = piper.read_joint_positions()
            now = time.perf_counter()
            gravity_torque = control.gravity(q)
            # gravity() 返回输出端 N·m；缩放与方向补偿由真机接口处理。
            piper.send_joint_torques(gravity_torque, positions=HOLD_POSITIONS, kp=KP, kd=KD)
            twin.update(q)

            if now >= next_print:
                feedback_torque = piper.read_joint_torques()
                feedforward_magnitude = np.abs(gravity_torque)
                print("关节角(rad):", q)
                print("相对保持目标的偏差(rad):", q - HOLD_POSITIONS)
                print("前馈幅值(N·m):", feedforward_magnitude)
                print("反馈幅值(N·m):", feedback_torque)
                print("其他力矩(N·m):", feedback_torque - feedforward_magnitude)
                next_print = now + PRINT_PERIOD

            next_tick += CONTROL_PERIOD
            time.sleep(max(0, next_tick - time.perf_counter()))
    finally:
        twin.close()
        piper.close()


if __name__ == "__main__":
    main()
