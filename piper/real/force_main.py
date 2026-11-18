"""运行：python3 piper/real/force_main.py，MIT 保持启动时当前位置并补偿重力。"""

import time

import numpy as np

from digital_twin import DigitalTwin
from pinocchio_control import PinocchioControl
from piper_arm import Piper


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.1  # s
HOLD_POSITIONS = np.array([0.114214, 0.586248, -0.464362, 0.019862, 0.192911, -0.019862])
KP = 1.0
KD = 0.01


def main():
    piper = Piper(can_name="can_piper", mode="mit")
    control = PinocchioControl()
    while piper.sdk.GetArmJointMsgs().time_stamp == 0:
        time.sleep(CONTROL_PERIOD)
    q = piper.read_joint_positions()
    twin = DigitalTwin(q, control.fk(q).translation)
    np.set_printoptions(precision=6, suppress=True)

    try:
        while not piper.sdk.EnablePiper():
            time.sleep(0.01)

        # 使能后读取当前姿态，后续固定保持该目标。
        hold_positions = piper.read_joint_positions().copy()
        print("保持目标(rad):", hold_positions)
        print("力矩反馈为电流估算幅值；包含 PD 输出，幅值差不等于力矩跟踪误差。")
        next_tick = time.perf_counter()
        next_print = next_tick
        while True:
            q = piper.read_joint_positions()
            now = time.perf_counter()
            gravity_torque = control.gravity(q)
            # gravity() 返回输出端 N·m；缩放与方向补偿由 Piper 统一处理。
            piper.send_mit(HOLD_POSITIONS, kp=KP, kd=KD, torques=gravity_torque)
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
