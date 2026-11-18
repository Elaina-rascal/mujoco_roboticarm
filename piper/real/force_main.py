"""运行：python3 piper/real/force_main.py，MIT 固定关节目标 + 单轴附加力矩测试。"""

import time

import numpy as np

from digital_twin import DigitalTwin
from pinocchio_control import PinocchioControl
from piper_arm import Piper


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.1  # s
HOLD_POSITIONS = np.array([0.114214, 0.586248, -0.464362, 0.019862, 0.192911, -0.019862])
KP = 10.0
KD = 0.8
TEST_JOINT = 6  # 1～6，其他关节保持 PD，前馈力矩为零。
TEST_TORQUE = 0.25  # 按当前 torque_scale 换算的附加力矩 N·m。
PHASE_DURATION = 3.0  # s；依次 0、正力矩、0、负力矩、0，最后保持零前馈。


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

        next_tick = time.perf_counter()
        start_time = next_tick
        next_print = next_tick
        time.sleep(3)
        while True:
            q = piper.read_joint_positions()
            now = time.perf_counter()
            phase = min(int((now - start_time) / PHASE_DURATION), 4)
            additional_torque = np.zeros(6)
            additional_torque[TEST_JOINT - 1] = (0.0, 0.3, 0.5, 0.7, 1)[phase] * TEST_TORQUE
            piper.send_mit(HOLD_POSITIONS, kp=KP, kd=KD, torques=additional_torque)
            twin.update(q)

            if now >= next_print:
                print(f"测试阶段: {phase + 1}/5，测试关节: joint{TEST_JOINT}")
                print("关节角(rad):", q)
                print("相对固定目标的偏差(rad):", q - HOLD_POSITIONS)
                print("附加前馈力矩(N·m):", additional_torque)
                next_print = now + PRINT_PERIOD

            next_tick += CONTROL_PERIOD
            time.sleep(max(0, next_tick - time.perf_counter()))
    finally:
        twin.close()
        piper.close()


if __name__ == "__main__":
    main()
