"""运行：python3 -m piper.real.position.run，拖动数字孪生小球进行实机 IK 控制。"""

import time

import numpy as np
import pinocchio as pin

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.piper import PiperInterface
from piper.real.digital_twin import DigitalTwin


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.1  # s
HOLD_POSITIONS = np.array([0.114214, 0.586248, -0.464362, 0.019862, 0.192911, -0.019862])
STARTUP_TIMEOUT = 30.0  # s
STARTUP_TOLERANCE = 0.02  # rad


def main():
    piper = PiperInterface(can_name="can_piper", mode="position")
    control = PinocchioControl()
    # 等到首帧真实反馈，再确定目标球起点。
    while not piper.has_feedback():
        time.sleep(CONTROL_PERIOD)
    q = piper.read_joint_positions()
    twin = None
    np.set_printoptions(precision=6, suppress=True)
    try:
        while not piper.enable():
            time.sleep(0.01)

        print("普通位置模式，上电默认目标(rad):", HOLD_POSITIONS)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        next_print = 0.0
        while True:
            piper.send_joint_positions(HOLD_POSITIONS, speed_percent=10)
            time.sleep(CONTROL_PERIOD)
            q = piper.read_joint_positions()
            error = q - HOLD_POSITIONS
            now = time.monotonic()
            if now >= next_print:
                print("启动关节角(rad):", q)
                print("相对默认目标的偏差(rad):", error)
                next_print = now + PRINT_PERIOD
            if np.all(np.abs(error) <= STARTUP_TOLERANCE):
                break
            if now >= deadline:
                raise TimeoutError("30 秒内未到达默认关节位置，停止进入 IK 控制")

        initial_target_position = control.fk(q).translation.copy()
        twin = DigitalTwin(q, initial_target_position)
        print("已到达默认位置，开始小球 IK 控制")

        next_tick = time.perf_counter()
        next_print = next_tick
        while True:
            q = piper.read_joint_positions()
            twin.update(q)
            target = twin.read_target_position()
            q_target = control.ik(q, target)
            piper.send_joint_positions(q_target, speed_percent=10)
            transform = control.fk(q).np
            now = time.perf_counter()
            if now >= next_print:
                rpy = pin.rpy.matrixToRpy(transform[:3, :3])
                print("关节角(rad):", q)
                print("小球目标(m):", target)
                print("末端位置(m):", transform[:3, 3])
                print("末端 RPY(rad):", rpy)
                print("末端位姿 T_base_ee:\n", transform)
                next_print = now + PRINT_PERIOD

            next_tick += CONTROL_PERIOD
            time.sleep(max(0, next_tick - time.perf_counter()))
    finally:
        if twin is not None:
            twin.close()
        piper.close()


if __name__ == "__main__":
    main()
