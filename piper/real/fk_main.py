"""运行：python3 piper/real/fk_main.py，拖动数字孪生小球进行实机 IK 控制。"""

import time

import numpy as np
import pinocchio as pin

from pinocchio_control import PinocchioControl
from piper_arm import Piper
from digital_twin import DigitalTwin


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.1  # s


def main():
    piper = Piper(can_name="can_piper")
    control = PinocchioControl()
    # 等到首帧真实反馈，再确定目标球起点。
    while piper.sdk.GetArmJointMsgs().time_stamp == 0:
        time.sleep(CONTROL_PERIOD)
    q = piper.read_joint_positions()
    twin = DigitalTwin(q, control.fk(q).translation)
    np.set_printoptions(precision=6, suppress=True)
    try:
        while not piper.sdk.EnablePiper():
            time.sleep(0.01)

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
        twin.close()
        piper.close()


if __name__ == "__main__":
    main()
