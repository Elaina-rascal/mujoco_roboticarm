"""运行：python3 piper/real/fk_main.py，读取关节角并计算 FK。"""

import time

import numpy as np
import pinocchio as pin

from fk import ForwardKinematics
from piper_arm import Piper


CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.1  # s


def main():
    piper = Piper(can_name="can_piper")
    fk = ForwardKinematics()
    np.set_printoptions(precision=6, suppress=True)
    next_tick = time.perf_counter()
    next_print = next_tick

    try:
        while True:
            q = piper.read_joint_positions()
            transform = fk.compute(q).np
            now = time.perf_counter()
            if now >= next_print:
                rpy = pin.rpy.matrixToRpy(transform[:3, :3])
                print("关节角(rad):", q)
                print("末端位置(m):", transform[:3, 3])
                print("末端 RPY(rad):", rpy)
                print("末端位姿 T_base_ee:\n", transform)
                next_print = now + PRINT_PERIOD

            next_tick += CONTROL_PERIOD
            time.sleep(max(0, next_tick - time.perf_counter()))
    finally:
        piper.close()


if __name__ == "__main__":
    main()
