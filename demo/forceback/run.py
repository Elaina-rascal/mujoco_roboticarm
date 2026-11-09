#!/usr/bin/env python3
"""零力矩交互查看；可在 MuJoCo 查看器中用鼠标施加外力。"""

import argparse
import time

from simulation import JoystickSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--seconds", type=float, default=5.0, help="无窗口仿真时长（秒）")
    args = parser.parse_args()
    if not 0 < args.seconds < float("inf"):
        parser.error("--seconds 必须为正的有限数值")
    sim = JoystickSimulation()
    sim.set_torque([0.0, 0.0])
    if args.headless:
        while sim.data.time < args.seconds:
            sim.step()
        print(sim.state())
        return

    import mujoco.viewer

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [-0.025, 0, 0.16]
            viewer.cam.distance = 0.65
            viewer.cam.azimuth = 70
            viewer.cam.elevation = -15
            start = time.monotonic()
            while viewer.is_running():
                # 在此处用 sim.set_torque([tau_x, tau_y]) 接入未来的控制器。
                # 当前不覆盖 ctrl，允许通过查看器 Control 面板手动调节力矩。
                with viewer.lock():
                    sim.step()
                viewer.sync()
                time.sleep(max(0.0, start + sim.data.time - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
