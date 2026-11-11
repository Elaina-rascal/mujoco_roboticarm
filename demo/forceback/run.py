#!/usr/bin/env python3
"""双轴阻抗与重力补偿；可在 MuJoCo 查看器中用鼠标施加外力。"""

import argparse
import time

from simulation import JoystickSimulation
from control import Controller




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--seconds", type=float, default=5.0, help="无窗口仿真时长（秒）")

    args = parser.parse_args()
    if not 0 < args.seconds < float("inf"):
        parser.error("--seconds 必须为正的有限数值")
    sim = JoystickSimulation()
    controller = None

    controller = Controller(stiffness=0.0, damping=0.0)
    sim.set_torque([0.0, 0.0])
    if args.headless:
        while sim.data.time < args.seconds:
            if controller is not None:
                tau = controller.update(sim.data.qpos, sim.data.qvel, time=sim.data.time)
                sim.set_torque(tau)
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
                with viewer.lock():
                    if controller is not None:
                        tau = controller.update(sim.data.qpos, sim.data.qvel, time=sim.data.time)
                        sim.set_torque(tau)
                    sim.step()
                viewer.sync()
                time.sleep(max(0.0, start + sim.data.time - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
