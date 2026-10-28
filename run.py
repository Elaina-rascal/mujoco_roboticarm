#!/usr/bin/env python3
"""直接运行机械臂仿真，控制逻辑在本脚本中修改。"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mujoco_arm.simulation import MujocoArm
import mujoco.viewer


# 可选 position、torque、ik。
MODE = "ik"


def main():
    arm = MujocoArm("torque" if MODE == "torque" else "position")
    controller = None
    if MODE == "ik":
        from control_py.ik_controller import IKController
        controller = IKController(center=(0.35, 0.15, 0.5))
    elif MODE == "torque":
        arm.set_control([0, 0, 0, 0, 0, 0])
    elif MODE != "position":
        raise ValueError(f"未知控制模式：{MODE}")
    # 位置模式默认保持 home 姿态，可在这里用 arm.set_control([...]) 设置目标角。

    next_ik = 0.0
    dt = float(arm.model.opt.timestep)
    try:
        with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
            start = time.monotonic()
            while viewer.is_running():
                if controller is not None and arm.data.time >= next_ik:
                    state = arm.joint_state()
                    arm.set_control(controller.update(state["name"], state["position"], state["time"]))
                    next_ik = float(arm.data.time) + max(0.02, dt)
                arm.step()
                viewer.sync()
                time.sleep(max(0.0, start + float(arm.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
