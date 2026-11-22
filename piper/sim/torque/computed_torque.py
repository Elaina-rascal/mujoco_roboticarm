"""运行：python3 -m piper.sim.torque.computed_torque；IK + M(q) 关节空间控制。"""

import argparse
import time

import numpy as np
import pinocchio as pin

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.mujoco import MujocoInterface
from piper.models import OFFICIAL_MODEL


GRIPPER_WIDTH = 0.04
STEPS_PER_CONTROL = 5  # 物理 0.002 s，控制 0.01 s。
KP = np.full(6, 100.0)  # s^-2
KD = np.full(6, 20.0)  # s^-1
INITIAL_POSITION = np.array([0.249335, 0.0308932, 0.266156])


class TorqueSimulation:
    """负责目标球、IK 调度和仿真步进；力矩计算在 PinocchioControl 中。"""

    def __init__(self):
        self.sim = MujocoInterface(mode="torque")
        self.control = PinocchioControl(model_path=OFFICIAL_MODEL, gripper_width=GRIPPER_WIDTH)
        self.q_target = self.control.ik(np.zeros(6), INITIAL_POSITION)
        self.sim.data.qpos[self.sim.q_ids] = self.q_target
        self.sim.set_gripper(GRIPPER_WIDTH)
        self.target_position = self.control.fk(self.q_target).translation.copy()
        self.sim.set_target(self.target_position)

    def step(self):
        q, dq, _, target = self.sim.state()
        # 目标不变时保持同一个 IK 解，避免每周期重求目标或发生关节漂移。
        if not np.array_equal(target, self.target_position):
            candidate = self.control.ik(q, target)
            # IK 拒绝时返回输入 q，此时保留上一次接受的目标。
            if not np.array_equal(candidate, q):
                self.q_target = candidate.copy()
            self.target_position = target.copy()
        error = pin.difference(self.control.model, q, self.q_target)
        torque = self.control.computed_torque(q, dq, error, kp=KP, kd=KD)
        self.sim.send_joint_torques(torque)
        self.sim.step(STEPS_PER_CONTROL)
        return error, torque


def main():
    parser = argparse.ArgumentParser(description="PiPER IK + M(q) 关节空间力矩仿真")
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--duration", type=float, default=5.0, help="无窗口运行秒数")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration 必须为正数")
    scene = TorqueSimulation()
    sim = scene.sim
    print("IK 求 q目标 → e=q目标-q → tau=M(q)@(Kp*e-Kd*dq)+C*dq+g。")
    print("IK 约束 XYZ 与工具水平，yaw 自由；求解后六个关节跟踪选定的 IK 解。")
    if args.headless:
        while sim.data.time < args.duration:
            error, _ = scene.step()
        print(f"仿真 {sim.data.time:.3f}s；关节误差范数 {np.linalg.norm(error):.6g} rad")
        return

    import mujoco.viewer

    print("双击目标球，Ctrl+右键拖动；不可达或未通过检查的目标保持上次 IK 解。")
    print("双击机械臂后可施加扰动，松开后恢复关节目标。")
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [0.2, 0, 0.3]
            viewer.cam.distance = 1.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            start = time.monotonic()
            while viewer.is_running():
                scene.step()
                sim.draw_ee_frames(viewer)
                viewer.sync()
                time.sleep(max(0.0, start + float(sim.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
