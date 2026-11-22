"""运行：python3 -m piper.sim.impedance.run；只运行仿真。"""

import argparse
import time

import numpy as np

from piper.controllers.impedance_controller import ImpedanceController
from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.mujoco import MujocoInterface
from piper.models import OFFICIAL_MODEL


GRIPPER_WIDTH = 0.04
STEPS_PER_CONTROL = 1  # 500 Hz，与物理步长一致。
INITIAL_JOINT_POSITIONS = np.array([0.0, 1.0, -1.1, 0.0, 0.4, 0.0])


def create_simulation():
    sim = MujocoInterface(mode="torque")
    # home 的 joint5=0 位于腕部奇异位形，六自由度演示从非奇异位形开始。
    sim.data.qpos[sim.q_ids] = INITIAL_JOINT_POSITIONS
    sim.set_gripper(GRIPPER_WIDTH)
    control = PinocchioControl(model_path=OFFICIAL_MODEL, gripper_width=GRIPPER_WIDTH)
    controller = ImpedanceController(control=control)
    pose = control.fk(sim.read_joint_positions())
    sim.set_target_pose(pose.translation, pose.rotation)
    return sim, controller


def control_step(sim, controller):
    q, dq, _, _ = sim.state()
    position, rotation = sim.target_pose()
    command = controller.compute(q, dq, position, rotation)
    sim.send_joint_torques(command.torque)
    sim.step(STEPS_PER_CONTROL)
    return command


def main():
    parser = argparse.ArgumentParser(description="PiPER 六自由度笛卡尔阻抗仿真")
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--duration", type=float, default=5.0, help="无窗口运行秒数，默认 5")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration 必须为正数")
    sim, controller = create_simulation()
    print("六自由度阻抗：世界系位置误差 + Log(R目标 @ R当前.T) 姿态误差。")
    print("Kp [N/m, N·m/rad]:", controller.kp, "Kd [N·s/m, N·m·s/rad]:", controller.kd)
    if args.headless:
        while sim.data.time < args.duration:
            command = control_step(sim, controller)
        print(f"仿真 {sim.data.time:.3f}s；位置误差 {np.linalg.norm(command.error[:3]):.6g} m；"
              f"姿态误差 {np.linalg.norm(command.error[3:]):.6g} rad")
        return

    import mujoco.viewer

    print("双击目标球选中；Ctrl+右键拖动位置，Ctrl+左键旋转姿态（按 Shift 切换操作平面）。")
    print("较长坐标轴为实际姿态，较短坐标轴为目标姿态；选中机械臂可施加扰动。")
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [0.2, 0, 0.3]
            viewer.cam.distance = 1.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            start = time.monotonic()
            while viewer.is_running():
                control_step(sim, controller)
                sim.draw_ee_frames(viewer, draw_target=True)
                viewer.sync()
                time.sleep(max(0.0, start + float(sim.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
