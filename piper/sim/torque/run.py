"""项目根目录运行：python3 -m piper.sim.torque.run。"""

import time

try:
    from .controller import Controller
    from .simulation import ArmSimulation
except ImportError:
    from controller import Controller
    from simulation import ArmSimulation


GRIPPER_WIDTH = 0.04  # m，双指总开度。
STEPS_PER_CONTROL = 5  # 物理 0.002 s，控制 0.01 s。


def main():
    import mujoco.viewer

    sim = ArmSimulation()
    _, _, ee_position, _ = sim.state()
    sim.set_target(ee_position)
    sim.set_gripper(GRIPPER_WIDTH)
    controller = Controller()
    print("PiPER 六轴 Pinocchio 纯重力补偿：tau = g(q)，无位置保持、无速度阻尼。")
    print("可在 MuJoCo 窗口中双击选中机械臂，按 Ctrl + 鼠标右键施加扰动力。")
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [0.2, 0, 0.3]
            viewer.cam.distance = 1.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            start = time.monotonic()
            while viewer.is_running():
                torque = controller.compute(sim.data.qpos)
                sim.set_command(torque)
                sim.step(STEPS_PER_CONTROL)
                sim.draw_ee_frames(viewer)
                viewer.sync()
                time.sleep(max(0.0, start + float(sim.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
