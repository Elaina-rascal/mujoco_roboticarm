"""直接运行：python3 piper/sim/run.py。"""

import time

try:
    from .controller import Controller
    from .simulation import ArmSimulation
except ImportError:
    from controller import Controller
    from simulation import ArmSimulation


TARGET_POSITION = [0.30, -0.10, 0.35]  # m，世界系；运行后可拖动绿球。
GRIPPER_WIDTH = 0.04  # m，双指总开度。
STEPS_PER_CONTROL = 5  # 物理 0.002 s，控制 0.01 s。


def main():
    import mujoco.viewer

    sim = ArmSimulation()
    sim.set_target(TARGET_POSITION)
    sim.set_gripper(GRIPPER_WIDTH)
    controller = Controller()
    dt = float(sim.model.opt.timestep) * STEPS_PER_CONTROL
    print("PiPER 六轴位置 IK：红点为夹爪中心，绿球为目标，只跟踪位置，末端朝向自由。")
    print("双击绿球，按住 Ctrl + 鼠标右键拖动；关闭窗口或 Ctrl+C 退出。")
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [0.2, 0, 0.3]
            viewer.cam.distance = 1.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            start = time.monotonic()
            while viewer.is_running():
                q, dq, position, target = sim.state()
                command = controller.compute(q, dq, position, target, dt)
                if command is not None:
                    sim.set_command(command)
                sim.step(STEPS_PER_CONTROL)
                viewer.sync()
                time.sleep(max(0.0, start + float(sim.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
