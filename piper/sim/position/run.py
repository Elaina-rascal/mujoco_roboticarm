"""项目根目录运行：python3 -m piper.sim.position.run。"""

import time

from piper.sim.position.controller import PositionIKController
from piper.interface.mujoco import MujocoInterface


GRIPPER_WIDTH = 0.04  # m，双指总开度。
STEPS_PER_CONTROL = 5  # 物理 0.002 s，控制 0.01 s。


def main():
    import mujoco.viewer

    sim = MujocoInterface(mode="position")
    _, _, ee_position, _ = sim.state()
    sim.set_target(ee_position)
    sim.set_gripper(GRIPPER_WIDTH)
    controller = PositionIKController()
    dt = float(sim.model.opt.timestep) * STEPS_PER_CONTROL
    print("PiPER 夹爪 IK：约束模式和目标 pitch 在 sim/position/controller.py 中配置，绿球为位置目标。")
    print("末端仅显示 body 坐标轴：红X、绿Y、蓝Z，轴长 18 cm，随夹爪旋转。")
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
                    sim.send_joint_positions(command)
                sim.step(STEPS_PER_CONTROL)
                sim.draw_ee_frames(viewer)
                viewer.sync()
                time.sleep(max(0.0, start + float(sim.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
