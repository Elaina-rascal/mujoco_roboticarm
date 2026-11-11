"""直接运行：python3 demo/IK/run.py。"""

import signal
import threading
import time

import mujoco.viewer

from controller import Controller
from simulation import ArmSimulation


# 直接在这里修改目标位置，单位 m，世界坐标系。
TARGET_POSITION = [0.35, -0.15, 0.85]
STEPS_PER_CONTROL = 5  # 物理步长 0.002 s，控制周期 0.01 s。


def main():
    sim = ArmSimulation()
    sim.set_target(TARGET_POSITION)
    controller = Controller()
    dt = float(sim.model.opt.timestep) * STEPS_PER_CONTROL
    print("红点是末端，绿球是目标；在 controller.py 中实现 Controller.compute()。")
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    threads_before = set(threading.enumerate())
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [0.25, 0, 0.55]
            viewer.cam.distance = 1.9
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -18
            start = time.monotonic()
            while not stop_requested and viewer.is_running():
                q, dq, position, target = sim.state()
                command = controller.compute(q, dq, position, target, dt)
                if command is not None:
                    sim.set_command(command)
                sim.step(STEPS_PER_CONTROL)
                viewer.sync()
                time.sleep(max(0.0, start + float(sim.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        # MuJoCo 3.7 的 Handle.close() 只请求退出，不等待渲染线程清理。
        # 等待 launch_passive 创建的线程，避免 atexit 提前终止 GLFW。
        for thread in threading.enumerate():
            if thread not in threads_before and thread.name.endswith("(_launch_internal)"):
                thread.join()
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    main()
