"""独立进程显示真实六轴关节角，使用共享数组传递最新反馈。"""

from multiprocessing import get_context
from pathlib import Path
import time


def _run(joint_positions, stop):
    import mujoco
    import mujoco.viewer
    import numpy as np

    model_path = Path(__file__).resolve().parents[1] / "sim" / "arm.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    q_ids = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
    # 数字孪生只显示反馈姿态，隐藏仿真中的可拖动目标球。
    model.geom_rgba[model.body("target").geomadr[0], 3] = 0
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    with joint_positions.get_lock():
        data.qpos[q_ids] = joint_positions[:]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.2, 0, 0.3]
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        while viewer.is_running() and not stop.is_set():
            start = time.perf_counter()
            with joint_positions.get_lock():
                q = np.array(joint_positions[:])
            with viewer.lock():
                data.qpos[q_ids] = q
                mujoco.mj_forward(model, data)
            viewer.sync()
            stop.wait(max(0, 1 / 60 - (time.perf_counter() - start)))


class DigitalTwin:
    def __init__(self):
        context = get_context("spawn")
        self.joint_positions = context.Array("d", 6)
        self.stop = context.Event()
        self.process = context.Process(
            target=_run, args=(self.joint_positions, self.stop), name="piper-digital-twin"
        )
        self.process.start()

    def update(self, q):
        """同步 joint1～joint6 的实际反馈角度（rad）。"""
        with self.joint_positions.get_lock():
            self.joint_positions[:] = q

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
