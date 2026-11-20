"""独立进程显示实机反馈与工具坐标轴，共享目标球位置。"""

from multiprocessing import get_context
from pathlib import Path
import time


from piper.controllers.pinocchio_controller import EE_TO_LEVEL
from piper.models import ARM_MODEL


def _draw_ee_frame(viewer, data, ee_id):
    import mujoco
    import numpy as np

    origin = data.site_xpos[ee_id]
    rotation = data.site_xmat[ee_id].reshape(3, 3) @ EE_TO_LEVEL
    colors = [[1, 0.1, 0.1, 1], [0.1, 1, 0.1, 1], [0.1, 0.4, 1, 1]]
    viewer.user_scn.ngeom = 0
    for axis, name in enumerate("XYZ"):
        geom = viewer.user_scn.geoms[axis]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), origin, np.eye(3).ravel(), colors[axis],
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.006,
            origin, origin + 0.18 * rotation[:, axis],
        )
        geom.label = f"tool.{name}"
        viewer.user_scn.ngeom += 1


def _run(joint_positions, target_position, stop, model_path):
    import mujoco
    import mujoco.viewer
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    q_ids = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
    ee_id = model.site("ee").id
    target_id = model.body("target").mocapid[0]
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    with joint_positions.get_lock():
        data.qpos[q_ids] = joint_positions[:]
    with target_position.get_lock():
        data.mocap_pos[target_id] = target_position[:]
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
                _draw_ee_frame(viewer, data, ee_id)
            viewer.sync()
            with viewer.lock():
                target = data.mocap_pos[target_id].copy()
            with target_position.get_lock():
                target_position[:] = target
            stop.wait(max(0, 1 / 60 - (time.perf_counter() - start)))


class DigitalTwin:
    def __init__(self, q, target_position, model_path=ARM_MODEL):
        context = get_context("spawn")
        self.joint_positions = context.Array("d", q)
        self.target_position = context.Array("d", target_position)
        self.stop = context.Event()
        self.process = context.Process(
            target=_run,
            args=(self.joint_positions, self.target_position, self.stop,
                  str(Path(model_path).resolve())),
            name="piper-digital-twin",
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

    def read_target_position(self):
        """读取 MuJoCo 目标球位置（世界系，m）。"""
        with self.target_position.get_lock():
            return self.target_position[:]
