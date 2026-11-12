"""PiPER 六轴机械臂与双指夹爪的 MuJoCo 仿真。"""

from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).with_name("arm.xml")
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
# 夹爪参考系 B 相对于模型 ee 的固定旋转：X_B=Z_ee, Y_B=Y_ee, Z_B=-X_ee。
# 控制与显示共用这一约定，水平正放时 X 向前、Y 向左、Z 向上。
EE_TO_BODY_ROTATION = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def vector(values, size):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"需要 {size} 个有限数值")
    return values


class ArmSimulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(n).id for n in ARM_JOINT_NAMES])
        self.q_ids = self.model.jnt_qposadr[self.joint_ids]
        self.v_ids = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([self.model.actuator(n).id for n in ARM_JOINT_NAMES])
        self.ee_id = self.model.site("ee").id
        self.target_id = int(self.model.body("target").mocapid[0])
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self._kinematics_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        """返回六轴 q、dq、末端位置和目标位置的副本（rad、rad/s、m）。"""
        return (
            self.data.qpos[self.q_ids].copy(),
            self.data.qvel[self.v_ids].copy(),
            self.data.site_xpos[self.ee_id].copy(),
            self.data.mocap_pos[self.target_id].copy(),
        )

    def set_target(self, position):
        self.data.mocap_pos[self.target_id] = vector(position, 3)
        mujoco.mj_forward(self.model, self.data)

    def set_command(self, q_target):
        limits = self.model.jnt_range[self.joint_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(
            vector(q_target, 6), limits[:, 0], limits[:, 1]
        )

    def set_gripper(self, width):
        """设置双指总开度，单位 m，范围 0～0.07。"""
        width = float(width)
        if not np.isfinite(width) or not 0 <= width <= 0.07:
            raise ValueError("夹爪开度需要在 0～0.07 m 之间")
        self.data.ctrl[self.model.actuator("gripper").id] = width / 2

    def step(self, count=1):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def draw_ee_frames(self, viewer):
        """画控制器所用夹爪参考系 B：三根箭头是 R_WB 的三个列向量。"""
        origin = self.data.site_xpos[self.ee_id].copy()
        body_rotation = self.data.site_xmat[self.ee_id].reshape(3, 3) @ EE_TO_BODY_ROTATION
        colors = np.array([[1, 0.1, 0.1, 1], [0.1, 1, 0.1, 1], [0.1, 0.4, 1, 1]])
        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            for axis, axis_name in enumerate("XYZ"):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3), origin, np.eye(3).ravel(), colors[axis],
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.006,
                    origin, origin + 0.18 * body_rotation[:, axis],
                )
                geom.label = f"body.{axis_name}"
                scene.ngeom += 1

    def fk(self, q):
        """六轴角度 → 世界系夹爪中心 4×4 位姿，不改变仿真状态。"""
        self._kinematics_data.qpos[:] = self.data.qpos
        self._kinematics_data.qpos[self.q_ids] = vector(q, 6)
        mujoco.mj_forward(self.model, self._kinematics_data)
        transform = np.eye(4)
        transform[:3, :3] = self._kinematics_data.site_xmat[self.ee_id].reshape(3, 3)
        transform[:3, 3] = self._kinematics_data.site_xpos[self.ee_id]
        return transform
