from pathlib import Path

import mujoco
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "universal_robots_ur5e"


class MujocoArm:
    """通过函数调用设置目标、推进仿真和读取状态。"""

    def __init__(self, mode="position"):
        if mode not in ("position", "torque"):
            raise ValueError("mode 必须为 position 或 torque")
        self.mode = mode
        scene = "scene_torque.xml" if mode == "torque" else "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / scene))
        self.data = mujoco.MjData(self.model)
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home)
        mujoco.mj_forward(self.model, self.data)

        self.joint_ids = np.flatnonzero(
            self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
        )
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
            for j in self.joint_ids
        ]
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]
        # 执行器名称与关节名称不同，按 transmission 中绑定的关节 ID 映射。
        self.actuator_ids = []
        for joint_id in self.joint_ids:
            matches = np.flatnonzero(
                (self.model.actuator_trnid[:, 0] == joint_id)
                & (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            )
            if len(matches) != 1:
                raise ValueError("每个机械臂关节需要对应一个执行器")
            self.actuator_ids.append(int(matches[0]))

    def set_control(self, values):
        """按 joint_names 顺序输入位置（rad）或力矩（N·m）。"""
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.joint_names),) or not np.all(np.isfinite(values)):
            raise ValueError(f"需要 {len(self.joint_names)} 个有限数值")
        bounds = self.model.actuator_ctrlrange[self.actuator_ids]
        limited = self.model.actuator_ctrllimited[self.actuator_ids].astype(bool)
        self.data.ctrl[self.actuator_ids] = np.where(
            limited, np.clip(values, bounds[:, 0], bounds[:, 1]), values
        )

    def step(self):
        mujoco.mj_step(self.model, self.data)
        # 更新到积分后的位姿，供 Python 调用方读取 body 坐标。
        mujoco.mj_forward(self.model, self.data)

    def joint_state(self):
        return {
            "time": float(self.data.time),
            "name": list(self.joint_names),
            "position": self.data.qpos[self.qpos_indices].tolist(),
            "velocity": self.data.qvel[self.dof_indices].tolist(),
        }

    def body_transforms(self):
        """返回各 body 相对世界系的位置和四元数（w, x, y, z）。"""
        return {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}": {
                "position": self.data.xpos[i].tolist(),
                "quaternion_wxyz": self.data.xquat[i].tolist(),
            }
            for i in range(1, self.model.nbody)
        }
