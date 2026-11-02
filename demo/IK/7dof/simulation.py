from pathlib import Path

import mujoco
import numpy as np


class ArmSimulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("arm.xml")))
        self.data = mujoco.MjData(self.model)
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_id = int(self.model.body_mocapid[target_id])
        # 使用独立缓存做 FK 查询，不影响正在运行的仿真。
        self._kinematics_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.site_xpos[self.ee_id].copy(),
            self.data.mocap_pos[self.target_id].copy(),
        )

    def set_target(self, position):
        self.data.mocap_pos[self.target_id] = self._vector(position, 3)
        mujoco.mj_forward(self.model, self.data)

    def set_command(self, q_target):
        q_target = self._vector(q_target, 7)
        self.data.ctrl[:] = np.clip(
            q_target, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        )

    def step(self, count=1):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def fk(self, q):
        """MuJoCo 参考 FK：输入七个角，返回世界系末端 4×4 位姿。

        可以用它核对自己写的 FK；不包含 IK 求解。
        """
        self._kinematics_data.qpos[:] = self._vector(q, 7)
        mujoco.mj_forward(self.model, self._kinematics_data)
        T = np.eye(4)
        T[:3, :3] = self._kinematics_data.site_xmat[self.ee_id].reshape(3, 3)
        T[:3, 3] = self._kinematics_data.site_xpos[self.ee_id]
        return T

    @staticmethod
    def _vector(values, size):
        values = np.asarray(values, dtype=float)
        if values.shape != (size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"需要 {size} 个有限数值")
        return values
