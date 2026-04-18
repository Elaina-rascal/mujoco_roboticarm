from __future__ import annotations

from typing import Dict

from geometry_msgs.msg import TransformStamped
import mujoco
import numpy as np
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class MujocoTfPublisher:
    """发布 MuJoCo 机体坐标系到 ROS2 TF。"""

    def __init__(self, node: Node, world_frame: str = "world") -> None:
        self._node = node
        self._world_frame = world_frame
        self._broadcaster = TransformBroadcaster(node)

    def _body_frame_name(self, model: mujoco.MjModel, body_id: int) -> str:  # type: ignore
        """将 MuJoCo body id 转成稳定的 TF frame 名称。"""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        if body_name is None or body_name == "":
            return f"body_{body_id}"
        return body_name

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData, stamp) -> None:  # type: ignore
        """按当前仿真状态发布所有 body 的世界系变换。"""
        transforms = []

        for body_id in range(1, model.nbody):
            frame_name = self._body_frame_name(model, body_id)

            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self._world_frame
            transform.child_frame_id = frame_name

            # MuJoCo 的 xpos/xmat 为 body 在世界坐标系下的位姿
            pos = data.xpos[body_id]
            rot_mat = data.xmat[body_id]

            transform.transform.translation.x = float(pos[0])
            transform.transform.translation.y = float(pos[1])
            transform.transform.translation.z = float(pos[2])

            # 使用 MuJoCo 提供的工具函数将 3x3 旋转矩阵转为四元数 (w, x, y, z)
            quat_wxyz = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat_wxyz, rot_mat)  # type: ignore

            transform.transform.rotation.w = float(quat_wxyz[0])
            transform.transform.rotation.x = float(quat_wxyz[1])
            transform.transform.rotation.y = float(quat_wxyz[2])
            transform.transform.rotation.z = float(quat_wxyz[3])

            transforms.append(transform)

        if transforms:
            self._broadcaster.sendTransform(transforms)
