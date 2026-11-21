"""世界系六自由度笛卡尔弹簧阻尼控制：tau = J.T @ wrench + g(q)。"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.base import vector
from piper.models import OFFICIAL_MODEL


def rotation_matrix(value):
    rotation = np.asarray(value, dtype=float)
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)):
        raise ValueError("目标姿态需要为有效的 3×3 旋转矩阵")
    return rotation


@dataclass
class ImpedanceCommand:
    torque: np.ndarray
    wrench: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    twist: np.ndarray
    error: np.ndarray
    gravity: np.ndarray
    force_torque: np.ndarray

    @property
    def force(self):
        return self.wrench[:3]

    @property
    def velocity(self):
        return self.twist[:3]


class ImpedanceController:
    """六轴输入 q/dq，输出含重力的关节力矩；不依赖真机或仿真接口。

    顺序均为 [x, y, z, rx, ry, rz]，位置 m、旋转向量 rad。
    姿态沿用 PinocchioControl.fk 的工具坐标系，所有误差、速度和 wrench
    均在世界系表达、作用于 ee 原点。target_rotation=None 时只控制 XYZ，
    不施加任何姿态弹簧或角速度阻尼；这不同于将某个 Log 分量置零。
    """

    def __init__(self, kp=(60, 60, 60, 5, 5, 5),
                 kd=(3, 3, 3, 0.5, 0.5, 0.5),
                 max_wrench=(20, 20, 20, 5, 5, 5), control=None):
        self.control = control if control is not None else PinocchioControl(model_path=OFFICIAL_MODEL)
        self.kp = vector(kp, 6).copy()
        self.kd = vector(kd, 6).copy()
        self.max_wrench = vector(max_wrench, 6).copy()
        if np.any(self.kp < 0) or np.any(self.kd < 0) or np.any(self.max_wrench <= 0):
            raise ValueError("刚度和阻尼必须非负，力与力矩限幅必须为正")

    def compute(self, q, dq, target_position, target_rotation=None, target_twist=None):
        """目标速度为世界系 [线速度, 角速度]，静态目标默认零。"""
        q, dq = vector(q, 6), vector(dq, 6)
        target_position = vector(target_position, 3)
        target_twist = np.zeros(6) if target_twist is None else vector(target_twist, 6)
        if target_rotation is not None:
            target_rotation = rotation_matrix(target_rotation)
        pose = self.control.fk(q)
        position, rotation = pose.translation.copy(), pose.rotation.copy()
        jacobian = pin.computeFrameJacobian(
            self.control.model, self.control.data, q, self.control.ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).copy()
        twist = jacobian @ dq
        error = np.zeros(6)
        error[:3] = target_position - position
        if target_rotation is not None:
            error[3:] = pin.log3(target_rotation @ rotation.T)
        wrench = self.kp * error + self.kd * (target_twist - twist)
        if target_rotation is None:
            wrench[3:] = 0
        wrench = np.clip(wrench, -self.max_wrench, self.max_wrench)
        gravity = self.control.gravity(q)
        force_torque = jacobian.T @ wrench
        return ImpedanceCommand(force_torque + gravity, wrench, position, rotation,
                                twist, error, gravity, force_torque)
