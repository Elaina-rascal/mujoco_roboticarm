"""真机和仿真的共同接口：六轴角度 rad，输出端力矩 N·m。"""

from abc import ABC, abstractmethod

import numpy as np


def vector(values, size):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"需要 {size} 个有限数值")
    return values


class ArmInterface(ABC):
    """关节顺序统一为 joint1～joint6，生命周期由具体实现管理。"""

    @abstractmethod
    def send_joint_positions(self, positions):
        """发送六轴目标角度（rad）。"""

    @abstractmethod
    def send_joint_torques(self, torques):
        """发送六轴前馈力矩（N·m）。"""

    @abstractmethod
    def read_joint_positions(self):
        """读取六轴当前角度（rad）。"""

    @abstractmethod
    def read_joint_torques(self):
        """读取六轴当前力矩（N·m）。"""
