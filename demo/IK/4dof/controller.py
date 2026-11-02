"""四自由度位置控制练习：核心算法由你实现。"""

from pathlib import Path

import pinocchio as pin 
import numpy as np


class Controller:
    def __init__(self):
        self.model = pin.buildModelFromMJCF(str(Path(__file__).with_name("arm.xml")))
        self.data = self.model.createData()
        if self.model.nq != 4 or self.model.nv != 4:
            raise ValueError("需要四个有界转动关节")
        if not self.model.existFrame("ee"):
            raise ValueError("模型中缺少末端 ee")
        self.ee_id = self.model.getFrameId("ee")

    def compute(self, q, dq, ee_position, target_position, dt):
        """返回四个目标关节角（rad），或 None 保持上一条命令。

        q / dq: (4,)，joint1 到 joint4，直接与 MuJoCo、Pinocchio 共用。
        ee_position / target_position: (3,)，世界系位置，m。
        dt: 控制周期，s。
        """
        # TODO：在这里实现自己的 IK / 控制算法。
        error_threshold = 1e-3
        T_target=pin.SE3(np.eye(3), target_position)
        q_target=q.copy()
        for _ in range(100):
            pin.forwardKinematics(self.model, self.data, q_target)
            pin.updateFramePlacements(self.model, self.data)
            # 计算当前末端位姿
            current_M = self.data.oMf[self.ee_id]  #type: ignore
            #算当前旋量误差
            v_error:np.ndarray = pin.log6(T_target*current_M.inverse()).vector
            if np.linalg.norm(v_error) < error_threshold:
                break
            #计算雅可比矩阵
            J = pin.computeFrameJacobian(self.model, self.data, q=q, frame_id=self.ee_id, reference_frame=pin.ReferenceFrame.WORLD)
            #直接用伪逆求解关节速度
            dq = np.linalg.pinv(J) @ v_error
            #更新关节角
            q_target += dq*0.1 
       
        return q_target
