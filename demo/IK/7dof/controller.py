"""在这里实现你自己的位置 IK / 控制算法。"""

from pathlib import Path

import pinocchio as pin
import numpy as np

class Controller:
    def __init__(self):
        # 和 MuJoCo 读取同一份模型，固定基座、七个有界转动关节。
        # q = [joint1, ..., joint7]，每个元素直接是 rad，不做重排或转换。
        self.model = pin.buildModelFromMJCF(str(Path(__file__).with_name("arm.xml")))
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId("ee")
        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError("模型应为七个单自由度转动关节")
        if not self.model.existFrame("ee"):
            raise ValueError("模型中缺少末端 ee")

    def compute(self, q, dq, ee_position, target_position, dt):
        """返回七个关节的目标角，或返回 None 保持上一条命令。

        q: (7,) 当前关节角，rad，直接传入 Pinocchio 即可。
        dq: (7,) 当前关节角速度，rad/s。
        ee_position: (3,) 当前末端位置，世界系，m。
        target_position: (3,) 目标位置，世界系，m。
        dt: 控制周期，s。

        数组均为状态副本，可以自由操作。这里只约束末端位置，不约束朝向。
        MuJoCo 内置位置执行器负责跟踪返回的目标角。
        """
        # TODO：把末端位置误差转换成七个关节的目标角。

        #设置迭代误差
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
            print(f"迭代误差: {np.linalg.norm(v_error):.3f}, 当前关节角: {q_target}")
        #加上零空间
        J= pin.computeFrameJacobian(self.model, self.data, q=q_target, frame_id=self.ee_id, reference_frame=pin.ReferenceFrame.WORLD)
        J_pinv = np.linalg.pinv(J)
        #计算零空间投影矩阵
        N = np.eye(7) - J_pinv @ J
        zero_q=np.ones(7)*0.1
        q_target += N @ zero_q
        return q_target
