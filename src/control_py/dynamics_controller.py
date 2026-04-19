from typing import List

import numpy as np
import pinocchio as pin


class PinocchioDynamicsController:
    """基于 Pinocchio 的关节空间动力学控制器。

    控制律：
    tau = rnea(q, qd, qdd_ref) + Kp * (q_ref - q) + Kd * (qd_ref - qd)
    """

    def __init__(
        self,
        model: pin.Model,
        kp: float = 80.0,
        kd: float = 12.0,
    ) -> None:
        self.model = model
        self.data = self.model.createData()

        # 直接按 Pinocchio 模型中的关节顺序建立索引表，不再做 name/id 映射。
        self.q_indices: List[int] = []
        self.v_indices: List[int] = []
        joints = self.model.joints if self.model.joints is not None else []
        for joint_id in range(1, len(joints)):
            joint = joints[joint_id]
            if joint is None:
                continue
            self.q_indices.append(int(joint.idx_q))
            self.v_indices.append(int(joint.idx_v))

        # 统一增益（简单稳定，后续可改成各关节独立增益）
        self.kp = float(kp)
        self.kd = float(kd)

    def compute_torque(
        self,
        joint_positions: List[float],
        joint_velocities: List[float],
        q_ref: np.ndarray,
    ) -> List[float]:
        """根据当前状态与参考位姿计算关节力矩，并按输入顺序返回。"""
        q = pin.neutral(self.model)
        qd = np.zeros(self.model.nv) #type: ignore
        qd_ref = np.zeros(self.model.nv) #type: ignore
        qdd_ref = np.zeros(self.model.nv) #type: ignore

        # 直接按关节顺序填充状态向量。
        count = min(len(joint_positions), len(self.q_indices), len(joint_velocities))
        for i in range(count):
            idx_q = self.q_indices[i]
            idx_v = self.v_indices[i]
            q[idx_q] = float(joint_positions[i])
            qd[idx_v] = float(joint_velocities[i])

        # 前馈动力学项（重力/科氏/离心等，qdd_ref 此处取 0）
        tau_ff = pin.rnea(self.model, self.data, q, qd, qdd_ref)

        # 反馈项：在速度空间 idx_v 计算，保证与力矩维度一致
        tau = np.array(tau_ff, dtype=float)
        count = min(len(q_ref), len(self.q_indices))
        for i in range(count):
            idx_q = self.q_indices[i]
            idx_v = self.v_indices[i]
            pos_err = float(q_ref[idx_q] - q[idx_q])
            vel_err = float(qd_ref[idx_v] - qd[idx_v])
            tau[idx_v] += self.kp * pos_err + self.kd * vel_err

        # 按输入顺序回写到 effort。
        out_effort: List[float] = []
        for i in range(min(len(self.v_indices), len(joint_positions))):
            out_effort.append(float(tau[self.v_indices[i]]))

        return out_effort
