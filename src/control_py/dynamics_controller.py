from typing import Dict, List

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
        joint_name_to_id: Dict[str, int],
        kp: float = 80.0,
        kd: float = 12.0,
    ) -> None:
        self.model = model
        self.data = self.model.createData()
        self.joint_name_to_id = joint_name_to_id

        # 统一增益（简单稳定，后续可改成各关节独立增益）
        self.kp = float(kp)
        self.kd = float(kd)

    def compute_torque(
        self,
        joint_names: List[str],
        joint_positions: List[float],
        joint_velocities: List[float],
        q_ref: np.ndarray,
    ) -> List[float]:
        """根据当前状态与参考位姿计算关节力矩，并按 JointState 顺序返回。"""
        q = pin.neutral(self.model)
        qd = np.zeros(self.model.nv) #type: ignore
        qd_ref = np.zeros(self.model.nv) #type: ignore
        qdd_ref = np.zeros(self.model.nv) #type: ignore

        # 把 ROS JointState 映射到 Pinocchio 配置向量
        for i, name in enumerate(joint_names):
            if name not in self.joint_name_to_id:
                continue
            joint_id = self.joint_name_to_id[name]
            idx_q = self.model.joints[joint_id].idx_q  # type: ignore
            idx_v = self.model.joints[joint_id].idx_v  # type: ignore
            q[idx_q] = float(joint_positions[i])
            qd[idx_v] = float(joint_velocities[i])

        # 前馈动力学项（重力/科氏/离心等，qdd_ref 此处取 0）
        tau_ff = pin.rnea(self.model, self.data, q, qd, qdd_ref)

        # 反馈项：在速度空间 idx_v 计算，保证与力矩维度一致
        tau = np.array(tau_ff, dtype=float)
        for i, name in enumerate(joint_names):
            if name not in self.joint_name_to_id:
                continue
            joint_id = self.joint_name_to_id[name]
            idx_q = self.model.joints[joint_id].idx_q  # type: ignore
            idx_v = self.model.joints[joint_id].idx_v  # type: ignore
            pos_err = float(q_ref[idx_q] - q[idx_q])
            vel_err = float(qd_ref[idx_v] - qd[idx_v])
            tau[idx_v] += self.kp * pos_err + self.kd * vel_err

        # 按输入关节名顺序回写到 effort
        out_effort: List[float] = []
        for name in joint_names:
            if name in self.joint_name_to_id:
                joint_id = self.joint_name_to_id[name]
                idx_v = self.model.joints[joint_id].idx_v  # type: ignore
                out_effort.append(float(tau[idx_v]))
            else:
                out_effort.append(0.0)

        return out_effort
