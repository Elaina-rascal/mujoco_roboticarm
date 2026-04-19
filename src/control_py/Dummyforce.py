from typing import List,Dict
import numpy as np
import pinocchio as pin
class DummyForce:
    '''将末端位置转化成末端力，并通过雅可比转化成关节力矩的控制器。'''
    def __init__(
        self,
        model: pin.Model,
        kp: float = 80.0,
        kd: float = 12.0,
    ) -> None:
        self.model = model
        self.data = self.model.createData()
        self.target_M = pin.SE3(np.eye(3), np.array([0.35, 0.15, 0.5]))
        self.ee_frame_name = "attachment_site"
        self.ee_id=self.model.getFrameId(self.ee_frame_name)
        # 直接按 Pinocchio 模型中的关节顺序建立索引表，不再做 name/id 映射。
        # 电机环而不是末端的pd控制
        self.kp = float(kp)
        self.kd = float(kd)
        #末端位置到力的pd控制
        self.end_Kp = np.diag([200, 200, 200, 40, 40, 40])
        self.end_dgain=0.1 #阻尼增益，防止末端振荡
    def compute_torque(
        self,
        joint_positions: List[float],
        joint_velocities: List[float],
    ) -> List[float]:
        '''
        根据当前状态与参考位姿计算关节力矩，并按输入顺序返回。
        '''
        q = pin.neutral(self.model) #当前关节位置
        qd = np.zeros(self.model.nv) #type: ignore #当前关节速度
        qd_ref = np.zeros(self.model.nv) #type: ignore #参考关节速度，这里取0
        qdd_ref = np.zeros(self.model.nv) #type: ignore #参考关节加速度，这里取0

        # 直接按关节顺序填充状态向量。
        count = min(len(joint_positions),len(joint_velocities))
        for i in range(count):
            q[i] = float(joint_positions[i])
            qd[i] = float(joint_velocities[i])
        # 计算末端位置误差
        # 更新运动学数据
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        # 计算当前末端位姿
        current_M = self.data.oMf[self.ee_id]  #type: ignore

        # 计算误差运动 (SE3 空间中的 log 映射)
        err_motion = pin.log6(current_M.inverse() * self.target_M)
        err = err_motion.vector
        # 末端力计算（简单的 PD 控制）
        end_effort = self.end_Kp @ err
        # 雅可比转化为关节力矩
        J  = pin.getFrameJacobian(
                self.model,
                self.data,
                self.ee_id,
                pin.ReferenceFrame.LOCAL,
            )
        tau = J.T @ end_effort
            # 前馈动力学项（重力/科氏/离心等，qdd_ref 此处取 0）
        tau_ff = pin.rnea(self.model, self.data, q, qd, qdd_ref)
        # PD 控制项
        torque_sum:np.ndarray=tau_ff+tau
        #每个电机的pd控制
        for i in range(count):
            pos_err = float(q[i] - q[i]) #这里位置误差为0，因为没有参考位置，只有末端位置
            vel_err = float(qd_ref[i]-qd[i])
            torque_sum[i] += self.kp * pos_err + self.kd * vel_err
        return torque_sum.tolist()
    def set_target_point(self, x: float, y: float, z: float) -> None:
        """设置末端目标点（保持单位旋转，仅更新平移）。"""
        self.target_M = pin.SE3(np.eye(3), np.array([x, y, z], dtype=float))
