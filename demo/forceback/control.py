"""双轴摇杆辅助函数，q=[第一轴角度, 第二轴角度]，单位 rad。

动力学：tau = M(q) @ ddq + c(q,dq) + g(q) + friction(dq)。
参数对应 joystick.xml；阻抗控制用角度误差和两轴力臂计算末端力，不补偿科氏力。
不含机械限位和零速静摩擦求解。
"""

import numpy as np

L = 0.25  # 轴心到握持点的距离，m
H = 0.09  # 轴心距地面的高度，m

# 由当前电机、连接件和摇杆的质量/惯量合并得到，单位 kg*m²。
I1 = 0.000414520104167  # 第一轴连接件和第二电机的绕轴惯量
IXX = 0.013348366666667  # 第二轴输出组件在轴心处的惯量
IYY = 0.013327886666667
IZZ = 0.0000778133333333
IYZ = 0.00003584


def fk(q):
    """正运动学，返回握持点世界坐标 [x,y,z]，单位 m。"""
    s1, s2 = np.sin(q)
    c1, c2 = np.cos(q)
    return np.array([L*s2, -L*s1*c2, H + L*c1*c2])


def jacobian(q):
    """3×2 位置雅可比，末端世界系线速度 v = J(q) @ dq。"""
    s1, s2 = np.sin(q)
    c1, c2 = np.cos(q)
    return L * np.array([[0, c2], [-c1*c2, s1*s2], [-s1*c2, -c1*s2]])


class Controller:
    """末端弹簧阻尼+重力补偿；刚度 N/m、阻尼 N*s/m，可分别设置两轴参数。"""

    def __init__(self, q_target=(0.0, 0.0), stiffness=40.0, damping=2.0,
                 sine_amplitude=0.0, sine_frequency=50):
        self.q_target = np.array(q_target, dtype=float, copy=True)
        self.stiffness = np.array(stiffness, dtype=float, copy=True)
        self.damping = np.array(damping, dtype=float, copy=True)
        self.tip_force = np.zeros(2)
        self.sine_amplitude = sine_amplitude  # 局部 x 方向附加力幅值，N
        self.sine_frequency = sine_frequency  # Hz

    def set_tip_force(self, force):
        """设置附加末端力 [Fx,Fy]（N），方向与弹簧力两分量一致，随杆转动。

        持续生效；传入 [0,0] 清除。它与弹簧阻尼力相加后由电机主动输出。
        """
        force = np.asarray(force, dtype=float)
        if force.shape != (2,) or not np.all(np.isfinite(force)):
            raise ValueError("force 必须为两个有限数值，单位 N")
        self.tip_force = force.copy()

    def axis_levers(self, q):
        """两轴有效力臂（m）。"""
        return np.array([L*np.cos(q[1]), L])

    def added_force(self, time):
        """附加末端力：常值输入加局部 x 方向正弦力，time 单位 s。"""
        sine_force = self.sine_amplitude*np.sin(2*np.pi*self.sine_frequency*time)
        return self.tip_force + np.array([0.0, sine_force])

    def impedance_force(self, q, dq):
        """返回两轴末端切向力 [F1,F2]（N），这里只算力，不换算电机力矩。

        输入 q 为关节角（rad），dq 为角速度（rad/s）。
        弹簧力指向目标，阻尼力反向于速度；所有乘法均为两轴逐元素相乘。
        """
        joint_angle = np.asarray(q, dtype=float)
        joint_velocity = np.asarray(dq, dtype=float)

        # 1. 当前有效力臂（m）：轴一随第二轴倾角变化，轴二固定为 L。
        lever_arm = self.axis_levers(joint_angle)

        # 2. 目标角度 - 当前角度（rad），正负号表示恢复方向。
        angle_error = self.q_target - joint_angle

        # 3. 角度误差转为末端位移的局部近似（m），角速度转为线速度（m/s）。
        tip_displacement = lever_arm * angle_error
        tip_velocity = lever_arm * joint_velocity

        # 4. 弹簧力 K*位移；阻尼力 -D*速度，两者单位均为 N。
        spring_force = self.stiffness * tip_displacement
        damping_force = -self.damping * tip_velocity
        tip_force = spring_force + damping_force

        print(f"弹簧力(N)={spring_force} 阻尼力(N)={damping_force}")
        return tip_force

    def update(self, q, dq, time=0.0):
        """返回 r*(F_impedance+F_added)+g(q)，单位 N*m。

        q 为 rad，dq 为 rad/s；刚度 K 为 N/m，阻尼 D 为 N*s/m。
        两轴逐元素计算，不做世界坐标投影或科氏力补偿。
        最终由仿真 set_torque() 限幅至 ±17 N*m。
        """
        force = self.impedance_force(q, dq) + self.added_force(time)
        print(f"柯氏力为{coriolis_vector(q,dq)}")
        return self.axis_levers(q)*force + gravity_vector(q)


def mass_matrix(q):
    """2×2 关节惯性矩阵。"""
    s, c = np.sin(q[1]), np.cos(q[1])
    return np.array([[I1 + IXX*c*c + IZZ*s*s, IYZ*s],
                     [IYZ*s, IYY]])


def coriolis_vector(q, dq):
    """科氏力/离心力合并项 c=C@dq，直接返回两个关节的力矩。"""
    s, c = np.sin(q[1]), np.cos(q[1])
    a = 2*(IZZ - IXX)*s*c  # dM11/dq2
    b = IYZ*c              # dM12/dq2
    return np.array([a*dq[0]*dq[1] + b*dq[1]**2, -0.5*a*dq[0]**2])


def gravity_vector(q):
    """重力补偿力矩 g，实际重力施加的力矩为 -g。"""
    s1, s2 = np.sin(q)
    c1, c2 = np.cos(q)
    # 0.00232、0.07849 为合并后的质量一阶矩，单位 kg*m。
    return 9.81 * np.array([0.00232*c1 - 0.07849*s1*c2, -0.07849*c1*s2])


def friction_torque(dq):
    """粘性阻尼 + 滑动摩擦；零速时取零，不求解静摩擦。"""
    dq = np.asarray(dq)
    return 0.03*dq + 0.01*np.sign(dq)
