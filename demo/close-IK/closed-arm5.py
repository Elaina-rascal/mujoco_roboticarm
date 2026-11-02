import numpy as np
from pymlg.numpy import SE2


def main():
    l1, l2, l3, l4 = 1.0, 1.0, 1.0, 1.0
    # q[0:2] 为右侧关节，q[2:4] 为左侧关节，单位 rad。
    # 原来的 q5 没有参与计算，这里只保留四个实际使用的关节角。
    # 避开完全伸直的一阶停滞点，不依赖伪逆舍入误差来启动迭代。
    q = np.array([0.7, 0.0, -0.7, 0.0], dtype=float)
    print(f"初始关节角: {q}")

    # pyMLG 的 SE2 旋量顺序为 [w, vx, vy]。
    # 绕点 [px, py] 逆时针转动的单位旋量为 [1, py, -px]。
    S1 = np.array([1.0, 0.0, -0.2])
    S2 = np.array([1.0, 0.0, -(0.2 + l1)])
    M1 = SE2.from_components(np.eye(2), [0.2 + l1 + l2, 0.0])
    print(f"零位右侧末端位姿:\n{M1}")

    S3 = np.array([1.0, 0.0, 0.2])
    S4 = np.array([1.0, 0.0, 0.2 + l3])
    M2 = SE2.from_components(np.eye(2), [-0.2 - l3 - l4, 0.0])

    for _ in range(500):
        # pyMLG 返回 NumPy 矩阵，使用 @ 做矩阵乘法。
        E1 = SE2.Exp(S1 * q[0])
        E3 = SE2.Exp(S3 * q[2])
        T1 = E1 @ SE2.Exp(S2 * q[1]) @ M1
        T2 = E3 @ SE2.Exp(S4 * q[3]) @ M2
        #这里面只需要position误差就够了，旋转误差不需要
        T1_error = T2 @ SE2.inverse(T1)
        T2_error = T1 @ SE2.inverse(T2)
        # Log 返回 (3, 1) 列向量，展平成 (3,) 以更新一维 q。
        v1_error = SE2.Log(T1_error).ravel()
        v2_error = SE2.Log(T2_error).ravel()
        #这里面J只需要考虑position部分就够了，旋转部分不需要
        J1 = np.column_stack([S1, SE2.adjoint(E1) @ S2])
        J2 = np.column_stack([S3, SE2.adjoint(E3) @ S4])
        # 获取两个末端点在世界坐标系下的位置
        p1 = T1[:2, 2]
        p2 = T2[:2, 2]

        # 空间旋量 -> 末端点位置雅可比
        P1 = np.array([
            [-p1[1], 1, 0],
            [ p1[0], 0, 1]
        ])

        P2 = np.array([
            [-p2[1], 1, 0],
            [ p2[0], 0, 1]
        ])

        J1_p = P1 @ J1
        J2_p = P2 @ J2
        # q[:2] += np.linalg.pinv(J1_p) @ v1_error[1::]
        # q[2:] += np.linalg.pinv(J2_p) @ v2_error[1::]
        q[:2]+= np.linalg.pinv(J1_p) @ (p2-p1)
        q[2:]+= np.linalg.pinv(J2_p) @ (p1-p2)
        v1_error[0] = 0
        v2_error[0] = 0
        # q[:2]+= np.linalg.pinv(J1)@v1_error
        # q[2:]+= np.linalg.pinv(J2)@v2_error
    T1 = SE2.Exp(S1 * q[0]) @ SE2.Exp(S2 * q[1]) @ M1
    T2 = SE2.Exp(S3 * q[2]) @ SE2.Exp(S4 * q[3]) @ M2
    #保留三位小数

    print(f"右侧末端位姿:\n{np.round(T1,3)}\n左侧末端位姿:\n{np.round(T2,3)}")
    print(f"当前关节角: {q}")
    print(f"两末端位置距离: {np.linalg.norm(T1[:2, 2] - T2[:2, 2]):.6f}")


if __name__ == '__main__':
    main()
