"""使用 NumPy、SciPy 的 SO(3) / SE(3) 基础运算。

使用列向量：p_world = R @ p_local + t。
六维向量统一按 [wx, wy, wz, vx, vy, vz] 排列，角度单位为 rad。
"""

import numpy as np
from scipy.linalg import expm


def skew(w):
    """三维向量 -> 反对称矩阵，满足 skew(w) @ p == cross(w, p)。"""
    x, y, z = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def hat(xi):
    """六维向量 [w, v] -> 4×4 李代数矩阵，底行全为零。"""
    xi = np.asarray(xi, dtype=float).reshape(6)
    result = np.zeros((4, 4))
    result[:3, :3] = skew(xi[:3])
    result[:3, 3] = xi[3:]
    return result


def exp_so3(w):
    """旋转向量 -> SO(3)；w 的方向是旋转轴，模长是角度。"""
    return expm(skew(w))


def make_transform(R=None, t=None):
    """由旋转 R 和平移 t 组成 4×4 齐次变换，默认返回单位阵。"""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def exp_se3(xi):
    """六维旋量的指数：SE3 = exp(hat(xi))。

    单位旋量轴 S 转过 q 时使用 exp_se3(S * q)。
    若已经有 se3 矩阵 X，可直接使用 scipy.linalg.expm(X)。
    """
    return expm(hat(xi))


def world_to_twist(w, v, p):
    """由世界系中的角速度、点速度和点位置得到空间旋量 [ws, vs]。

    w：刚体角速度；p：刚体上一点的位置；v：该点的线速度。
    三者必须都用世界坐标系表达。
    ws = w，vs = v - cross(w, p)，故任一点速度为 vs + cross(ws, p)。
    """
    w = np.asarray(w, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    p = np.asarray(p, dtype=float).reshape(3)
    return np.concatenate((w, v - np.cross(w, p)))


def rotation_axis(direction, point):
    """世界系轴方向、轴上一点 -> 单位转动旋量轴 S（六维）。

    自动归一化 direction。固定转轴上的点速度为零，所以 S = [w, -w×p]。
    关节转角 q 对应的变换为 exp_se3(S * q)。
    """
    direction = np.asarray(direction, dtype=float).reshape(3)
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length == 0:
        raise ValueError("旋转轴方向必须为有限非零向量")
    return world_to_twist(direction / length, np.zeros(3), point)


def twist_to_axis(xi):
    """空间旋量 -> (单位轴方向, 轴上距世界原点最近的点, 螺距)。

    螺距为每弧度沿轴的位移。纯平移返回 (移动方向, None, inf)，
    零旋量没有唯一轴，抛出 ValueError。
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    w, v = xi[:3], xi[3:]
    speed = np.linalg.norm(w)
    if speed == 0:
        linear_speed = np.linalg.norm(v)
        if linear_speed == 0:
            raise ValueError("零旋量没有唯一的轴")
        return v / linear_speed, None, np.inf
    direction = w / speed
    point = np.cross(direction, v / speed)
    pitch = float(np.dot(direction, v / speed))
    return direction, point, pitch


def inverse_transform(T):
    """刚体变换的逆，假设输入中的 R 是合法旋转矩阵。"""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)
