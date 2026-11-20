"""真机用五次多项式轨迹，与仿真各自独立。"""

import xml.etree.ElementTree as ET

import numpy as np

from piper.models import OFFICIAL_MODEL


MOVE_TIME = 3.0
HOLD_TIME = 2.0


def make_waypoints():
    """沿用仿真 demo 的模型 home、随机种子和幅度，生成相同的八个姿态。"""
    root = ET.parse(OFFICIAL_MODEL).getroot()
    home = np.fromstring(root.find("keyframe/key[@name='home']").get("qpos"), sep=" ")[:6]
    limits = np.array([
        np.fromstring(root.find(f".//joint[@name='joint{i}']").get("range"), sep=" ")
        for i in range(1, 7)
    ])
    rng = np.random.default_rng(42)
    amplitude = np.array([0.7, 0.5, 0.6, 0.5, 0.5, 0.6])
    poses = home + rng.uniform(-1, 1, (6, 6)) * amplitude
    poses = np.clip(poses, limits[:, 0] + 0.05, limits[:, 1] - 0.05)
    return np.vstack([home, poses, home])


def trajectory(t, waypoints):
    """返回六轴目标 q、dq、ddq；移动与停留交替，整条轨迹循环。"""
    segment_time = MOVE_TIME + HOLD_TIME
    t = t % ((len(waypoints) - 1) * segment_time)
    index = int(t // segment_time)
    elapsed = t - index * segment_time
    start, end = waypoints[index:index + 2]

    if elapsed >= MOVE_TIME:
        return end.copy(), np.zeros(6), np.zeros(6)

    s = elapsed / MOVE_TIME
    # 五次多项式：每段起止速度和加速度均为零。
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    dh = (30 * s**2 - 60 * s**3 + 30 * s**4) / MOVE_TIME
    ddh = (60 * s - 180 * s**2 + 120 * s**3) / MOVE_TIME**2
    delta = end - start
    return start + delta * h, delta * dh, delta * ddh
