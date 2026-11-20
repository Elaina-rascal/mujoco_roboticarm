"""重力辨识轨迹演示：平滑移动到不同姿态，停留后再移动，不采集数据。"""

import time

import numpy as np

from piper.interface.mujoco import MujocoInterface


MOVE_TIME = 3.0  # s，每两个姿态之间的移动时间。
HOLD_TIME = 2.0  # s，到达后静止，供观察重力辨识所需的静态姿态。


def make_waypoints(sim):
    """从 home 出发，经过六个随机姿态，最后回到 home。"""
    home = sim.read_joint_positions()
    rng = np.random.default_rng(42)
    amplitude = np.array([0.7, 0.5, 0.6, 0.5, 0.5, 0.6])  # 六轴最大偏移 rad。
    poses = home + rng.uniform(-1, 1, (6, 6)) * amplitude
    limits = sim.model.jnt_range[sim.joint_ids]
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


def main():
    import mujoco.viewer

    sim = MujocoInterface(mode="position")
    sim.set_gripper(0.04)
    waypoints = make_waypoints(sim)
    print("重力辨识轨迹：移动 3 秒、停留 2 秒，回到 home 后循环。")
    print("仅演示轨迹，不采集数据；关闭窗口或 Ctrl+C 退出。")

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [0.2, 0, 0.3]
            viewer.cam.distance = 1.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            start_time = time.monotonic()
            while viewer.is_running():
                q, _, _ = trajectory(float(sim.data.time), waypoints)
                sim.send_joint_positions(q)
                sim.set_target(sim.fk(q)[:3, 3])
                sim.step(5)
                sim.draw_ee_frames(viewer)
                viewer.sync()
                time.sleep(max(0, start_time + float(sim.data.time) - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
