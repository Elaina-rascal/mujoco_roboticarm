"""真机重力辨识轨迹：在停留段采集实测角度和力矩，退出时保存 npz。"""

from pathlib import Path
import time

import numpy as np

from piper.interface.piper_new import PiperInterface
from piper.real.estimate.trajectory import HOLD_TIME, MOVE_TIME, make_waypoints, trajectory


CONTROL_PERIOD = 1 / 200  # s


def move_to_start(arm, target):
    """从真实当前位置平滑移到首帧，到位后才开始循环轨迹。"""
    waypoints = np.vstack([arm.read_joint_positions(), target])
    start_time = time.monotonic()
    while time.monotonic() - start_time < 30.0:
        elapsed = time.monotonic() - start_time
        q, _, _ = trajectory(min(elapsed, MOVE_TIME), waypoints)
        arm.send_joint_positions(q)
        if elapsed >= MOVE_TIME and np.max(np.abs(arm.read_joint_positions() - target)) < 0.02:
            return
        time.sleep(CONTROL_PERIOD)
    raise TimeoutError("30 秒内未到达首个关键帧，停止执行轨迹")


def main():
    waypoints = make_waypoints()
    samples = []
    output = Path(__file__).parent / "data" / "gravity.npz"
    arm = PiperInterface(can_name="can_piper", mode="position")
    try:
        deadline = time.monotonic() + 5.0
        while not arm.has_feedback():
            if time.monotonic() >= deadline:
                raise TimeoutError("等待关节反馈超时")
            time.sleep(CONTROL_PERIOD)

        deadline = time.monotonic() + 10.0
        while not arm.enable():
            if time.monotonic() >= deadline:
                raise TimeoutError("真机使能超时")
            time.sleep(CONTROL_PERIOD)

        # 在运动前确认当前固件支持力矩反馈，避免只运行轨迹而没有可用测量。
        arm.read_joint_torques()
        print("平滑移动到首个关键帧……")
        move_to_start(arm, waypoints[0])
        print("开始循环：移动 3 秒、停留 2 秒，停留后半段采集角度和力矩。")
        print(f"Ctrl+C 退出并保持当前位置，数据保存到：{output}")
        start_time = time.monotonic()
        while True:
            tick = time.monotonic()
            elapsed = tick - start_time
            q, _, _ = trajectory(elapsed, waypoints)
            arm.send_joint_positions(q)
            # 只在每次停留的最后 1 秒采样，不把运动过程作为重力辨识数据。
            if elapsed % (MOVE_TIME + HOLD_TIME) >= MOVE_TIME + HOLD_TIME / 2:
                read_start = time.monotonic()
                measured_q = arm.read_joint_positions()
                measured_tau = arm.read_joint_torques()
                sample_time = (read_start + time.monotonic()) / 2 - start_time
                samples.append((sample_time, measured_q.copy(), measured_tau.copy()))
            time.sleep(max(0, CONTROL_PERIOD - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            arm.close()
        finally:
            # 即使退出保持失败，也先保留已经采到的数据。
            if samples:
                t, q, tau = zip(*samples)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output, t=np.array(t), q=np.array(q), tau=np.array(tau),
                    joint_names=np.array([f"joint{i}" for i in range(1, 7)]),
                    firmware=arm.firmware_version,
                    torque_source="motor_current_estimate",
                )
                print(f"已保存 {len(samples)} 组实测角度/力矩：{output}")
            else:
                print("未采到停留段数据，没有生成 npz。")


if __name__ == "__main__":
    main()
