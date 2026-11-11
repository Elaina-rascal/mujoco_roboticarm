"""容器内运行：python3 piper/real/run.py。"""

from copy import deepcopy
import signal
import time

import numpy as np
import pinocchio as pin
from piper_sdk import C_PiperInterface_V2

if __package__:
    from .fk import ForwardKinematics
else:
    from fk import ForwardKinematics


CAN_NAME = "can0"
READ_PERIOD = 0.01  # s，100 Hz 读取。
PRINT_PERIOD = 0.1  # s，10 Hz 显示。
FEEDBACK_TIMEOUT = 2.0  # s，启动未收到数据或反馈停止时退出。


def read_joint_positions(piper):
    """SDK 反馈单位 0.001°，按 joint1～joint6 返回弧度和反馈时间戳。"""
    message = deepcopy(piper.GetArmJointMsgs())
    if message.time_stamp <= 0:
        return None
    joint_state = message.joint_state
    angles = np.array(
        [getattr(joint_state, f"joint_{i}") for i in range(1, 7)], dtype=float
    )
    return np.deg2rad(angles * 0.001), message.time_stamp


def main():
    fk = ForwardKinematics()
    # can0 已在主机配置为 1 Mbps；容器没有 ip 命令，跳过 SDK 的外部命令检查。
    piper = C_PiperInterface_V2(can_name=CAN_NAME, judge_flag=False)
    last_stamp = None
    last_received = time.monotonic()
    next_print = last_received
    np.set_printoptions(precision=6, suppress=True)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        # 仅接收反馈；跳过 SDK 查询初始化，不使能或发送运动指令。
        piper.ConnectPort(piper_init=False)
        print(f"读取 {CAN_NAME} 的六轴关节反馈，Ctrl+C 退出。")
        while not stop_requested:
            start = time.monotonic()
            state = read_joint_positions(piper)
            if state is not None:
                q, stamp = state
                if stamp != last_stamp:
                    last_stamp = stamp
                    last_received = start
                    transform = fk.compute(q).np
                    if start >= next_print:
                        rpy = pin.rpy.matrixToRpy(transform[:3, :3])
                        print(f"\n反馈时间戳: {stamp:.6f}")
                        print("关节角(rad):", q)
                        print("关节角(deg):", np.rad2deg(q))
                        print("末端位置(m):", transform[:3, 3])
                        print("末端 RPY(rad):", rpy)
                        print("末端位姿 T_base_ee:\n", transform)
                        next_print = start + PRINT_PERIOD
            if start - last_received > FEEDBACK_TIMEOUT:
                raise TimeoutError(f"{CAN_NAME} 超过 {FEEDBACK_TIMEOUT} s 未收到新的关节反馈")
            time.sleep(max(0.0, READ_PERIOD - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            piper.DisconnectPort()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    main()
