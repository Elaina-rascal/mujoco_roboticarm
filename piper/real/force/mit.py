"""运行：python3 -m piper.real.force.mit，MIT 位置保持并补偿重力。"""

import time
import numpy as np
import pinocchio as pin
from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.piper_new import PiperInterface
from piper.models import OFFICIAL_MODEL
from piper.real.digital_twin import DigitalTwin



CONTROL_PERIOD = 1 / 200  # s
PRINT_PERIOD = 0.1  # s
KP = 5
KD = 0.05


STARTUP_TIMEOUT = 30.0
STARTUP_TOLERANCE = 0.02  # rad
INITIAL_POSITION = np.array([0.2, 0.1, 0.45])
def move_to_initial_pose(piper, control):
    """调用现有 IK，以 10% 位置速度就位；工具水平，yaw 自由。"""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not piper.has_feedback():
        if time.monotonic() >= deadline:
            raise TimeoutError("等待真机关节反馈超时")
        time.sleep(CONTROL_PERIOD)
    q = piper.read_joint_positions()
    target = control.ik(q, INITIAL_POSITION,check=False)
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while True:
        # 就位完成前一直使用普通位置模式，不施加空间弹簧力。
        piper.send_joint_positions(target, speed_percent=10)
        enabled = piper.enable()
        q = piper.read_joint_positions()
        if enabled and np.all(np.abs(q - target) <= STARTUP_TOLERANCE):
            return target
        if time.monotonic() >= deadline:
            raise TimeoutError("30 秒内未到达初始末端位姿，停止进入力控制")
        time.sleep(CONTROL_PERIOD)
def main():
    control = PinocchioControl(model_path=OFFICIAL_MODEL)
    piper = PiperInterface(can_name="can_piper", mode="position")
    twin = None
    np.set_printoptions(precision=6, suppress=True)
    try:
        hold_positions = move_to_initial_pose(piper, control)
        q = piper.read_joint_positions()
        twin = DigitalTwin(q, INITIAL_POSITION.copy(), model_path=OFFICIAL_MODEL)
        # 保持给定末端初始位姿对应的关节角。
        print("保持目标(rad):", hold_positions)
        print("力矩反馈为电流估算幅值；包含 PD 输出，幅值差不等于力矩跟踪误差。")
        next_tick = time.perf_counter()
        next_print = next_tick
        while twin.process.is_alive() and not twin.stop.is_set():
            q = piper.read_joint_positions()
            now = time.perf_counter()
            gravity_torque = control.gravity(q)
            # gravity() 返回输出端 N·m；缩放与方向补偿由真机接口处理。
            piper.send_joint_torques(gravity_torque, positions=hold_positions, kp=KP, kd=KD)
            twin.update(q)

            if now >= next_print:
                feedback_torque = piper.read_joint_torques()
                feedforward_magnitude = gravity_torque
                print("关节角(rad):", q)
                print("相对保持目标的偏差(rad):", q - hold_positions)
                print("前馈幅值(N·m):", feedforward_magnitude)
                print("反馈幅值(N·m):", feedback_torque)
                next_print = now + PRINT_PERIOD

            next_tick += CONTROL_PERIOD
            time.sleep(max(0, next_tick - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if twin is not None:
                twin.close()
        finally:
            piper.close()


if __name__ == "__main__":
    main()
