"""运行：python3 -m piper.real.estimate.friction_estimation，依次测试真机 J1～J6。"""

import time

import numpy as np

from piper.estimate.friction_estimator import FrictionEstimator


def main():
    from piper.controllers.pinocchio_controller import PinocchioControl
    from piper.interface.piper_new import PiperInterface
    from piper.models import OFFICIAL_MODEL

    control = PinocchioControl(model_path=OFFICIAL_MODEL)
    arm = PiperInterface(can_name="can_piper", mode="position")
    results = []
    try:
        deadline = time.monotonic() + 5.0
        while not arm.has_feedback():
            if time.monotonic() > deadline:
                raise TimeoutError("等待关节角反馈超时")
            time.sleep(0.01)

        q = arm.read_joint_positions()
        estimator = FrictionEstimator(arm, control)
        # 六轴都要有双向行程；不自动移动到其他构型。
        margin = estimator.travel + np.deg2rad(4)
        lower = control.model.lowerPositionLimit
        upper = control.model.upperPositionLimit
        for i in range(6):
            if q[i] - margin <= lower[i] or q[i] + margin >= upper[i]:
                raise ValueError(f"J{i + 1} 双向行程不足，请先用位置模式调整姿态")

        # 先给当前位置目标，再使能，避免沿用旧的位置目标。
        deadline = time.monotonic() + 5.0
        while True:
            arm.send_joint_positions(q, speed_percent=5)
            if arm.enable():
                break
            if time.monotonic() > deadline:
                raise TimeoutError("使能机械臂超时")
            time.sleep(0.02)

        print(f"依次测试 J1～J6：每档 {estimator.torque_step:.2f} N·m，"
              f"目标速度 {np.rad2deg(estimator.speed):g}°/s，"
              f"启动后双向各走 {np.rad2deg(estimator.travel):g}°。", flush=True)
        print("测试轴不加位置弹簧；其他轴保持。Ctrl+C 退出并保持当前位置。", flush=True)
        for joint in range(1, 7):
            print(f"开始 J{joint}：正向 → 回位 → 反向 → 回位。", flush=True)
            result = estimator.measure(joint)
            results.append(result)
            print(f"J{joint}：正向 {result['positive']:.3f}，反向 {result['negative']:.3f}，"
                  f"摩擦 {result['friction']:.3f}，偏置 {result['bias']:.3f} N·m", flush=True)
    except KeyboardInterrupt:
        print("已中断测试。", flush=True)
    finally:
        try:
            arm.close()
        finally:
            if results:
                print("已完成关节的摩擦估计(N·m):", flush=True)
                for i in range(len(results)):
                    print(f"J{i + 1}: {results[i]['friction']:.3f}", flush=True)


if __name__ == "__main__":
    main()
