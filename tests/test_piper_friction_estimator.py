"""理想单轴摩擦替身与虚拟时钟验证；不加载 SDK 或运行真机。"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from piper.estimate.friction_estimator import FrictionEstimator


class FakeArm:
    def __init__(self, static=0.2, bias=0.03):
        self.q = np.array([0., 1., -1., 0., 0.3, 0.])
        self.v = 0.0
        self.command = np.zeros(6)
        self.static, self.bias = static, bias
        self.mode = "position"
        self.torque_limits = np.full(6, 16.)
        self.position_commands = []

    def read_joint_positions(self):
        return self.q.copy()

    def send_joint_positions(self, q, speed_percent):
        self.mode = "position"
        self.q = np.array(q).copy()
        self.v = 0.0
        self.position_commands.append(self.q.copy())

    def send_joint_torques(self, torque, **kwargs):
        assert kwargs["kp"] == kwargs["kd"] == 0
        self.mode = "torque"
        self.command = np.array(torque).copy()

    def advance(self, dt):
        if self.mode != "torque":
            return
        for _ in range(10):
            h = dt / 10
            drive = self.command[0] - self.bias
            if abs(self.v) < 1e-9 and abs(drive) <= self.static:
                self.v = 0.0
                continue
            friction = 0.15 * np.sign(self.v if abs(self.v) >= 1e-9 else drive)
            new_v = self.v + (drive - friction) / 0.2 * h
            if self.v * new_v < 0 and abs(drive) <= self.static:
                new_v = 0.0
            self.v = new_v
            self.q[0] += self.v * h


class FrictionEstimatorTests(unittest.TestCase):
    def estimator(self, arm):
        timer = [0.0]

        def sleep(dt):
            arm.advance(dt)
            timer[0] += dt

        control = SimpleNamespace(
            model=SimpleNamespace(lowerPositionLimit=np.full(6, -3.),
                                  upperPositionLimit=np.full(6, 3.)),
            gravity=lambda q: np.zeros(6),
        )
        clock_patch = patch("piper.estimate.friction_estimator.time.monotonic", side_effect=lambda: timer[0])
        sleep_patch = patch("piper.estimate.friction_estimator.time.sleep", side_effect=sleep)
        clock_patch.start()
        sleep_patch.start()
        self.addCleanup(clock_patch.stop)
        self.addCleanup(sleep_patch.stop)
        return FrictionEstimator(arm, control)

    def test_bidirectional_half_difference_and_return(self):
        arm = FakeArm()
        reference = arm.q.copy()
        estimator = self.estimator(arm)
        result = estimator.measure(1)
        self.assertAlmostEqual(result["friction"], 0.2, delta=0.05)
        self.assertAlmostEqual(result["bias"], 0.03, delta=0.05)
        self.assertGreater(result["positive"], 0)
        self.assertLess(result["negative"], 0)
        # 正向测完回位，反向测完回位，轨迹都达到当前设定行程。
        angles = np.array(arm.position_commands)[:, 0]
        self.assertGreater(angles.max(), estimator.travel)
        self.assertLess(angles.min(), -estimator.travel)
        np.testing.assert_allclose(arm.q, reference, atol=1e-8)
        self.assertEqual(arm.mode, "position")

    def test_no_motion_stops_at_torque_limit_and_holds(self):
        arm = FakeArm(static=2.0)
        estimator = self.estimator(arm)
        estimator.max_torque = 0.15
        with self.assertRaisesRegex(RuntimeError, "附加力矩上限"):
            estimator.measure(1)
        self.assertEqual(arm.mode, "position")

    def test_overspeed_aborts_and_holds(self):
        arm = FakeArm()
        estimator = self.estimator(arm)
        advance = arm.advance

        def overspeed(dt):
            advance(dt)
            if arm.mode == "torque":
                arm.q[0] += 0.1

        arm.advance = overspeed
        with self.assertRaisesRegex(RuntimeError, "超速"):
            estimator.measure(1)
        self.assertEqual(arm.mode, "position")

    def test_invalid_joint(self):
        estimator = self.estimator(FakeArm())
        for joint in (0, 7, True):
            with self.assertRaises(ValueError):
                estimator.measure(joint)


if __name__ == "__main__":
    unittest.main()
