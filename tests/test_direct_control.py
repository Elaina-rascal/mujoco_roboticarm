"""验证执行器映射和两种控制模式。"""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mujoco_arm.simulation import MujocoArm


class DirectControlTest(unittest.TestCase):
    def test_position_control_moves_toward_target(self):
        arm = MujocoArm()
        target = arm.data.qpos[arm.qpos_indices].copy()
        target[0] += 0.2
        initial_error = np.linalg.norm(arm.data.qpos[arm.qpos_indices] - target)
        arm.set_control(target)
        for _ in range(500):
            arm.step()
        self.assertLess(np.linalg.norm(arm.data.qpos[arm.qpos_indices] - target), initial_error)
        self.assertTrue(np.all(np.isfinite(arm.joint_state()["velocity"])))
        self.assertIn("wrist_3_link", arm.body_transforms())

    def test_torque_mapping_and_limits(self):
        arm = MujocoArm("torque")
        # 电机名 shoulder_pan 与关节名 shoulder_pan_joint 不同，仍应正确应用。
        arm.set_control([200, -200, 5, 40, -40, 1])
        np.testing.assert_allclose(arm.data.ctrl[arm.actuator_ids], [150, -150, 5, 28, -28, 1])
        arm.step()
        self.assertGreater(arm.data.time, 0)
        self.assertTrue(np.all(np.isfinite(arm.data.qpos)))

    def test_invalid_control_does_not_change_command(self):
        arm = MujocoArm()
        previous = arm.data.ctrl.copy()
        for command in ([0] * 5, [float("nan")] * 6):
            with self.assertRaises(ValueError):
                arm.set_control(command)
            np.testing.assert_array_equal(arm.data.ctrl, previous)



if __name__ == "__main__":
    unittest.main()
