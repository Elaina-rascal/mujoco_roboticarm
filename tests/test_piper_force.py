"""XYZ 阻抗的几何、逐轴限幅及真机入口契约；不连接 CAN。"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.models import OFFICIAL_MODEL
from piper.real.force.vmc import move_to_initial_pose
from piper.real.force import mit, vmc as run


class XYZImpedanceTests(unittest.TestCase):
    def setUp(self):
        self.controller = PinocchioControl(model_path=OFFICIAL_MODEL)
        self.q = np.array([0.2, 1.0, -1.1, 0.2, 0.4, -0.3])
        self.position = self.controller.fk(self.q).translation.copy()

    def numerical_jacobian(self):
        step = 1e-6
        columns = []
        for index in range(6):
            delta = np.zeros(6)
            delta[index] = step
            plus = self.controller.fk(self.q + delta).translation.copy()
            minus = self.controller.fk(self.q - delta).translation.copy()
            columns.append((plus - minus) / (2 * step))
        return np.column_stack(columns)

    def test_translation_jacobian_force_mapping_and_each_axis_limit(self):
        dq = np.array([0.1, -0.2, 0.1, 0.05, -0.04, 0.02])
        numerical = self.numerical_jacobian()
        command = run.compute_impedance(self.controller, self.q, dq, self.position + [1, -1, 1])
        # 三轴同时饱和，合力可以超过 20 N。
        np.testing.assert_allclose(command.force, [20, -20, 20])
        self.assertGreater(np.linalg.norm(command.force), 20)
        np.testing.assert_allclose(command.velocity, numerical @ dq, atol=1e-9)
        np.testing.assert_allclose(command.torque - command.gravity,
                                   numerical.T @ command.force, atol=1e-8)

    def test_equilibrium_and_damping_remove_energy(self):
        command = run.compute_impedance(self.controller, self.q, np.zeros(6), self.position)
        np.testing.assert_allclose(command.force, 0, atol=1e-12)
        np.testing.assert_allclose(command.torque, command.gravity, atol=1e-12)
        dq = np.array([0.1, -0.1, 0.1, 0.02, -0.04, 0.02])
        command = run.compute_impedance(self.controller, self.q, dq, self.position)
        power = (command.torque - command.gravity) @ dq
        self.assertLess(power, 0)
        self.assertAlmostEqual(power, -np.sum(run.KD * command.velocity**2), places=10)

    def test_initial_ik_keeps_tool_level_with_yaw_free(self):
        import pinocchio as pin
        for seed in (np.zeros(6), np.array([0.1, 0.5, -0.3, 0, -0.2, 0])):
            q = self.controller.ik(seed, run.INITIAL_POSITION)
            pose = self.controller.fk(q)
            error = self.controller.H @ np.r_[run.INITIAL_POSITION - pose.translation,
                                              -pin.log3(pose.rotation)]
            self.assertLessEqual(np.linalg.norm(error), self.controller.max_task_error)
            self.assertGreater(abs(pin.log3(pose.rotation)[2]), 0.05)

    def test_startup_arrives_in_position_mode_before_force_control(self):
        target = self.controller.ik(np.zeros(6), run.INITIAL_POSITION)
        arm = MagicMock()
        arm.has_feedback.return_value = True
        arm.read_joint_positions.side_effect = [np.zeros(6), target.copy()]
        arm.enable.return_value = True
        np.testing.assert_allclose(move_to_initial_pose(arm, self.controller), target)
        arm.send_joint_positions.assert_called_once()
        arm.send_joint_torques.assert_not_called()

    def test_startup_timeout_does_not_enter_force_control(self):
        arm = MagicMock()
        arm.has_feedback.return_value = True
        arm.read_joint_positions.return_value = np.zeros(6)
        arm.enable.return_value = True
        with patch("piper.real.force.vmc.time.monotonic", side_effect=[0, 0, 31]):
            with self.assertRaises(TimeoutError):
                move_to_initial_pose(arm, self.controller)
        arm.send_joint_torques.assert_not_called()

    def test_entrypoint_sends_pure_torque_and_closes_resources(self):
        arm = MagicMock()
        arm.has_feedback.return_value = True
        arm.read_joint_positions.return_value = self.q.copy()
        arm.enable.return_value = True
        twin = MagicMock()
        twin.process.is_alive.side_effect = [True, False]
        twin.stop.is_set.return_value = False
        twin.read_target_position.return_value = self.position.copy()
        with patch.object(run, "PiperInterface", return_value=arm), \
             patch.object(run, "DigitalTwin", return_value=twin), \
             patch.object(run, "move_to_initial_pose"), \
             patch("builtins.print"):
            run.main()
        self.assertEqual(arm.send_joint_torques.call_count, 1)
        for call in arm.send_joint_torques.call_args_list:
            self.assertEqual(call.kwargs, {"kp": 0.0, "kd": 0.0})
            np.testing.assert_allclose(call.args[0], self.controller.gravity(self.q))
        twin.close.assert_called_once()
        arm.close.assert_called_once()

    def test_mit_holds_initial_pose_with_joint_pd_and_gravity(self):
        target = self.controller.ik(np.zeros(6), run.INITIAL_POSITION)
        arm = MagicMock()
        arm.read_joint_positions.return_value = target.copy()
        twin = MagicMock()
        twin.process.is_alive.side_effect = [True, False]
        twin.stop.is_set.return_value = False
        with patch.object(mit, "PiperInterface", return_value=arm), \
             patch.object(mit, "DigitalTwin", return_value=twin), \
             patch.object(mit, "move_to_initial_pose", return_value=target), \
             patch("builtins.print"):
            mit.main()
        arm.send_joint_torques.assert_called_once()
        call = arm.send_joint_torques.call_args
        np.testing.assert_allclose(call.args[0], self.controller.gravity(target))
        np.testing.assert_allclose(call.kwargs["positions"], target)
        self.assertEqual((call.kwargs["kp"], call.kwargs["kd"]), (mit.KP, mit.KD))
        twin.close.assert_called_once()
        arm.close.assert_called_once()

    def test_invalid_command_exits_through_position_hold(self):
        arm = MagicMock()
        arm.has_feedback.return_value = True
        arm.read_joint_positions.return_value = self.q.copy()
        arm.enable.return_value = True
        twin = MagicMock()
        twin.process.is_alive.return_value = True
        twin.stop.is_set.return_value = False
        twin.read_target_position.return_value = [np.nan, 0, 0]
        with patch.object(run, "PiperInterface", return_value=arm), \
             patch.object(run, "DigitalTwin", return_value=twin), \
             patch.object(run, "move_to_initial_pose"), \
             patch("builtins.print"):
            with self.assertRaises(ValueError):
                run.main()
        # 非法目标未发出任何阻抗指令。
        self.assertEqual(arm.send_joint_torques.call_count, 0)
        twin.close.assert_called_once()
        arm.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
