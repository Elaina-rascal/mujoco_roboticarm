"""六自由度误差、雅克比功率一致性与离线动态跟踪；不连接真机。"""

import unittest

import mujoco
import numpy as np
import pinocchio as pin

from piper.controllers.impedance_controller import ImpedanceController
from piper.interface.mujoco import EE_TO_BODY_ROTATION
from piper.sim.impedance.run import create_simulation, control_step


class ImpedanceTests(unittest.TestCase):
    def setUp(self):
        self.controller = ImpedanceController()
        self.q = np.array([0.2, 1.0, -1.1, 0.2, 0.4, -0.3])
        self.pose = self.controller.control.fk(self.q)

    def test_six_dimensional_jacobian_and_wrench_mapping(self):
        # 从实际 FK 的平移和世界系相对旋转独立差分，验证工具旋转约定。
        columns = []
        for i in range(6):
            delta = np.eye(6)[i] * 1e-6
            plus = self.controller.control.fk(self.q + delta)
            minus = self.controller.control.fk(self.q - delta)
            columns.append(np.r_[plus.translation - minus.translation,
                                     pin.log3(plus.rotation @ minus.rotation.T)] / 2e-6)
        jacobian = np.column_stack(columns)
        dq = np.array([0.1, -0.2, 0.1, 0.05, -0.04, 0.02])
        command = self.controller.compute(
            self.q, dq, self.pose.translation + [0.02, -0.03, 0.01],
            pin.exp3(np.array([0.08, -0.05, 0.06])) @ self.pose.rotation,
        )
        np.testing.assert_allclose(command.twist, jacobian @ dq, atol=1e-9)
        np.testing.assert_allclose(command.force_torque, jacobian.T @ command.wrench, atol=1e-8)
        np.testing.assert_allclose(command.error[3:], [0.08, -0.05, 0.06], atol=1e-12)
        self.assertAlmostEqual(command.force_torque @ dq, command.wrench @ command.twist)

    def test_equilibrium_rotational_damping_and_limits(self):
        equilibrium = self.controller.compute(self.q, np.zeros(6), self.pose.translation, self.pose.rotation)
        np.testing.assert_allclose(equilibrium.torque, equilibrium.gravity, atol=1e-12)
        dq = np.arange(1, 7) * 0.02
        command = self.controller.compute(self.q, dq, self.pose.translation, self.pose.rotation)
        self.assertLess(command.force_torque @ dq, 0)
        self.assertAlmostEqual(command.force_torque @ dq,
                               -np.sum(self.controller.kd * command.twist**2))
        limited = self.controller.compute(
            self.q, np.zeros(6), self.pose.translation + [10, -10, 10],
            pin.exp3(np.ones(3) * 1.2) @ self.pose.rotation,
        )
        np.testing.assert_allclose(limited.wrench, [20, -20, 20, 5, 5, 5])
        free = self.controller.compute(self.q, dq, self.pose.translation)
        np.testing.assert_array_equal(free.wrench[3:], 0)

    def test_invalid_rotation_is_rejected(self):
        for rotation in (np.zeros((3, 3)), np.diag([1, 1, -1]), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                self.controller.compute(self.q, np.zeros(6), self.pose.translation, rotation)

    def test_simulation_tracks_position_and_all_three_rotation_directions(self):
        # 目标由邻近关节位姿生成，确保物理可达；逐个验证世界旋转分量均收敛。
        sim, controller = create_simulation()
        sim.data.qpos[sim.q_ids] = self.q
        mujoco.mj_forward(sim.model, sim.data)
        pose = controller.control.fk(self.q)
        np.testing.assert_allclose(
            sim.data.site_xmat[sim.ee_id].reshape(3, 3) @ EE_TO_BODY_ROTATION,
            pose.rotation, atol=1e-12,
        )
        q_target = self.q + [0.04, 0.03, -0.04, 0.04, -0.03, 0.04]
        target = controller.control.fk(q_target)
        sim.set_target_pose(target.translation, target.rotation)
        position, rotation = sim.target_pose()
        np.testing.assert_allclose(position, target.translation)
        np.testing.assert_allclose(rotation, target.rotation, atol=1e-12)
        initial = controller.compute(self.q, np.zeros(6), position, rotation).error
        self.assertTrue(np.all(np.abs(initial) > 1e-4))
        for _ in range(4000):
            command = control_step(sim, controller)
        self.assertTrue(np.all(np.isfinite(sim.data.qpos)))
        self.assertLess(np.linalg.norm(command.error[:3]), 5e-4)
        self.assertLess(np.linalg.norm(command.error[3:]), 2e-3)


if __name__ == "__main__":
    unittest.main()
