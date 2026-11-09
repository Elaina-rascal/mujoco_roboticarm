"""验证双轴摇杆的物理结构和输出端力矩接口。"""

import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "demo/forceback/simulation.py"
SPEC = importlib.util.spec_from_file_location("forceback_simulation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
JoystickSimulation = MODULE.JoystickSimulation


class ForcebackTest(unittest.TestCase):
    def test_mass_distribution_and_joint_inertia(self):
        sim = JoystickSimulation()
        model, data = sim.model, sim.data
        self.assertEqual((model.nq, model.nv, model.nu), (2, 2, 2))
        outer = model.body("motor_mount").id
        motor_y = model.body("motor_y_housing").id
        self.assertEqual(model.body_parentid[motor_y], outer)
        self.assertAlmostEqual(model.body_mass[motor_y], 0.38)
        self.assertAlmostEqual(model.body_subtreemass[outer], 1.14)
        inertia = np.zeros((2, 2))
        # Read the public packed mass matrix independently of wrapper state.
        if hasattr(data, "qM"):  # MuJoCo 3.x
            mujoco.mj_fullM(model, inertia, data.qM)
        else:  # MuJoCo 4.x
            mujoco.mj_fullM(model, data, inertia)
        stick_pivot = 0.5 * (3 * 0.012**2 + 0.195**2) / 12 + 0.5 * 0.1525**2
        hub = 0.08 * (0.04**2 + 0.06**2) / 12
        motor_transverse = 0.38 * (3 * 0.03925**2 + 0.04**2) / 12
        expected_y = stick_pivot + hub + 0.08 * 0.028**2
        expected_x = expected_y + 0.08 * 0.016**2 + 0.0001455 + 0.18 * 0.020**2 + motor_transverse
        np.testing.assert_allclose(inertia, np.diag([expected_x, expected_y]), atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(inertia) > 0))

    def test_grip_radius_at_different_angles(self):
        sim = JoystickSimulation()
        for angles in ([0, 0], [0.2, -0.3], [-0.4, 0.1]):
            sim.data.qpos[:] = angles
            mujoco.mj_forward(sim.model, sim.data)
            # 握持距离以第二轴的世界坐标轴心为基准，允许两轴安装位置分离。
            pivot = sim.data.xanchor[sim.model.joint("stick_y").id]
            offset = sim.state()["grip_position"] - pivot
            self.assertAlmostEqual(np.linalg.norm(offset), 0.25)
        sim.data.qpos[:] = [0, 0]
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose(sim.state()["grip_position"], [0, 0, 0.34])

    def test_motor_centers_and_stick_share_first_output_plane(self):
        sim = JoystickSimulation()
        model, data = sim.model, sim.data
        center_x = data.xpos[model.body("motor_x_housing").id]
        center_y = data.xpos[model.body("motor_y_housing").id]
        # 零位：左电机输出沿 X，中间电机中心在该轴线上，摇杆也在 Y=0 平面。
        np.testing.assert_allclose(data.xaxis[model.joint("stick_x").id], [1, 0, 0])
        np.testing.assert_allclose(data.xaxis[model.joint("stick_y").id], [0, 1, 0])
        np.testing.assert_allclose((center_y - center_x)[1:], [0, 0])
        self.assertAlmostEqual(data.xpos[model.body("stick").id][1], 0)
        self.assertAlmostEqual(sim.state()["grip_position"][1], 0)

    def test_torque_limits_and_mapping(self):
        sim = JoystickSimulation()
        for command, expected in (([100, -100], [17, -17]), ([0.3, -0.7], [0.3, -0.7])):
            sim.set_torque(command)
            mujoco.mj_forward(sim.model, sim.data)
            np.testing.assert_allclose(sim.state()["actuator_torque"], expected)
        previous = sim.data.ctrl.copy()
        for invalid in ([1], [0, float("nan")], [0, float("inf")]):
            with self.assertRaises(ValueError):
                sim.set_torque(invalid)
            np.testing.assert_array_equal(sim.data.ctrl, previous)

    def test_motor_y_housing_follows_only_first_axis(self):
        sim = JoystickSimulation()
        body_id = sim.model.body("motor_y_housing").id
        initial = sim.data.xpos[body_id].copy()
        initial_rotation = sim.data.xmat[body_id].copy()
        sim.data.qpos[:] = [0, 0.3]
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose(sim.data.xpos[body_id], initial)
        np.testing.assert_allclose(sim.data.xmat[body_id], initial_rotation)
        sim.data.qpos[:] = [0.3, 0.3]
        mujoco.mj_forward(sim.model, sim.data)
        # 电机中心位于第一轴上，位置不变，但壳体姿态随第一轴转动。
        np.testing.assert_allclose(sim.data.xpos[body_id], initial)
        self.assertGreater(np.linalg.norm(sim.data.xmat[body_id] - initial_rotation), 0.1)

    def test_disturbed_zero_torque_and_loaded_stops_are_stable(self):
        sim = JoystickSimulation()
        sim.data.qpos[:] = [0.1, -0.15]
        for command in ([0, 0], [17, -17], [-17, 17]):
            sim.set_torque(command)
            for _ in range(2000):
                sim.step()
            self.assertTrue(np.all(np.isfinite(sim.data.qpos)))
            self.assertTrue(np.all(np.isfinite(sim.data.qvel)))
            self.assertLess(np.max(np.abs(sim.data.qpos)), np.deg2rad(35))
        self.assertAlmostEqual(sim.data.time, 6.0, places=8)
        self.assertTrue(all(w.number == 0 for w in sim.data.warning))


if __name__ == "__main__":
    unittest.main()
