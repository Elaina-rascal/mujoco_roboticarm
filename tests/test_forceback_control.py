"""用 MuJoCo 对照检查二维解析公式。"""

import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np

DEMO = Path(__file__).resolve().parents[1] / "demo/forceback"
SPEC = importlib.util.spec_from_file_location("forceback_control", DEMO / "control.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class ForcebackControlTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(DEMO / "joystick.xml"))
        self.data = mujoco.MjData(self.model)
        self.controller = control.Controller()
        self.samples = np.random.default_rng(42).uniform(-0.4, 0.4, (12, 2))

    def test_fk_and_jacobian(self):
        site = self.model.site("grip").id
        for q in self.samples:
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            jp, jr = np.zeros((3, 2)), np.zeros((3, 2))
            mujoco.mj_jacSite(self.model, self.data, jp, jr, site)
            np.testing.assert_allclose(control.fk(q), self.data.site_xpos[site], atol=1e-14)
            np.testing.assert_allclose(control.jacobian(q), jp, atol=1e-14)
            eps = 1e-6
            numeric = np.column_stack([(control.fk(q+eps*e)-control.fk(q-eps*e))/(2*eps)
                                       for e in np.eye(2)])
            np.testing.assert_allclose(control.jacobian(q), numeric, atol=1e-10)

    def test_dynamics_terms(self):
        for q in self.samples:
            dq = np.array([0.7, -0.4])
            self.data.qpos[:], self.data.qvel[:] = q, dq
            mujoco.mj_forward(self.model, self.data)
            full = np.zeros((2, 2))
            if hasattr(self.data, "qM"):
                mujoco.mj_fullM(self.model, full, self.data.qM)
            else:
                mujoco.mj_fullM(self.model, self.data, full)
            np.testing.assert_allclose(control.mass_matrix(q), full, atol=1e-13)
            np.testing.assert_allclose(control.coriolis_vector(q, dq)+control.gravity_vector(q),
                                       self.data.qfrc_bias, atol=1e-13)
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            np.testing.assert_allclose(control.gravity_vector(q), self.data.qfrc_bias, atol=1e-13)

    def test_joint_impedance_restoring_direction_and_damping(self):
        target = np.array([0.13, -0.17])
        self.controller.q_target = target
        np.testing.assert_allclose(self.controller.update(target, [0, 0]),
                                   control.gravity_vector(target), atol=1e-14)
        for q in self.samples:
            # 去掉重力补偿后，恢复力矩必须指向目标，做功降低角度误差。
            spring = self.controller.update(q, [0, 0])-control.gravity_vector(q)
            self.assertGreaterEqual(spring @ (target-q), 0)
            # 任一轴的误差不应通过控制律引入另一轴的弹簧力矩。
            for axis in range(2):
                displaced = target.copy()
                displaced[axis] += 0.1
                tau = self.controller.update(displaced, [0, 0])-control.gravity_vector(displaced)
                self.assertLess(tau[axis], 0)
                self.assertAlmostEqual(tau[1-axis], 0)
        dq = np.array([0.7, -0.4])
        damping_tau = self.controller.update(target, dq)-control.gravity_vector(target)
        self.assertLess(damping_tau @ dq, 0)
        gravity_only = control.Controller(stiffness=0, damping=0)
        np.testing.assert_allclose(gravity_only.update(target, dq), control.gravity_vector(target))

    def test_lever_mapping_matches_tip_jacobian(self):
        dq = np.array([0.7, -0.4])
        for q in self.samples:
            j = control.jacobian(q)
            r = self.controller.axis_levers(q)
            np.testing.assert_allclose(r, np.linalg.norm(j, axis=0), atol=1e-14)
            force = self.controller.impedance_force(q, dq)
            world_force = (j/r) @ force
            tau = self.controller.update(q, dq)-control.gravity_vector(q)
            np.testing.assert_allclose(tau, j.T @ world_force, atol=1e-13)
            self.assertAlmostEqual(tau @ dq, force @ (r*dq))

    def test_impedance_returns_to_target(self):
        self.data.qpos[:] = [0.28, -0.24]
        q_target = np.array([0.08, -0.10])
        controller = control.Controller(q_target=q_target)
        peak = 0.0
        for _ in range(3000):
            tau = controller.update(self.data.qpos, self.data.qvel)
            self.data.ctrl[:] = np.clip(tau, -17, 17)
            peak = max(peak, np.max(np.abs(tau)))
            mujoco.mj_step(self.model, self.data)
        np.testing.assert_allclose(self.data.qpos, q_target, atol=0.004)
        np.testing.assert_allclose(self.data.qvel, 0, atol=0.005)
        self.assertLess(peak, 17)
        self.assertTrue(all(w.number == 0 for w in self.data.warning))


if __name__ == "__main__":
    unittest.main()
