"""离线验证被控对象与控制模型分离，不连接真机。"""

import unittest

import mujoco
import numpy as np

from piper.interface.mujoco import MujocoInterface
from piper.models import MUJOCO_MODEL, OFFICIAL_MODEL
from piper.sim.torque.computed_torque import TorqueSimulation


class MujocoPlantTests(unittest.TestCase):
    def test_plant_changes_dynamics_but_preserves_kinematics(self):
        nominal = MujocoInterface("torque", model_path=OFFICIAL_MODEL)
        plant = MujocoInterface("torque", model_path=MUJOCO_MODEL)
        q = np.array([0.2, 1.0, -1.1, 0.2, 0.4, -0.3])
        np.testing.assert_allclose(plant.fk(q), nominal.fk(q), atol=1e-12)
        self.assertGreater(np.linalg.norm(plant.model.body_mass - nominal.model.body_mass), 0.01)
        self.assertGreater(np.linalg.norm(plant.model.body_inertia - nominal.model.body_inertia), 1e-5)
        np.testing.assert_allclose(plant.model.dof_frictionloss[plant.v_ids],
                                   [0.24, 0.5, 0.44, 0.3, 0.36, 0.15])
        self.assertTrue(np.all(plant.model.dof_damping[plant.v_ids] > 0))
        np.testing.assert_array_equal(nominal.model.dof_frictionloss[nominal.v_ids], 0)
        self.assertTrue(np.all(plant.model.body_inertia[1:-1] >= 0))

    def test_nominal_controller_leaves_residual_error_on_mismatched_plant(self):
        errors = []
        for model_path in (OFFICIAL_MODEL, MUJOCO_MODEL):
            scene = TorqueSimulation(model_path=model_path)
            self.assertEqual(scene.control.model_path, OFFICIAL_MODEL.resolve())
            # 从目标附近释放，检验能否克服摩擦继续回到目标。
            scene.sim.data.qpos[scene.sim.q_ids] += [0.03, 0.02, -0.03, 0.03, 0.03, -0.03]
            mujoco.mj_forward(scene.sim.model, scene.sim.data)
            for _ in range(500):
                scene.step()
            mujoco.mj_forward(scene.sim.model, scene.sim.data)
            self.assertTrue(np.all(np.isfinite(scene.sim.data.qpos)))
            errors.append(np.linalg.norm(scene.q_target - scene.sim.read_joint_positions()))
        self.assertLess(errors[0], 1e-5)
        self.assertGreater(errors[1], 1e-3)

    def test_hardened_friction_suppresses_creep_without_changing_force_limit(self):
        speeds = []
        for soft in (True, False):
            scene = TorqueSimulation()
            sim = scene.sim
            if soft:
                sim.model.dof_solimp[sim.v_ids] = [0.9, 0.95, 0.001, 0.5, 2]
                sim.model.dof_solref[sim.v_ids] = [0.02, 1]
            sim.data.qpos[sim.q_ids] += [0.03, 0.02, -0.03, 0.03, 0.03, -0.03]
            mujoco.mj_forward(sim.model, sim.data)
            for _ in range(500):
                scene.step()
            speeds.append(np.max(np.abs(sim.data.qvel[sim.v_ids])))
        self.assertLess(speeds[1], speeds[0] * 0.01)
        self.assertLess(speeds[1], 2e-6)


if __name__ == "__main__":
    unittest.main()
