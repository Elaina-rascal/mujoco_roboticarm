"""重构回归：仿真行为、模型路径与真机 SDK 的单位/方向转换。"""

import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np

from piper.sim.torque.controller import GravityController
from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.base import ArmInterface
from piper.interface.mujoco import MujocoInterface
from piper.interface.piper import PiperInterface
from piper.models import ARM_MODEL, OFFICIAL_MODEL


class PiperInterfaceTests(unittest.TestCase):
    def test_position_tracking_and_feedback(self):
        arm = MujocoInterface("position")
        self.assertIsInstance(arm, ArmInterface)
        target = arm.read_joint_positions()
        target[0] += 0.1
        arm.send_joint_positions(target)
        arm.step(500)
        self.assertLess(np.linalg.norm(arm.read_joint_positions() - target), 0.02)
        self.assertEqual(arm.read_joint_torques().shape, (6,))
        feedback = arm.read_joint_positions()
        feedback[:] = 100
        self.assertLess(np.max(np.abs(arm.read_joint_positions())), 10)

    def test_torque_compensation_and_limits(self):
        arm = MujocoInterface("torque")
        controller = GravityController()
        q_initial = arm.read_joint_positions()
        torque = controller.compute(arm.data.qpos)
        arm.send_joint_torques(torque)
        arm.step(50)
        np.testing.assert_allclose(arm.read_joint_positions(), q_initial, atol=1e-8)
        np.testing.assert_allclose(arm.read_joint_torques(), torque, atol=1e-9)
        arm.send_joint_torques([200, -200, 5, 40, -40, 1])
        arm.step()
        np.testing.assert_allclose(arm.read_joint_torques(), [100, -100, 5, 40, -40, 1])

    def test_invalid_commands_leave_simulation_unchanged(self):
        for mode in ("position", "torque"):
            arm = MujocoInterface(mode)
            send = arm.send_joint_positions if mode == "position" else arm.send_joint_torques
            previous = arm.data.ctrl.copy()
            for command in ([0] * 5, [float("nan")] * 6):
                with self.assertRaises(ValueError):
                    send(command)
                np.testing.assert_array_equal(arm.data.ctrl, previous)

    def test_models_and_entrypoints_load_without_hardware(self):
        for path in (ARM_MODEL, OFFICIAL_MODEL):
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(model.nq, 8)
            control = PinocchioControl(model_path=path)
            self.assertEqual(control.gravity(np.zeros(6)).shape, (6,))
        for module in ("piper.sim.position.run", "piper.sim.torque.run",
                       "piper.real.position.run", "piper.real.force.mit",
                       "piper.real.force.vmc",
                       "piper.real.digital_twin"):
            importlib.import_module(module)

    def test_real_position_units_and_lifecycle(self):
        # 用 SDK 替身验证协议换算，测试不会打开 CAN 或发送真机指令。
        sdk = MagicMock()
        sdk.GetPiperFirmwareVersion.return_value = "S-V1.5-8"
        sdk.EnablePiper.return_value = True
        sdk.GetArmJointMsgs.return_value = SimpleNamespace(
            time_stamp=1,
            joint_state=SimpleNamespace(**{f"joint_{i}": 1000 * i for i in range(1, 7)}),
        )
        sdk.GetArmStatus.return_value = SimpleNamespace(arm_status=SimpleNamespace(mode_feed=1))
        factory = MagicMock(return_value=sdk)
        with patch.dict("sys.modules", {"piper_sdk": SimpleNamespace(C_PiperInterface_V2=factory)}):
            arm = PiperInterface(mode="position")
            q = np.deg2rad(np.arange(1, 7))
            np.testing.assert_allclose(arm.read_joint_positions(), q)
            arm.send_joint_positions(q)
            sdk.JointCtrl.assert_called_with(1000, 2000, 3000, 4000, 5000, 6000)
            self.assertTrue(arm.enable())
            self.assertTrue(arm.has_feedback())
            arm.close()
            sdk.DisconnectPort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
