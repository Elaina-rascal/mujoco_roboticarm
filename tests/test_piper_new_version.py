"""新版接口的启动版本门槛；不测试力矩前馈，不连接 CAN。"""

from types import SimpleNamespace
import importlib
import unittest
from unittest.mock import MagicMock, patch

from piper.interface.piper_new import PiperInterface


class VersionGateTests(unittest.TestCase):
    def test_all_real_hardware_entrypoints_use_new_interface(self):
        for name in ("piper.real.position.run", "piper.real.force.mit",
                     "piper.real.force.vmc", "piper.real.estimate.trajectory_demo"):
            with self.subTest(module=name):
                self.assertIs(importlib.import_module(name).PiperInterface, PiperInterface)

    def setUp(self):
        self.factory = MagicMock()
        self.sdk = self.factory.create_arm.return_value
        self.sdk.get_fps.return_value = 200.0
        self.sdk.get_comm_error.return_value = None
        self.config = MagicMock(return_value={"joint_torque_c": [1.] * 6})
        self.module = SimpleNamespace(
            AgxArmFactory=self.factory, create_agx_arm_config=self.config,
            resolve_firmware_profile=MagicMock(return_value="v189"),
        )

    def test_188_and_older_disconnect_before_control_mode(self):
        for version in ("S-V1.5-8", "S-V1.8-7", "S-V1.8-8"):
            self.sdk.reset_mock()
            self.sdk.get_firmware.return_value = {"software_version": version}
            with self.subTest(version=version), \
                 patch.dict("sys.modules", {"pyAgxArm": self.module}):
                with self.assertRaises(NotImplementedError):
                    PiperInterface()
            self.sdk.disconnect.assert_called_once()
            self.sdk.set_motion_mode.assert_not_called()

    def test_newer_versions_use_matching_profile(self):
        for version in ("S-V1.8-9", "S-V1.9-1", "S-V1.10-1"):
            self.sdk.reset_mock()
            self.sdk.get_firmware.return_value = {"software_version": version + "\x00\n"}
            with self.subTest(version=version), \
                 patch.dict("sys.modules", {"pyAgxArm": self.module}):
                arm = PiperInterface()
                self.assertEqual(arm.firmware_version, version)
                self.module.resolve_firmware_profile.assert_called_with("piper", version)
                self.sdk.get_firmware.assert_called_once_with(timeout=3.0, min_interval=0.0)
        self.assertEqual(self.factory.create_arm.call_count, 3)

    def test_unknown_or_unreadable_version_disconnects(self):
        for reply, error in ((None, TimeoutError), ({"software_version": "unknown"}, ValueError),
                             ({"hardware_version": "H-V1.2-1"}, ValueError)):
            self.sdk.reset_mock()
            self.sdk.get_firmware.return_value = reply
            with self.subTest(reply=reply), patch.dict("sys.modules", {"pyAgxArm": self.module}):
                with self.assertRaises(error):
                    PiperInterface()
            self.sdk.disconnect.assert_called_once()
            self.sdk.set_motion_mode.assert_not_called()

    def test_cold_receiver_is_ready_before_first_query(self):
        events = []
        rates = iter([0.0, 0.0, 200.0])
        self.sdk.get_fps.side_effect = lambda: events.append("fps") or next(rates)
        self.sdk.get_firmware.side_effect = lambda **kw: events.append("query") or {"software_version": "S-V1.9-1"}
        with patch.dict("sys.modules", {"pyAgxArm": self.module}), \
             patch("piper.interface.piper_new.time.sleep") as sleep:
            PiperInterface()
        self.assertEqual(events, ["fps", "fps", "fps", "query"])
        self.assertEqual(sleep.call_count, 2)

    def test_partial_or_lost_reply_is_retried(self):
        self.sdk.get_firmware.side_effect = [None, {"software_version": "S-V1.9-1"}]
        with patch.dict("sys.modules", {"pyAgxArm": self.module}):
            arm = PiperInterface()
        self.assertEqual(arm.firmware_version, "S-V1.9-1")
        self.assertEqual(self.sdk.get_firmware.call_count, 2)
        self.sdk.get_firmware.assert_called_with(timeout=3.0, min_interval=0.0)

    def test_no_feedback_times_out_without_query_or_motion(self):
        self.sdk.get_fps.return_value = 0.0
        with patch.dict("sys.modules", {"pyAgxArm": self.module}), \
             patch("piper.interface.piper_new.time.monotonic", side_effect=[0, 0, 3]), \
             patch("piper.interface.piper_new.time.sleep"):
            with self.assertRaisesRegex(TimeoutError, "CAN 未收到有效反馈"):
                PiperInterface()
        self.sdk.get_firmware.assert_not_called()
        self.sdk.set_motion_mode.assert_not_called()
        self.sdk.disconnect.assert_called_once()

    def test_comm_error_during_query_stops_without_entering_control(self):
        self.sdk.get_firmware.return_value = None
        errors = iter([None, OSError("Network is down")])
        self.sdk.get_comm_error.side_effect = lambda: next(errors)
        with patch.dict("sys.modules", {"pyAgxArm": self.module}):
            with self.assertRaisesRegex(ConnectionError, "CAN 通信异常"):
                PiperInterface()
        self.sdk.get_firmware.assert_called_once()
        self.sdk.set_motion_mode.assert_not_called()
        self.sdk.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
