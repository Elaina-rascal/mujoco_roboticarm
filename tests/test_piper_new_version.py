"""新版接口的启动版本门槛；不测试力矩前馈，不连接 CAN。"""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from piper.interface.piper_new import PiperInterface


class VersionGateTests(unittest.TestCase):
    def setUp(self):
        self.factory = MagicMock()
        self.sdk = self.factory.create_arm.return_value
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
                self.sdk.get_firmware.assert_called_once_with(timeout=3.0)
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


if __name__ == "__main__":
    unittest.main()
