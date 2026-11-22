"""六维 IK 的完整轨迹、check=False 和非收敛记录，不连接真机。"""

import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.models import OFFICIAL_MODEL
from piper.sim.position.ik6_timelapse import solve_ik6, INITIAL_Q, TARGET_POSITION


class IK6TraceTests(unittest.TestCase):
    def setUp(self):
        self.control = PinocchioControl(model_path=OFFICIAL_MODEL, gripper_width=.04)

    def test_all_iterations_and_full_pose_are_recorded_with_checks_disabled(self):
        trace = solve_ik6(self.control, INITIAL_Q, TARGET_POSITION, np.eye(3), check=False)
        self.assertTrue(trace['converged'])
        self.assertTrue(trace['accepted'])
        self.assertEqual(trace['task_dimension'], 6)
        self.assertEqual([s['iteration'] for s in trace['frames']], list(range(len(trace['frames']))))
        np.testing.assert_allclose(trace['frames'][0]['q'], INITIAL_Q)
        self.assertGreater(trace['frames'][-1]['max_joint_change_deg'], 50)
        for state in trace['frames']:
            pose = self.control.fk(state['q'])
            expected = np.r_[TARGET_POSITION - pose.translation, -pin.log3(pose.rotation)]
            np.testing.assert_allclose(state['error'], expected)
            target_pose = pin.SE3(np.eye(3), TARGET_POSITION)
            self.assertAlmostEqual(state['log6_error_norm'],
                                   np.linalg.norm(pin.log6(pose.actInv(target_pose)).vector))
        checked = solve_ik6(self.control, INITIAL_Q, TARGET_POSITION, np.eye(3), check=True)
        self.assertFalse(checked['accepted'])
        np.testing.assert_allclose(checked['frames'][-1]['q'], trace['frames'][-1]['q'])

    def test_nonconverged_attempt_keeps_terminal_candidate(self):
        trace = solve_ik6(self.control, INITIAL_Q, TARGET_POSITION,
                          pin.rpy.rpyToMatrix(0., 0., np.pi), check=False)
        self.assertFalse(trace['converged'])
        self.assertEqual(len(trace['frames']), 101)
        self.assertEqual(trace['frames'][-1]['iteration'], 100)
        self.assertTrue(trace['frames'][-1]['joints_at_limit'])
        self.assertGreater(trace['frames'][-1]['position_error'], .1)
        self.assertFalse(np.allclose(trace['frames'][-1]['q'], INITIAL_Q))

    def test_diagnostic_continues_twenty_updates_after_first_rank_loss(self):
        trace = solve_ik6(self.control, [.2, 1, -1.1, .2, .4, -.3], TARGET_POSITION,
                          pin.rpy.rpyToMatrix(0., 0., np.pi), max_iterations=1000,
                          check=False, stop_on_limit=True)
        self.assertEqual(trace['frames'][0]['effective_rank'], 6)
        self.assertEqual(trace['stop_reason'], 'post_rank_loss')
        first = trace['first_rank_loss']
        self.assertEqual(trace['frames'][first]['effective_rank'], 5)
        self.assertEqual(trace['frames'][-1]['iteration'], first + 20)
        self.assertLess(len(trace['frames']), 100)
        self.assertFalse(trace['converged'])

    def test_diagnostic_stops_when_configuration_no_longer_changes(self):
        with patch('piper.sim.position.ik6_timelapse.pin.integrate',
                   side_effect=lambda model, q, step: q.copy()):
            trace = solve_ik6(self.control, [.2, 1, -1.1, .2, .4, -.3], TARGET_POSITION,
                              np.eye(3), max_iterations=1000, check=False, stop_on_limit=True)
        self.assertEqual(trace['stop_reason'], 'stagnation')
        self.assertEqual(trace['frames'][-1]['iteration'], 2)
        self.assertEqual(trace['frames'][-1]['step_change'], 0)

    def test_selected_target_really_stalls_without_rank_cutoff(self):
        trace = solve_ik6(self.control, [.2, 1, -1.1, .2, .4, -.3], [.3, .2, .3],
                          np.eye(3), max_iterations=1000, check=False, stop_on_limit=True)
        final = trace['frames'][-1]
        self.assertEqual(trace['stop_reason'], 'stagnation')
        self.assertIsNone(trace['first_rank_loss'])
        self.assertFalse(trace['converged'])
        self.assertLess(final['step_change'], trace['step_tolerance'])
        self.assertGreater(final['log6_error_norm'], .01)
        self.assertEqual(final['joints_at_limit'], [4, 5, 6])
        self.assertEqual(len(final['log6_error']), 6)


if __name__ == '__main__':
    unittest.main()
