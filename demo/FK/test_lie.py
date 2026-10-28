import unittest

import numpy as np

from lie import exp_se3, hat, inverse_transform, rotation_axis, twist_to_axis, world_to_twist


class LieTest(unittest.TestCase):
    def test_hat_point_velocity(self):
        w, p, v = [0, 0, 2], [1, 0, 0], [0, 3, 0]
        xi = world_to_twist(w, v, p)
        np.testing.assert_allclose(xi, [0, 0, 2, 0, 1, 0])
        np.testing.assert_allclose(hat(xi) @ [*p, 1], [*v, 0])

    def test_same_motion_from_different_points(self):
        w, vs = np.array([1., 2., 3.]), np.array([4., 5., 6.])
        for p in (np.array([1., 0., 0.]), np.array([2., -1., 3.])):
            v = vs + np.cross(w, p)
            np.testing.assert_allclose(world_to_twist(w, v, p), np.r_[w, vs])

    def test_offset_rotation_axis(self):
        S = rotation_axis([0, 0, 5], [1, 0, 7])
        np.testing.assert_allclose(S, [0, 0, 1, 0, -1, 0])
        T = exp_se3(S * (np.pi / 2))
        np.testing.assert_allclose(T @ [1, 0, 7, 1], [1, 0, 7, 1], atol=1e-12)
        np.testing.assert_allclose(T @ [2, 0, 0, 1], [1, 1, 0, 1], atol=1e-12)
        np.testing.assert_allclose(T @ inverse_transform(T), np.eye(4), atol=1e-12)
        direction, point, pitch = twist_to_axis(2 * S)
        np.testing.assert_allclose(direction, [0, 0, 1])
        np.testing.assert_allclose(point, [1, 0, 0])
        self.assertEqual(pitch, 0)

    def test_screw_pitch_and_translation(self):
        xi = world_to_twist([0, 0, 2], [0, 0, 6], [1, 0, 0])
        direction, point, pitch = twist_to_axis(xi)
        np.testing.assert_allclose(direction, [0, 0, 1])
        np.testing.assert_allclose(point, [1, 0, 0])
        self.assertEqual(pitch, 3)
        pure_translation = [0, 0, 0, 1, 2, 3]
        np.testing.assert_allclose(exp_se3(pure_translation)[:3, 3], [1, 2, 3])
        direction, point, pitch = twist_to_axis(pure_translation)
        np.testing.assert_allclose(direction, np.array([1, 2, 3]) / np.sqrt(14))
        self.assertIsNone(point)
        self.assertEqual(pitch, np.inf)

    def test_zero_motion(self):
        np.testing.assert_allclose(exp_se3(np.zeros(6)), np.eye(4))
        with self.assertRaises(ValueError):
            rotation_axis([0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            twist_to_axis(np.zeros(6))


if __name__ == '__main__':
    unittest.main()
