"""Tests for math_utilities.

Whilst the project itself avoids any libraries, test programs make use of
the standard library for comparison.

Two tolerances appear throughout:
    places=15  -- absolute, |a - b| < 0.5e-15. Used for comparing ideally exact
    values with reasonable precision. A tighter tolerance may fail
    on rounding rather than defects.
    delta=1e-14 -- absolute, |a - b| < 1e-14. Used for bulk random comparison,
    wherein a few functions compose and rounding accumulates.
A double has about 15-16 significant decimal digits, so 1e-15 is roughly one
unit in the last place for values near 1.

Every random test seeds the generator. Without a seed a failure would be
unreproducible -- rerunning would draw different numbers and might pass.
"""

import math
import random
import unittest
import math_utilities as mu


class TestConstants(unittest.TestCase):
    def test_pi(self):
        self.assertAlmostEqual(mu.pi, math.pi, places=15)

    def test_tau(self):
        self.assertAlmostEqual(mu.tau, 2 * math.pi, places=15)

    def test_e(self):
        self.assertAlmostEqual(mu.e, math.e, places=15)


class TestSin(unittest.TestCase):
    def test_known(self):
        # Quarter-turn landmarks. Points for which sin is exactly 0, 1, or -1,
        # so any error in the Taylor series or sign handling of range reduction
        # is easily legible.
        self.assertAlmostEqual(mu.sin(0), 0.0, places=15)
        self.assertAlmostEqual(mu.sin(mu.pi / 2), 1.0, places=15)
        self.assertAlmostEqual(mu.sin(mu.pi), 0.0, places=15)
        self.assertAlmostEqual(mu.sin(-mu.pi / 2), -1.0, places=15)

    def test_odd_symmetry(self):
        # sin(-x) == -sin(x) must hold true for every x. Values are chosen from
        # across quadrants to ensure all branches of the range reduction are tested.
        for x in (0.3, 1.0, 2.5, 4.7):
            self.assertAlmostEqual(mu.sin(-x), -mu.sin(x), places=15)

    def test_stdlib_comparison(self):
        # -10 to 10 covers roughly three full periods either side of zero, ensuring
        # the reduction runs but stays within its well-conditioned range. 5000
        # samples is enough to hit every branch several times relatively quickly.
        random.seed(0)
        for _ in range(5000):
            x = random.uniform(-10, 10)
            self.assertAlmostEqual(mu.sin(x), math.sin(x), delta=1e-14)

    def test_range_reduction(self):
        # Precision degrades as the argument grows. Folding x into [-pi, pi] subtracts
        # two large numbers and loses significant digits. The looser tolerances below
        # are measured limits to document the cost of that method. The accuracy
        # is recorded where it sits - tightening either bound will result in
        # failure.
        random.seed(1)
        for _ in range(1000):
            x = random.uniform(-1e3, 1e3)
            self.assertAlmostEqual(mu.sin(x), math.sin(x), delta=1e-12)
        for _ in range(1000):
            x = random.uniform(-1e6, 1e6)
            self.assertAlmostEqual(mu.sin(x), math.sin(x), delta=1e-9)


class TestCos(unittest.TestCase):
    def test_known(self):
        self.assertAlmostEqual(mu.cos(0), 1.0, places=15)
        self.assertAlmostEqual(mu.cos(mu.pi / 2), 0.0, places=15)
        self.assertAlmostEqual(mu.cos(mu.pi), -1.0, places=15)
        self.assertAlmostEqual(mu.cos(-mu.pi / 2), 0.0, places=15)

    def test_even_symmetry(self):
        for x in (0.3, 1.0, 2.5, 4.7):
            self.assertAlmostEqual(mu.cos(-x), mu.cos(x), places=15)

    def test_stdlib_comparison(self):
        random.seed(2)
        for _ in range(5000):
            x = random.uniform(-10, 10)
            self.assertAlmostEqual(mu.cos(x), math.cos(x), delta=1e-14)

    def test_range_reduction(self):
        random.seed(3)
        for _ in range(1000):
            x = random.uniform(-1e3, 1e3)
            self.assertAlmostEqual(mu.cos(x), math.cos(x), delta=1e-12)
        for _ in range(1000):
            x = random.uniform(-1e6, 1e6)
            self.assertAlmostEqual(mu.cos(x), math.cos(x), delta=1e-9)

    def test_sin_cos_identity(self):
        # sin^2 + cos^2 == 1. Checking the two functions against each other rather
        # than against the stdlib reveals shared errors in range reduction. Tolerance
        # is 1e-14 as squaring roughly doubles the relative error.
        random.seed(2)
        for _ in range(2000):
            x = random.uniform(-20, 20)
            self.assertAlmostEqual(mu.sin(x) ** 2 + mu.cos(x) ** 2, 1.0, delta=1e-14)


class TestTan(unittest.TestCase):
    def test_known(self):
        # Known values. pi/4 is where sin and cos are equal, meaning tan is exactly 1.
        self.assertAlmostEqual(mu.tan(0), 0.0, places=15)
        self.assertAlmostEqual(mu.tan(mu.pi / 4), 1.0, places=15)
        self.assertAlmostEqual(mu.tan(-mu.pi / 4), -1.0, places=15)

    def test_stdlib_comparison(self):
        # Deliberately confined to [-1, 1], keeping well clear of the poles at
        # +/-pi/2 (about 1.571).
        random.seed(4)
        for _ in range(2000):
            x = random.uniform(-1.0, 1.0)
            self.assertAlmostEqual(mu.tan(x), math.tan(x), delta=1e-14)

    def test_prewarp(self):
        # Tests prewarp with several sampling rates and cutoff frequencies,
        # near the values used in the project.
        for fs in (250, 360, 500):
            for fc in (5, 15, 40):
                x = mu.pi * fc / fs
                self.assertAlmostEqual(mu.tan(x), math.tan(x), delta=1e-14)


class TestArctan(unittest.TestCase):
    def test_known(self):
        self.assertAlmostEqual(mu.arctan(0), 0.0, places=15)
        self.assertAlmostEqual(mu.arctan(1), mu.pi / 4, places=15)
        self.assertAlmostEqual(mu.arctan(-1), -mu.pi / 4, places=15)

    def test_argument_reduction_branches(self):
        # Each value here tests a different branch of the range reduction,
        # alongside a repeated negative to check sign handling.
        for x in (0.1, 0.7, 1.5, -0.1, -0.7, -1.5, 100.0, -100.0):
            self.assertAlmostEqual(mu.arctan(x), math.atan(x), delta=1e-14)


class TestArcsinArccos(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(mu.arcsin(-1), -mu.pi / 2, places=15)
        self.assertAlmostEqual(mu.arcsin(1), mu.pi / 2, places=15)
        self.assertAlmostEqual(mu.arccos(-1), mu.pi, places=15)
        self.assertAlmostEqual(mu.arccos(1), 0.0, places=15)

    def test_stdlib_comparison(self):
        random.seed(7)
        for _ in range(5000):
            x = random.uniform(-0.99, 0.99)
            self.assertAlmostEqual(mu.arcsin(x), math.asin(x), delta=1e-14)
            self.assertAlmostEqual(mu.arccos(x), math.acos(x), delta=1e-14)

    def test_domain_errors(self):
        for x in (1.5, -1.5, 2.0):
            with self.assertRaises(ValueError):
                mu.arcsin(x)
            with self.assertRaises(ValueError):
                mu.arccos(x)


class TestExp(unittest.TestCase):
    def test_known(self):
        # places=14 rather than 15 as range reduction roughly doubles errors
        # for x = 1.
        self.assertAlmostEqual(mu.exp(0), 1.0, places=15)
        self.assertAlmostEqual(mu.exp(1), math.e, places=14)

    def test_reciprocal(self):
        for x in (0.5, 1.0, 5.0, 12.0):
            self.assertAlmostEqual(mu.exp(-x) * mu.exp(x), 1.0, delta=1e-12)

    def test_stdlib_comparison(self):
        # Relative error, not absolute. Over -30 to 30 the result spans e^-30
        # (about 1e-13) to e^30 (about 1e13) -- 26 orders of magnitude. An
        # absolute tolerance meaningful at the top would be uselessly loose at
        # the bottom, and vice versa. Dividing by the expected value checks
        # how many significant digits are correct.
        random.seed(7)
        for _ in range(5000):
            x = random.uniform(-30, 30)
            expected = math.exp(x)
            self.assertLess(abs(mu.exp(x) - expected) / expected, 1e-12)

    def test_loop(self):
        for x in (0.4, 0.6, 3.0, 20.0, -20.0):
            self.assertLess(abs(mu.exp(x) - math.exp(x)) / math.exp(x), 1e-12)


class TestDifferentiate(unittest.TestCase):
    def test_constant_signal_has_zero_derivative(self):
        # As the weights sum to 0, a constant input is expected to cancel to
        # exactly 0.0, with no rounding error.
        self.assertTrue(all(v == 0.0 for v in mu.differentiate([5.0] * 10)))

    def test_preserves_length(self):
        # 37 is arbitrary and deliberately not round - a length that is not a
        # multiple of the 5-sample stencil would expose an off-by-one in the
        # edge clamping that a length like 40 might hide.
        self.assertEqual(len(mu.differentiate([1.0] * 37)), 37)

    def test_linear_ramp_has_constant_derivative(self):
        ramp = [float(i) for i in range(50)]
        middle = mu.differentiate(ramp)[10:40]
        for value in middle:
            self.assertAlmostEqual(value, middle[0], places=12)

    def test_sign_follows_slope(self):
        rising = [float(i) for i in range(20)]
        falling = [float(-i) for i in range(20)]
        self.assertGreater(mu.differentiate(rising)[10], 0)
        self.assertLess(mu.differentiate(falling)[10], 0)


class TestIntegrate(unittest.TestCase):
    def test_running_window_sum(self):
        self.assertEqual(mu.integrate([1.0] * 10, 3), [1, 2, 3, 3, 3, 3, 3, 3, 3, 3])

    def test_preserves_length(self):
        self.assertEqual(len(mu.integrate([1.0] * 37, 5)), 37)

    def test_zeros_stay_zero(self):
        self.assertTrue(all(v == 0.0 for v in mu.integrate([0.0] * 20, 4)))

    def test_window_forgets_old_samples(self):
        signal = [1.0] * 5 + [0.0] * 10
        out = mu.integrate(signal, 3)
        self.assertAlmostEqual(out[-1], 0.0, places=12)


class TestMedian(unittest.TestCase):
    def test_odd_length(self):
        self.assertEqual(mu.median([3, 1, 2]), 2)

    def test_even_length_averages_middle_pair(self):
        self.assertEqual(mu.median([4, 1, 3, 2]), 2.5)

    def test_single_value(self):
        self.assertEqual(mu.median([7]), 7)

    def test_does_not_mutate_input(self):
        values = [3, 1, 2]
        mu.median(values)
        self.assertEqual(values, [3, 1, 2])

    def test_ignores_outlier(self):
        self.assertEqual(mu.median([800, 800, 800, 5000]), 800)


if __name__ == "__main__":
    unittest.main()
