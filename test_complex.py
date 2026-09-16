"""Tests for complex.py."""

import cmath
import math
import random
import unittest

from complex import Complex


def to_builtin(z):
    """Convert this project's Complex into Python's built-in complex."""
    return complex(z.real, z.imag)


def assert_close(case, z, expected, delta=1e-12):
    """Compares a Complex against a built-in complex, part by part.

    delta=1e-12 is chosen as most callers chain several operations before
    comparing. The looser bound absorbs accumulated rounding error without
    masking defects.
    """
    case.assertAlmostEqual(z.real, expected.real, delta=delta)
    case.assertAlmostEqual(z.imag, expected.imag, delta=delta)


class TestConstruction(unittest.TestCase):
    def test_stores_parts(self):
        z = Complex(3, -4)
        self.assertEqual(z.real, 3)
        self.assertEqual(z.imag, -4)

    def test_repr_positive_imaginary(self):
        self.assertEqual(repr(Complex(3, 4)), "(3 + 4i)")

    def test_repr_negative_imaginary(self):
        self.assertEqual(repr(Complex(3, -4)), "(3 - 4i)")

    def test_repr_trims_integral_floats(self):
        self.assertEqual(repr(Complex(3.0, 4.0)), "(3 + 4i)")


class TestArithmetic(unittest.TestCase):
    def setUp(self):
        random.seed(11)
        self.pairs = []
        for _ in range(200):
            a = Complex(random.uniform(-10, 10), random.uniform(-10, 10))
            b = Complex(random.uniform(-10, 10), random.uniform(-10, 10))
            self.pairs.append((a, b))

    def test_addition_matches_builtin(self):
        for a, b in self.pairs:
            assert_close(self, a + b, to_builtin(a) + to_builtin(b))

    def test_subtraction_matches_builtin(self):
        for a, b in self.pairs:
            assert_close(self, a - b, to_builtin(a) - to_builtin(b))

    def test_multiplication_matches_builtin(self):
        for a, b in self.pairs:
            assert_close(self, a * b, to_builtin(a) * to_builtin(b))

    def test_division_matches_builtin(self):
        for a, b in self.pairs:
            assert_close(self, a / b, to_builtin(a) / to_builtin(b))

    def test_multiplication_commutes(self):
        for a, b in self.pairs:
            assert_close(self, a * b, to_builtin(b * a))

    def test_division_inverts_multiplication(self):
        # (a * b) / b == a checks the two operations against each other. delta is
        # loosened to 1e-10 due to the extra operations.
        for a, b in self.pairs:
            assert_close(self, (a * b) / b, to_builtin(a), delta=1e-10)

    def test_distributive(self):
        for a, b in self.pairs[:50]:
            c = Complex(2.5, -1.5)
            assert_close(self, a * (b + c), to_builtin(a * b + a * c), delta=1e-11)


class TestScalarOperands(unittest.TestCase):
    def test_add_scalar(self):
        assert_close(self, Complex(1, 2) + 3, complex(4, 2))
        assert_close(self, 3 + Complex(1, 2), complex(4, 2))

    def test_subtract_scalar(self):
        assert_close(self, Complex(1, 2) - 3, complex(-2, 2))
        assert_close(self, 3 - Complex(1, 2), complex(2, -2))

    def test_multiply_scalar(self):
        assert_close(self, Complex(1, 2) * 3, complex(3, 6))
        assert_close(self, 3 * Complex(1, 2), complex(3, 6))

    def test_divide_scalar(self):
        assert_close(self, Complex(2, 4) / 2, complex(1, 2))
        assert_close(self, 2 / Complex(1, 1), complex(1, -1))

    def test_reverse_ops_used_by_signals(self):
        # Test the specifc expressions used in signals.py.
        z = Complex(0, 1)
        assert_close(self, 5.0 / z, complex(0, -5))
        assert_close(self, 1 - Complex(0.5, 0.5), complex(0.5, -0.5))


class TestDivisionByZero(unittest.TestCase):
    def test_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Complex(1, 1) / Complex(0, 0)

    def test_raises_for_scalar_zero(self):
        with self.assertRaises(ZeroDivisionError):
            Complex(1, 1) / 0


class TestMagnitudeAndPhase(unittest.TestCase):
    def test_magnitude_known(self):
        self.assertAlmostEqual(Complex(3, 4).magnitude(), 5.0, places=15)
        self.assertAlmostEqual(Complex(0, 0).magnitude(), 0.0, places=15)

    def test_magnitude_matches_builtin(self):
        random.seed(12)
        for _ in range(500):
            z = Complex(random.uniform(-9, 9), random.uniform(-9, 9))
            self.assertAlmostEqual(z.magnitude(), abs(to_builtin(z)), delta=1e-13)

    def test_phase_on_axes(self):
        self.assertAlmostEqual(Complex(1, 0).phase(), 0.0, places=15)
        self.assertAlmostEqual(Complex(0, 1).phase(), math.pi / 2, places=15)
        self.assertAlmostEqual(Complex(0, -1).phase(), -math.pi / 2, places=15)
        self.assertAlmostEqual(Complex(0, 0).phase(), 0.0, places=15)

    def test_phase_all_four_quadrants(self):
        for re, im in ((1, 1), (-1, 1), (-1, -1), (1, -1), (-2, 0.5), (-2, -0.5)):
            z = Complex(re, im)
            self.assertAlmostEqual(z.phase(), cmath.phase(to_builtin(z)), delta=1e-13)


class TestExp(unittest.TestCase):
    def test_exp_of_zero(self):
        assert_close(self, Complex(0, 0).exp(), complex(1, 0))

    def test_eulers_identity(self):
        # exp(i*pi) == -1
        z = Complex(0, math.pi).exp()
        self.assertAlmostEqual(z.real, -1.0, delta=1e-14)
        self.assertAlmostEqual(z.imag, 0.0, delta=1e-14)

    def test_matches_builtin(self):
        random.seed(13)
        for _ in range(500):
            z = Complex(random.uniform(-5, 5), random.uniform(-8, 8))
            assert_close(self, z.exp(), cmath.exp(to_builtin(z)), delta=1e-11)

    def test_butterworth_pole_angles(self):
        # Test the angles signals.py actually evaluates (those for a second-order Butterworth filter).
        for angle in (3 * math.pi / 4, 5 * math.pi / 4):
            assert_close(
                self, Complex(0, angle).exp(), cmath.exp(complex(0, angle)), delta=1e-14
            )

    def test_magnitude_of_imaginary_exp_is_one(self):
        random.seed(14)
        for _ in range(500):
            angle = random.uniform(-20, 20)
            self.assertAlmostEqual(
                Complex(0, angle).exp().magnitude(), 1.0, delta=1e-13
            )


if __name__ == "__main__":
    unittest.main()
