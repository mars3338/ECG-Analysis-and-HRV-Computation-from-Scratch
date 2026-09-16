"""Tests for signals.py."""

import unittest
import signals

FS = 360  # 360 Hz is the MIT-BIH sampling rate

# Reference scipy coefficients
HIGHPASS_5HZ = (
    0.9401569638962209,
    -1.8803139277924419,
    0.9401569638962209,
    -1.8767295268397763,
    0.8838983287451074,
)

LOWPASS_15HZ = (
    0.014401440346511196,
    0.028802880693022392,
    0.014401440346511196,
    -1.6329931618554523,
    0.6905989232414971,
)


class TestCoefficients(unittest.TestCase):
    def test_highpass_matches_reference(self):
        # delta=1e-12 rather than exact equality; the design path runs through
        # tan, complex exp, and several divisions, which accumulate rounding error.
        # 1e-12 ensures more than enough precision.
        got = signals.calculate_filter_coefficients(5, FS, "high")
        for actual, expected in zip(got, HIGHPASS_5HZ):
            self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_lowpass_matches_reference(self):
        got = signals.calculate_filter_coefficients(15, FS, "low")
        for actual, expected in zip(got, LOWPASS_15HZ):
            self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            signals.calculate_filter_coefficients(5, FS, "band")

    def test_returns_five_coefficients(self):
        self.assertEqual(len(signals.calculate_filter_coefficients(5, FS, "high")), 5)

    def test_numerator_is_symmetric(self):
        for fc, mode in ((5, "high"), (15, "low")):
            b0, b1, b2, _, _ = signals.calculate_filter_coefficients(fc, FS, mode)
            self.assertAlmostEqual(b0, b2, places=15)

    def test_highpass_numerator_signs(self):
        # (1 - z^-1)^2 expands to 1 - 2z^-1 + z^-2, so b1 must be exactly -2*b0.
        b0, b1, b2, _, _ = signals.calculate_filter_coefficients(5, FS, "high")
        self.assertAlmostEqual(b1, -2 * b0, places=15)

    def test_lowpass_numerator_signs(self):
        # (1 + z^-1)^2 expands to 1 + 2z^-1 + z^-2, so b1 must be exactly 2*b0.
        b0, b1, b2, _, _ = signals.calculate_filter_coefficients(15, FS, "low")
        self.assertAlmostEqual(b1, 2 * b0, places=15)

    def test_shared_denominator(self):
        # Both modes use the same denominator, so a1 and a2 must be identical.
        _, _, _, hp_a1, hp_a2 = signals.calculate_filter_coefficients(15, FS, "high")
        _, _, _, lp_a1, lp_a2 = signals.calculate_filter_coefficients(15, FS, "low")
        self.assertAlmostEqual(hp_a1, lp_a1, places=12)
        self.assertAlmostEqual(hp_a2, lp_a2, places=12)

    def test_stable_across_sampling_rates(self):
        for fs in (250, 360, 500, 1000):
            for fc, mode in ((5, "high"), (15, "low"), (40, "low")):
                _, _, _, a1, a2 = signals.calculate_filter_coefficients(fc, fs, mode)
                self.assertLess(abs(a2), 1.0)
                self.assertLess(abs(a1), 1 + a2)


class TestFilterBehaviour(unittest.TestCase):
    def setUp(self):
        self.highpass = signals.calculate_filter_coefficients(5, FS, "high")
        self.lowpass = signals.calculate_filter_coefficients(15, FS, "low")

    def test_highpass_removes_dc(self):
        # A constant signal is pure DC - zero frequency. As a highpass filter has
        # both zeros there, the output should be zero. 1e-10 is far looser than the
        # the measured 1.7e-14, leaving room for the transient at the edges of the
        # signal to decay differently at other cutoffs.
        constant = [1.0] * 2000
        out = signals.filtfilt(constant, *self.highpass)
        self.assertLess(max(abs(v) for v in out[500:1500]), 1e-10)

    def test_lowpass_passes_dc(self):
        # The inverse: as a lowpass filter has zeros at Nyquist, a constant signal
        # should pass through at gain 1.
        constant = [1.0] * 2000
        out = signals.filtfilt(constant, *self.lowpass)
        self.assertLess(max(abs(v - 1.0) for v in out[500:1500]), 1e-10)

    def test_edge_transient_exists(self):
        constant = [1.0] * 2000
        out = signals.filtfilt(constant, *self.highpass)
        self.assertGreater(max(abs(v) for v in out[:20]), 1e-3)

    def test_zero_phase(self):
        impulse = [0.0] * 1201
        impulse[600] = 1.0
        out = signals.filtfilt(impulse, *self.lowpass)
        for k in range(1, 400):
            self.assertAlmostEqual(out[600 + k], out[600 - k], delta=1e-12)

    def test_linearity(self):
        # filter(a + b) must equal filter(a) + filter(b).
        a = [float(i % 7) for i in range(500)]
        b = [float((i * 3) % 11) for i in range(500)]
        ya = signals.apply_filter(a, *self.lowpass)
        yb = signals.apply_filter(b, *self.lowpass)
        yab = signals.apply_filter([a[i] + b[i] for i in range(500)], *self.lowpass)
        for i in range(500):
            self.assertAlmostEqual(yab[i], ya[i] + yb[i], delta=1e-10)

    def test_output_length_preserved(self):
        signal = [0.5] * 137
        self.assertEqual(len(signals.apply_filter(signal, *self.lowpass)), 137)
        self.assertEqual(len(signals.filtfilt(signal, *self.lowpass)), 137)

    def test_empty_signal(self):
        self.assertEqual(signals.apply_filter([], *self.lowpass), [])
        self.assertEqual(signals.filtfilt([], *self.lowpass), [])

    def test_zero_input_gives_zero_output(self):
        out = signals.filtfilt([0.0] * 100, *self.highpass)
        self.assertTrue(all(v == 0.0 for v in out))

    def test_lowpass_attenuates_high_frequency(self):
        import math_utilities as mu

        def tone(freq, n=2000):
            return [mu.sin(2 * mu.pi * freq * i / FS) for i in range(n)]

        slow = signals.filtfilt(tone(5), *self.lowpass)
        fast = signals.filtfilt(tone(60), *self.lowpass)
        slow_amp = max(abs(v) for v in slow[500:1500])
        fast_amp = max(abs(v) for v in fast[500:1500])
        self.assertGreater(slow_amp, 0.5)
        self.assertLess(fast_amp, 0.05)

    def test_highpass_attenuates_baseline_wander(self):
        import math_utilities as mu

        drift = [mu.sin(2 * mu.pi * 0.5 * i / FS) for i in range(4000)]
        out = signals.filtfilt(drift, *self.highpass)
        self.assertLess(max(abs(v) for v in out[1000:3000]), 0.05)


if __name__ == "__main__":
    unittest.main()
