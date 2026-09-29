"""Tests for project.py

Numbers in this file are in milliseconds or sample indices:
    800ms - a resting RR interval, 75bpm
    500ms - short interval resultant of a premature beat
    1100ms - long interval resultant of a compensatory pause
    300ms/2000ms - physiological bounds, 200bpm and 30bpm
    fs=1000 - used in synthetic tests for ease of RMSSD computation
"""

import os
import unittest

import project

MITDB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mitdb")
HAS_DATA = os.path.isdir(MITDB)
needs_data = unittest.skipUnless(HAS_DATA, "MIT-BIH data not present in mitdb/")


def peaks_from(intervals, start=0):
    """Build a list of peak indices from a list of RR intervals."""
    peaks = [start]
    for interval in intervals:
        peaks.append(peaks[-1] + interval)
    return peaks


class TestFlag(unittest.TestCase):
    def test_clean_series_all_valid(self):
        # Ensures the filter isn't too aggressive. A clean series of 800ms intervals
        # should be accepted.
        self.assertTrue(all(project.flag([800.0] * 20)))

    def test_rejects_ectopic_pair(self):
        rr = [800.0] * 5 + [500.0, 1100.0] + [800.0] * 5
        valid = project.flag(rr)
        self.assertFalse(valid[5])
        self.assertFalse(valid[6])
        self.assertTrue(all(valid[:5]))

    def test_propagates_one_step_only(self):
        # An ordinary beat following an ectopic pair should be rejected, as it is not
        # a normal-to-normal interval. However, the next beat, if valid, should be accepted.
        rr = [800.0] * 5 + [500.0, 1100.0] + [800.0] * 5
        valid = project.flag(rr)
        self.assertFalse(valid[7])
        self.assertTrue(valid[8])

    def test_does_not_cascade(self):
        # Ensures the above behavior does not cascade.
        rr = [800.0] * 20 + [400.0] + [800.0] * 20
        valid = project.flag(rr)
        self.assertEqual(sum(valid), 39)

    def test_rejects_below_physiological_minimum(self):
        rr = [800.0] * 10 + [250.0] + [800.0] * 10
        self.assertFalse(project.flag(rr)[10])

    def test_rejects_above_physiological_maximum(self):
        rr = [800.0] * 10 + [2500.0] + [800.0] * 10
        self.assertFalse(project.flag(rr)[10])

    def test_tolerates_gradual_drift(self):
        rr = [800.0 + 2 * i for i in range(40)]
        self.assertTrue(all(project.flag(rr)))

    def test_returns_one_flag_per_interval(self):
        rr = [800.0] * 13
        self.assertEqual(len(project.flag(rr)), 13)

    def test_edge_intervals_are_less_robust(self):
        rr = [250.0] + [800.0] * 10
        valid = project.flag(rr)
        self.assertFalse(valid[0])


class TestComputeHrv(unittest.TestCase):
    def test_returns_none_for_too_few_peaks(self):
        for count in (0, 1, 2):
            self.assertIsNone(project.compute_hrv(list(range(count)), 360))

    def test_constant_rhythm_has_zero_variability(self):
        # A constant rhythm should have zero variability, as there are no
        # differences between intervals.
        rmssd, sdnn, excluded, total = project.compute_hrv(peaks_from([800] * 20), 1000)
        self.assertAlmostEqual(rmssd, 0.0, places=12)
        self.assertAlmostEqual(sdnn, 0.0, places=12)
        self.assertEqual(excluded, 0)
        self.assertEqual(total, 20)

    def test_alternating_rhythm_known_rmssd(self):
        # An alternating rhythm of 800ms and 840ms has a known RMSSD of 40ms. With a mean
        # of 820ms and every interval 20ms away, the n-1 denominator gives 20.5195670417ms
        # over 20 intervals for SDNN.
        intervals = [800 if i % 2 == 0 else 840 for i in range(20)]
        rmssd, sdnn, excluded, total = project.compute_hrv(peaks_from(intervals), 1000)
        self.assertAlmostEqual(rmssd, 40.0, places=9)
        self.assertAlmostEqual(sdnn, 20.5195670417, places=8)
        self.assertEqual(excluded, 0)

    def test_counts_exclusions(self):
        # Test that the function correctly counts excluded intervals.
        intervals = [800] * 5 + [500, 1100] + [800] * 5
        _, _, excluded, total = project.compute_hrv(peaks_from(intervals), 1000)
        self.assertEqual(total, 12)
        self.assertEqual(excluded, 3)

    def test_ectopic_beat_does_not_inflate_rmssd(self):
        intervals = [800] * 10 + [500, 1100] + [800] * 10
        rmssd, _, _, _ = project.compute_hrv(peaks_from(intervals), 1000)
        self.assertLess(rmssd, 1.0)

    def test_differences_never_span_a_gap(self):
        # Correct code forms no difference across a series that steps from one constant
        # to another over a single ectopic pair. The resulting RMSSD should be near zero,
        # as all valid differences are zero.
        intervals = [800] * 10 + [1500, 400] + [600] * 10
        rmssd, _, _, _ = project.compute_hrv(peaks_from(intervals), 1000)
        self.assertLess(rmssd, 1.0)

    def test_sampling_rate_scales_result(self):
        intervals = [800 if i % 2 == 0 else 840 for i in range(20)]
        fast = project.compute_hrv(peaks_from(intervals), 1000)
        slow = project.compute_hrv(peaks_from(intervals), 500)
        self.assertAlmostEqual(slow[0], 2 * fast[0], places=9)


class TestValidate(unittest.TestCase):
    def test_perfect_match(self):
        sensitivity, ppv, tp, fp, fn = project.validate(
            [100, 200, 300], [100, 200, 300], 360
        )
        self.assertEqual((tp, fp, fn), (3, 0, 0))
        self.assertEqual((sensitivity, ppv), (1.0, 1.0))

    def test_all_false_positives(self):
        _, _, tp, fp, fn = project.validate([100, 200, 300], [], 360)
        self.assertEqual((tp, fp, fn), (0, 3, 0))

    def test_all_false_negatives(self):
        _, _, tp, fp, fn = project.validate([], [100, 200, 300], 360)
        self.assertEqual((tp, fp, fn), (0, 0, 3))

    def test_empty_inputs_do_not_divide_by_zero(self):
        sensitivity, ppv, tp, fp, fn = project.validate([], [], 360)
        self.assertEqual((sensitivity, ppv, tp, fp, fn), (0.0, 0.0, 0, 0, 0))

    def test_tolerance_boundary(self):
        # At 360 Hz the window is int(0.15 * 360) = 54 samples inclusive.
        self.assertEqual(project.validate([154], [100], 360)[2], 1)
        self.assertEqual(project.validate([155], [100], 360)[2], 0)

    def test_missed_beat_counted_once(self):
        _, _, tp, fp, fn = project.validate([100, 300], [100, 200, 300], 360)
        self.assertEqual((tp, fp, fn), (2, 0, 1))

    def test_extra_beat_counted_once(self):
        _, _, tp, fp, fn = project.validate([100, 200, 300], [100, 300], 360)
        self.assertEqual((tp, fp, fn), (2, 1, 0))


@needs_data
class TestLoad(unittest.TestCase):
    def test_sampling_rate_and_length(self):
        record, fs, _ = project.load("100", 0, 300)
        self.assertEqual(fs, 360)
        self.assertEqual(len(record), 300 * 360)

    def test_selects_mlii_when_channels_are_reversed(self):
        # Record 114 has reversed channels - this test ensures the code selects the correct one.
        record, fs, reference = project.load("114", 0, 300)
        peaks = project.detect(project.clean(record, fs), fs)
        sensitivity, ppv, _, _, _ = project.validate(peaks, reference, fs)
        self.assertGreater(sensitivity, 0.95)
        self.assertGreater(ppv, 0.95)

    def test_falls_back_to_v5_when_no_mlii(self):
        # Records 102 and 104 carry V5 and V2, with no MLII at all.
        for name in ("102", "104"):
            record, fs, reference = project.load(name, 0, 300)
            self.assertEqual(len(record), 300 * 360)
            self.assertGreater(len(reference), 200)

    def test_reference_peaks_are_sorted_and_in_range(self):
        record, fs, reference = project.load("100", 0, 300)
        self.assertEqual(reference, sorted(reference))
        self.assertGreaterEqual(min(reference), 0)
        self.assertLessEqual(max(reference), len(record))

    def test_window_offset_shifts_reference_peaks(self):
        _, _, first = project.load("100", 0, 60)
        _, _, second = project.load("100", 60, 60)
        self.assertGreaterEqual(min(second), 0)
        self.assertNotEqual(first, second)

    def test_skip_annotations_handled(self):
        for name, expected in (("201", 1900), ("232", 1700)):
            _, _, reference = project.load(name, 0, 1800)
            self.assertGreater(len(reference), expected)

    def test_start_beyond_record_raises(self):
        with self.assertRaises(ValueError):
            project.load("100", 99999, 60)

    def test_window_beyond_record_raises(self):
        with self.assertRaises(ValueError):
            project.load("100", 0, 99999)

    def test_signal_is_in_millivolts(self):
        record, _, _ = project.load("100", 0, 300)
        self.assertLess(max(abs(v) for v in record), 10.0)
        self.assertLess(abs(sum(record) / len(record)), 1.0)


@needs_data
class TestEndToEnd(unittest.TestCase):
    def test_clean_record_detected_accurately(self):
        record, fs, reference = project.load("100", 0, 300)
        peaks = project.detect(project.clean(record, fs), fs)
        sensitivity, ppv, _, _, _ = project.validate(peaks, reference, fs)
        self.assertGreater(sensitivity, 0.99)
        self.assertGreater(ppv, 0.99)

    def test_hrv_within_expected_range(self):
        # The 15-40ms window is wide enough to survive variation between records, but tight
        # enough to catch gross errors.
        record, fs, _ = project.load("100", 0, 300)
        peaks = project.detect(project.clean(record, fs), fs)
        rmssd, sdnn, excluded, total = project.compute_hrv(peaks, fs)
        self.assertTrue(15.0 < rmssd < 40.0)
        self.assertTrue(15.0 < sdnn < 40.0)
        self.assertLess(excluded / total, 0.10)

    def test_detects_plausible_heart_rate(self):
        record, fs, _ = project.load("100", 0, 300)
        peaks = project.detect(project.clean(record, fs), fs)
        bpm = len(peaks) / 300 * 60
        self.assertTrue(50 < bpm < 100)

    def test_peaks_are_sorted_and_respect_refractory_period(self):
        record, fs, _ = project.load("100", 0, 300)
        peaks = project.detect(project.clean(record, fs), fs)
        self.assertEqual(peaks, sorted(peaks))
        gaps = [peaks[i] - peaks[i - 1] for i in range(1, len(peaks))]
        self.assertGreaterEqual(min(gaps), int(0.20 * fs))

    def test_arrhythmic_record_reports_high_exclusion(self):
        record, fs, _ = project.load("208", 0, 300)
        peaks = project.detect(project.clean(record, fs), fs)
        _, _, excluded, total = project.compute_hrv(peaks, fs)
        self.assertGreater(excluded / total, 0.30)

    def test_detect_handles_short_input(self):
        self.assertEqual(project.detect([0.0] * 3, 360), [])

    def test_detect_handles_flat_signal(self):
        peaks = project.detect([0.0] * 3600, 360)
        self.assertIsInstance(peaks, list)


if __name__ == "__main__":
    unittest.main()
