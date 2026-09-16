"""Tests for fair transform support and representation-derived descriptors."""

import unittest

import numpy as np

import tfpq_qualifier as tq


class RepresentationFeatureTests(unittest.TestCase):
    def test_common_frequency_support_and_finite_features(self) -> None:
        record = tq.generate_signal(
            tq.SignalSpec(disturbance="oscillatory_transient", sample_rate=12800)
        )
        for method in ("stft", "wavelet", "s_transform", "vmd"):
            representation = tq.transform(
                record, method, {"frequency_min": 10, "frequency_max": 3000}
            )
            self.assertTrue(np.all(representation.frequencies >= 10))
            self.assertTrue(np.all(representation.frequencies <= 3000))
            features = tq.extract_representation_features(representation)
            self.assertTrue(np.isfinite(features["tf_energy"]))
            self.assertTrue(np.isfinite(features["fundamental_deviation_peak"]))

    def test_transform_variants_are_active(self) -> None:
        record = tq.generate_signal(tq.SignalSpec(sample_rate=3200))
        hann = tq.transform(record, "stft", {"window_name": "hann"})
        blackman = tq.transform(record, "stft", {"window_name": "blackman"})
        self.assertFalse(np.array_equal(hann.values, blackman.values))
        narrow = tq.transform(record, "s_transform", {"gaussian_width_factor": 0.5})
        wide = tq.transform(record, "s_transform", {"gaussian_width_factor": 2.0})
        self.assertFalse(np.array_equal(narrow.values, wide.values))


if __name__ == "__main__":
    unittest.main()
