"""Tests for fair transform support and representation-derived descriptors."""

import unittest

import numpy as np

from tfpq_qualifier.models import SignalRecord
from tfpq_qualifier.representation_features import extract_representation_features
from tfpq_qualifier.synthetic import SyntheticDataGenerator, load_conditions
from tfpq_qualifier.transforms import transform


class RepresentationFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generator = SyntheticDataGenerator()
        conditions = load_conditions("manifests/development_conditions.csv")
        condition = next(
            item for item in conditions if item.stratum == "oscillatory_transient"
        )
        generated = generator.generate(condition, realization=0)
        cls.record = SignalRecord(
            samples=generated.samples_pu,
            time=generated.time_s,
            sample_rate=generated.sample_rate_hz,
            disturbance=generated.disturbance,
            properties=dict(generated.truth),
            conditions={"fundamental_frequency": 50.0},
            partition=generated.partition,
            seed=generated.seed,
        )

    def test_common_frequency_support_and_finite_features(self) -> None:
        for method in ("stft", "wavelet", "s_transform", "vmd"):
            representation = transform(
                self.record, method, {"frequency_min": 10, "frequency_max": 3000}
            )
            self.assertTrue(np.all(representation.frequencies >= 10))
            self.assertTrue(np.all(representation.frequencies <= 3000))
            features = extract_representation_features(representation)
            self.assertTrue(np.isfinite(features["tf_energy"]))
            self.assertTrue(np.isfinite(features["fundamental_deviation_peak"]))

    def test_transform_variants_are_active(self) -> None:
        hann = transform(self.record, "stft", {"window_name": "hann"})
        blackman = transform(self.record, "stft", {"window_name": "blackman"})
        self.assertFalse(np.array_equal(hann.values, blackman.values))
        narrow = transform(self.record, "s_transform", {"gaussian_width_factor": 0.5})
        wide = transform(self.record, "s_transform", {"gaussian_width_factor": 2.0})
        self.assertFalse(np.array_equal(narrow.values, wide.values))

    def test_invalid_transform_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            transform(self.record, "unsupported")


if __name__ == "__main__":
    unittest.main()
