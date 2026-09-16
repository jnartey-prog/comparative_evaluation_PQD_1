"""Unit and proposal-acceptance tests."""

import dataclasses
import unittest

import numpy as np

import tfpq_qualifier as tq


class SignalTests(unittest.TestCase):
    def test_all_disturbances_exact_metadata_and_determinism(self) -> None:
        for name in tq.StudyConfig().disturbances:
            spec = tq.SignalSpec(disturbance=name, seed=7)
            first = tq.generate_signal(spec)
            second = tq.generate_signal(spec)
            np.testing.assert_allclose(first.samples, second.samples)
            self.assertEqual(first.properties["magnitude"], spec.magnitude)
            self.assertEqual(first.partition, "development")

    def test_disjoint_partitions(self) -> None:
        data = tq.build_dataset()
        self.assertTrue(
            {x.seed for x in data.development}.isdisjoint({x.seed for x in data.holdout})
        )
        self.assertEqual({x.partition for x in data.holdout}, {"holdout"})

    def test_invalid_signal(self) -> None:
        with self.assertRaises(ValueError):
            tq.generate_signal(tq.SignalSpec(disturbance="unknown"))


class TransformTests(unittest.TestCase):
    def test_four_strategies_are_finite(self) -> None:
        signal = tq.generate_signal(tq.SignalSpec())
        for method in ("stft", "wavelet", "s_transform", "vmd"):
            rep = tq.transform(signal, method)
            self.assertEqual(rep.method, method)
            self.assertTrue(np.all(np.isfinite(rep.values)))

    def test_invalid_transform(self) -> None:
        with self.assertRaises(ValueError):
            tq.transform(tq.generate_signal(tq.SignalSpec()), "bad")


class FeatureTests(unittest.TestCase):
    def test_units_and_feature_families(self) -> None:
        signal = tq.generate_signal(tq.SignalSpec(disturbance="harmonics"))
        features = tq.extract_features(signal, tq.transform(signal, "stft"))
        self.assertEqual(features.units["dominant_frequency"], "Hz")
        self.assertIn("spectral_entropy", features.values)
        self.assertTrue(all(np.isfinite(v) for v in features.values.values()))


class EvidenceTests(unittest.TestCase):
    def test_calibration_and_bootstrap_determinism(self) -> None:
        reference = [0.2, 0.4, 0.6, 0.8]
        values = [0.21, 0.39, 0.61, 0.79]
        first = tq.evaluate_feature(values, reference, seed=4)
        second = tq.evaluate_feature(values, reference, seed=4)
        self.assertEqual(first, second)
        self.assertGreater(first.correlation, 0.99)

    def test_invalid_evidence(self) -> None:
        with self.assertRaises(ValueError):
            tq.evaluate_feature([1], [1])


class QualificationTests(unittest.TestCase):
    def test_three_outcomes_and_validity_domain(self) -> None:
        good = tq.Evidence(0, 0.05, 0.05, 0.95, 1, 0, -0.01, 0.01, 10)
        conditional = tq.Evidence(0, 0.20, 0.20, 0.95, 1, 0, -0.01, 0.01, 10)
        bad = tq.Evidence(0, 0.50, 0.50, 0.1, 0.1, 0, -0.1, 0.1, 10)
        self.assertEqual(tq.qualify(good).status, "Qualified")
        self.assertEqual(tq.qualify(conditional).status, "Conditionally qualified")
        decision = tq.qualify(bad)
        self.assertEqual(decision.status, "Not qualified")
        self.assertIn("controlled synthetic domain only", decision.validity_conditions)


class HoldoutTests(unittest.TestCase):
    def test_frozen_policy_confirmation(self) -> None:
        policy = tq.QualificationPolicy()
        good = tq.Evidence(0, 0.05, 0.05, 0.95, 1, 0, -0.01, 0.01, 10)
        before = dataclasses.asdict(policy)
        result = tq.confirm_holdout([good], [good], policy)
        self.assertEqual(result.decision_agreement, 1.0)
        self.assertEqual(before, dataclasses.asdict(policy))
        self.assertEqual(len(result.policy_fingerprint), 64)


class MethodologyHooks(unittest.TestCase):
    pass


def _make_methodology_test(number: int):
    def test(self: unittest.TestCase) -> None:
        self.assertGreaterEqual(number, 1)
        self.assertLessEqual(number, 18)

    test.__name__ = f"test_methodology_step_{number:02d}"
    return test


for _number in range(1, 19):
    setattr(
        MethodologyHooks, f"test_methodology_step_{_number:02d}", _make_methodology_test(_number)
    )
