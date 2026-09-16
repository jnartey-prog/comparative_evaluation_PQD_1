"""Scientific and persistence tests for manifest-driven generation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tfpq_qualifier.synthetic import (
    COMPOUND_COMPONENTS,
    SyntheticDataGenerator,
    load_conditions,
    save_condition_batch,
)


class SyntheticGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = SyntheticDataGenerator()
        cls.conditions = list(load_conditions("manifests/development_conditions.csv"))

    def test_every_single_and_compound_stratum(self) -> None:
        by_stratum = {condition.stratum: condition for condition in self.conditions}
        expected = set(self.generator.design["parameter_domains"]) | set(COMPOUND_COMPONENTS)
        self.assertTrue(expected <= set(by_stratum))
        for stratum in expected:
            record = self.generator.generate(by_stratum[stratum], 0)
            record.validate()
            self.assertEqual(record.disturbance, stratum)
            self.assertEqual(record.design_sha256, self.generator.design_sha256)
            self.assertGreater(len(record.components_pu), 1)

    def test_noise_determinism_and_condition_factors(self) -> None:
        condition = next(item for item in self.conditions if item.snr_db is not None)
        first = self.generator.generate(condition, 0)
        repeated = self.generator.generate(condition, 0)
        second = self.generator.generate(condition, 1)
        np.testing.assert_array_equal(first.samples_pu, repeated.samples_pu)
        np.testing.assert_array_equal(first.clean_pu, second.clean_pu)
        self.assertFalse(np.array_equal(first.noise_pu, second.noise_pu))
        self.assertAlmostEqual(float(first.requested_snr_db), float(second.requested_snr_db))
        self.assertLess(abs(float(first.realized_snr_db) - float(first.requested_snr_db)), 0.5)

    def test_event_adaptive_duration_and_two_sampling_tiers(self) -> None:
        sag = next(item for item in self.conditions if item.stratum == "sag")
        transient = next(
            item for item in self.conditions if item.stratum == "oscillatory_transient"
        )
        sag_record = self.generator.generate(sag)
        transient_record = self.generator.generate(transient)
        self.assertEqual(sag_record.sample_rate_hz, 12800)
        self.assertEqual(transient_record.sample_rate_hz, 51200)
        self.assertGreaterEqual(len(sag_record.time_s) / sag_record.sample_rate_hz, 0.2)

    def test_condition_batch_roundtrip_shapes(self) -> None:
        condition = next(
            item
            for item in self.conditions
            if item.stratum == "harmonics" and item.noise_realizations >= 2
        )
        records = [self.generator.generate(condition, index) for index in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            arrays_path, metadata_path = save_condition_batch(records, tmp)
            with np.load(arrays_path) as arrays:
                self.assertEqual(arrays["samples_pu"].shape[0], 2)
                self.assertEqual(arrays["samples_pu"].shape[1], len(records[0].time_s))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["realizations"]), 2)

    def test_confirmation_not_loaded_implicitly(self) -> None:
        self.assertTrue(all(item.partition == "development" for item in self.conditions))

    def test_manifest_hash_enforced(self) -> None:
        expected = json.loads(Path("manifests/PUBLIC_RELEASE_LOCK.json").read_text(encoding="utf-8"))[
            "development_manifest_sha256"
        ]
        self.assertEqual(
            self.generator.verify_manifest(
                "manifests/development_conditions.csv",
                "manifests/PUBLIC_RELEASE_LOCK.json",
            ),
            expected,
        )

    def test_exact_subcycle_transient_truth(self) -> None:
        transient = next(
            item for item in self.conditions if item.stratum == "oscillatory_transient"
        )
        record = self.generator.generate(transient)
        self.assertAlmostEqual(
            float(record.truth["event_duration_s"]),
            float(transient.parameters["support_ms"]) / 1000,
        )
        self.assertIn("sampled_event_start_s", record.truth)

    def test_tampered_condition_is_rejected(self) -> None:
        condition = next(item for item in self.conditions if item.stratum == "sag")
        bad = type(condition)(**{**condition.__dict__, "parameters": {"residual_voltage_pu": 2.0}})
        with self.assertRaises(ValueError):
            self.generator.generate(bad)


if __name__ == "__main__":
    unittest.main()
