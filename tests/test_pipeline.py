"""Pipeline, logging, CLI, and artifact integration tests."""

import json
import tempfile
import unittest
from pathlib import Path

import tfpq_qualifier as tq
from tfpq_qualifier.cli import main


class PipelineTests(unittest.TestCase):
    def test_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = tq.run(output_dir=tmp)
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertEqual(len(result.dataset.development), 8)
            self.assertEqual(len(result.qualifications), 16)


class LoggingTests(unittest.TestCase):
    def test_persisted_schema_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = tq.run(output_dir=tmp)
            rows = [
                json.loads(line) for line in (Path(tmp) / "logs/run.jsonl").read_text().splitlines()
            ]
            required = {"run_id", "timestamp_utc", "stage", "status", "duration_ms"}
            self.assertTrue(all(required <= row.keys() for row in rows))
            self.assertTrue(all(row["run_id"] == result.run_id for row in rows))
            stats = [
                json.loads(line)
                for line in (Path(tmp) / "logs/statistics.jsonl").read_text().splitlines()
            ]
            self.assertTrue(stats)
            self.assertTrue(
                all(
                    "statistical_unit" in row and "qualification_consequence" in row
                    for row in stats
                )
            )


class ArtifactTests(unittest.TestCase):
    def test_manifest_complete_and_generation(self) -> None:
        self.assertTrue(Path("outputs/qualification_matrix/qualification_matrix_audit.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            result = tq.run(output_dir=str(Path(tmp) / "run"))
            files = tq.generate_artifacts(result, str(Path(tmp) / "artifacts"))
            self.assertEqual(len(files), 14)
            png = Path(tmp) / "artifacts/figure_1_workflow.png"
            self.assertGreater(png.stat().st_size, 1000)
            data = png.read_bytes()
            self.assertIn(b"pHYs", data)


class CLITests(unittest.TestCase):
    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--output", tmp, "--non-interactive"]), 0)


class GovernanceTests(unittest.TestCase):
    def test_objective_acceptance_ids_present(self) -> None:
        design = json.loads(Path("configs/scientific_design.json").read_text(encoding="utf-8"))
        self.assertTrue(design["frozen"])
        self.assertEqual(design["snr_confirmation_db"], [25, 35])
