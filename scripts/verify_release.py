"""Rapid, read-only verification of the published reproducibility package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"s_transform", "stft", "vmd", "wavelet"}
DATABASE_EXPECTATIONS = {
    "outputs/development_features.sqlite": {
        "rows": 40_960,
        "stages": {
            "broad_nuisance_screen": 23_040,
            "compound_screen": 16_000,
            "property_response": 1_920,
        },
        "rows_per_method": 10_240,
    },
    "outputs/formal_features.sqlite": {
        "rows": 80_800,
        "stages": {"formal_repeatability": 80_800},
        "rows_per_method": 20_200,
    },
    "outputs/confirmation_features.sqlite": {
        "rows": 318_400,
        "stages": {"compound_screen": 8_000, "sealed_confirmation": 310_400},
        "rows_per_method": 79_600,
    },
}


def canonical_payload(path: Path) -> bytes:
    """Return exact binary bytes or UTF-8 text with platform-neutral line endings."""
    payload = path.read_bytes()
    if b"\x00" in payload:
        return payload
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return payload
    return text.replace("\r\n", "\n").encode("utf-8")


def sha256(path: Path) -> tuple[int, str]:
    payload = canonical_payload(path)
    return len(payload), hashlib.sha256(payload).hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def check_manifest(failures: list[str]) -> int:
    manifest_path = ROOT / "FILE_MANIFEST_SHA256.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"manifest file missing: {row['path']}")
            continue
        size, digest = sha256(path)
        if size != int(row["bytes"]):
            failures.append(f"manifest size mismatch: {row['path']}")
            continue
        if digest != row["sha256"]:
            failures.append(f"manifest SHA-256 mismatch: {row['path']}")
    return len(rows)


def check_database(relative_path: str, expected: dict[str, Any], failures: list[str]) -> None:
    path = ROOT / relative_path
    if not path.is_file():
        failures.append(f"database missing: {relative_path}")
        return
    if path.stat().st_size < 1_000_000:
        failures.append(
            f"database is not materialised: {relative_path}; install Git LFS and run git lfs pull"
        )
        return
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            failures.append(f"database integrity failure: {relative_path}: {integrity}")
        rows = connection.execute("SELECT COUNT(*) FROM feature_results").fetchone()[0]
        if rows != expected["rows"]:
            failures.append(f"database row-count mismatch: {relative_path}: {rows}")
        failures_count = connection.execute(
            "SELECT COUNT(*) FROM feature_results WHERE failure=1"
        ).fetchone()[0]
        if failures_count != 0:
            failures.append(
                f"unexpected computational extraction failures: {relative_path}: {failures_count}"
            )
        stage_counts = dict(
            connection.execute(
                "SELECT stage, COUNT(*) FROM feature_results GROUP BY stage"
            ).fetchall()
        )
        if stage_counts != expected["stages"]:
            failures.append(f"database stage-count mismatch: {relative_path}: {stage_counts}")
        method_counts = dict(
            connection.execute(
                "SELECT method, COUNT(*) FROM feature_results GROUP BY method"
            ).fetchall()
        )
        expected_methods = {method: expected["rows_per_method"] for method in METHODS}
        if method_counts != expected_methods:
            failures.append(f"database method-count mismatch: {relative_path}: {method_counts}")
    finally:
        connection.close()


def require_close(
    actual: float, expected: float, name: str, failures: list[str], tolerance: float = 1e-12
) -> None:
    if abs(actual - expected) > tolerance:
        failures.append(f"headline metric mismatch: {name}: {actual} != {expected}")


def check_summaries(failures: list[str]) -> dict[str, float]:
    development = load_json("outputs/formal_analysis/formal_development_summary.json")
    confirmation = load_json("outputs/holdout_analysis/holdout_stratum_method_summary.json")
    if development.get("status") != "PASS" or development.get("records") != 80_800:
        failures.append("formal development summary status or record count does not match")
    if development.get("groups") != 32:
        failures.append("formal development summary does not contain 32 groups")
    if confirmation.get("status") != "PASS" or confirmation.get("records") != 310_400:
        failures.append("confirmation summary status or primary record count does not match")
    if confirmation.get("groups") != 32:
        failures.append("confirmation summary does not contain 32 groups")
    if confirmation.get("stage_counts") != {
        "compound_screen": 8_000,
        "sealed_confirmation": 310_400,
    }:
        failures.append("confirmation summary stage counts do not match")

    development_stft = next(
        row
        for row in development["summaries"]
        if row["stratum"] == "flicker" and row["method"] == "stft"
    )
    confirmation_stft = next(
        row
        for row in confirmation["summaries"]
        if row["stratum"] == "flicker" and row["method"] == "stft"
    )
    metrics = {
        "development_standardized_rmse": development_stft["standardized_rmse"],
        "development_tolerance_coverage": development_stft[
            "all_record_pass_proportion_primary_tolerance"
        ],
        "confirmation_standardized_rmse": confirmation_stft["standardized_rmse"],
        "confirmation_tolerance_coverage": confirmation_stft[
            "all_record_pass_proportion_primary_tolerance"
        ],
    }
    require_close(metrics["development_standardized_rmse"], 0.7018251515486409, "development RMSE", failures)
    require_close(metrics["development_tolerance_coverage"], 0.9975, "development coverage", failures)
    require_close(metrics["confirmation_standardized_rmse"], 0.7850376848685267, "confirmation RMSE", failures)
    require_close(metrics["confirmation_tolerance_coverage"], 0.9918556701030928, "confirmation coverage", failures)
    return metrics


def check_qualification_matrix(failures: list[str]) -> int:
    path = ROOT / "outputs/qualification_matrix/primary_qualification_matrix.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32:
        failures.append(f"qualification matrix row-count mismatch: {len(rows)}")
    qualified = [row for row in rows if row["final_status"] == "Qualified"]
    expected = [("flicker", "stft")]
    observed = [(row["stratum"], row["method"]) for row in qualified]
    if observed != expected:
        failures.append(f"qualified-combination mismatch: {observed}")
    development_passes = [
        (row["stratum"], row["method"])
        for row in rows
        if row["development_all_gates_pass"] == "True"
    ]
    if development_passes != expected:
        failures.append(f"development six-gate decision mismatch: {development_passes}")
    return len(qualified)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the verification report as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    manifest_rows = check_manifest(failures)
    for relative_path, expected in DATABASE_EXPECTATIONS.items():
        check_database(relative_path, expected, failures)
    metrics = check_summaries(failures)
    qualified_count = check_qualification_matrix(failures)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "manifest_files_verified": manifest_rows,
        "databases_verified": len(DATABASE_EXPECTATIONS),
        "qualified_combinations": qualified_count,
        "headline_metrics": metrics,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Release verification: {report['status']}")
        print(f"Files verified: {manifest_rows}")
        print(f"SQLite databases verified: {len(DATABASE_EXPECTATIONS)}")
        print(f"Qualified combinations: {qualified_count} (expected: 1, flicker/STFT)")
        print(
            "STFT flicker standardized RMSE: "
            f"development={metrics['development_standardized_rmse']:.6f}, "
            f"confirmation={metrics['confirmation_standardized_rmse']:.6f}"
        )
        print(
            "STFT flicker tolerance coverage: "
            f"development={metrics['development_tolerance_coverage']:.6f}, "
            f"confirmation={metrics['confirmation_tolerance_coverage']:.6f}"
        )
        for failure in failures:
            print(f"FAIL: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
