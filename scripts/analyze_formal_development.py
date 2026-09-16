"""Analyze frozen formal development features without accessing confirmation outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tfpq_qualifier.formal_analysis import (
    METHODS,
    TARGETS,
    FormalMetric,
    build_metric,
    load_json,
    summarize_group,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def finite_or_blank(value: Any) -> Any:
    """Return blank for nonfinite floats in CSV output."""
    return "" if isinstance(value, float) and not math.isfinite(value) else value


def json_safe(value: Any) -> Any:
    """Replace nonfinite floats recursively because formal JSON forbids NaN tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="outputs/formal_features.sqlite")
    parser.add_argument("--output-dir", default="outputs/formal_analysis")
    parser.add_argument("--bootstrap-resamples", type=int)
    args = parser.parse_args()

    database = ROOT / args.database
    output = ROOT / args.output_dir
    policy_path = ROOT / "configs/structural_undefined_frequency_policy.json"
    threshold_policy_path = ROOT / "configs/qualification_threshold_policy.json"
    design_path = ROOT / "configs/scientific_design.json"
    screening_path = ROOT / "outputs/method_blind_screening_summary.json"
    extraction_audit_path = ROOT / "outputs/formal_extraction_audit.json"
    manifest_path = ROOT / "manifests/formal_repeatability_conditions.csv"

    policy = load_json(policy_path)
    validate_policy(policy)
    threshold_policy = load_json(threshold_policy_path)
    threshold_hash = threshold_policy["policy_sha256"]
    from tfpq_qualifier.formal_analysis import canonical_sha256

    if canonical_sha256(threshold_policy, "policy_sha256") != threshold_hash:
        raise ValueError("qualification-threshold policy hash mismatch")
    design = load_json(design_path)
    screening = load_json(screening_path)
    extraction_audit = load_json(extraction_audit_path)
    if extraction_audit["status"] != "PASS" or extraction_audit["confirmation_rows"] != 0:
        raise ValueError("formal extraction must pass with zero confirmation rows")
    if hashlib.sha256(database.read_bytes()).hexdigest() != extraction_audit["database_sha256"]:
        raise ValueError("formal database hash does not match the passed extraction audit")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != extraction_audit["manifest_sha256"]:
        raise ValueError("formal manifest hash does not match the passed extraction audit")

    bootstrap_resamples = args.bootstrap_resamples or int(design["final_bootstrap_resamples"])
    if bootstrap_resamples < 100:
        raise ValueError("formal bootstrap requires at least 100 resamples")
    calibrations = {
        tuple(key.split("::", 1)): (float(value[0]), float(value[1]))
        for key, value in screening["calibration_models"].items()
    }
    scales = {
        stratum: max(
            float(design["parameter_domains"][stratum][target][1])
            - float(design["parameter_domains"][stratum][target][0]),
            1e-15,
        )
        for stratum, (target, _) in TARGETS.items()
    }

    metrics: list[FormalMetric] = []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        stages = connection.execute("SELECT stage,count(1) FROM feature_results GROUP BY stage").fetchall()
        if [(row[0], row[1]) for row in stages] != [("formal_repeatability", 80800)]:
            raise ValueError("database contains unexpected stages or row counts")
        rows = connection.execute(
            "SELECT condition_id,realization,stage,stratum,method,failure,features_json,"
            "truth_json,requested_snr_db,realized_snr_db,seed FROM feature_results "
            "ORDER BY stratum,method,condition_id,realization"
        )
        for row in rows:
            if int(row["failure"]):
                raise ValueError("formal extraction audit reported zero failures but a failure row exists")
            key = (str(row["stratum"]), str(row["method"]))
            if key not in calibrations:
                raise ValueError(f"missing development calibration: {key}")
            metrics.append(build_metric(row, calibrations[key], scales[key[0]], threshold_policy))

    expected = len(TARGETS) * len(METHODS)
    grouped: dict[tuple[str, str], list[FormalMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[(metric.stratum, metric.method)].append(metric)
    if len(grouped) != expected:
        raise ValueError(f"expected {expected} stratum-method groups, found {len(grouped)}")

    summaries = []
    for index, key in enumerate(sorted(grouped)):
        summaries.append(
            summarize_group(
                grouped[key], bootstrap_resamples=bootstrap_resamples, seed=20260815 + index * 2
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "record_level_metrics.csv"
    with record_path.open("w", newline="", encoding="utf-8") as stream:
        rows = [asdict(metric) for metric in metrics]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_blank(value) for key, value in row.items()})

    summary_path = output / "stratum_method_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: finite_or_blank(value) for key, value in row.items()})

    summary_json_path = output / "formal_development_summary.json"
    summary_document = {
        "status": "PASS",
        "scope": "formal development only; no confirmation outcomes accessed",
        "policy_id": policy["policy_id"],
        "policy_sha256": policy["policy_sha256"],
        "qualification_threshold_policy_sha256": threshold_hash,
        "database_sha256": extraction_audit["database_sha256"],
        "manifest_sha256": extraction_audit["manifest_sha256"],
        "bootstrap_resamples": bootstrap_resamples,
        "records": len(metrics),
        "groups": len(summaries),
        "qualification_status": "DEVELOPMENT_GATES_ONLY_FINAL_STATUS_REQUIRES_CONFIRMATION",
        "summaries": json_safe(summaries),
    }
    summary_json_path.write_text(
        json.dumps(summary_document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (record_path, summary_path, summary_json_path)
    }
    audit = {
        "status": "PASS",
        "formal_rows_read": len(metrics),
        "confirmation_rows_read": 0,
        "expected_groups": expected,
        "actual_groups": len(summaries),
        "policy_sha256": policy["policy_sha256"],
        "qualification_threshold_policy_sha256": threshold_hash,
        "undefined_primary_frequency_records": sum(
            metric.primary_failure_due_to_nonestimability for metric in metrics
        ),
        "outputs_sha256": hashes,
        "errors": [],
        "warnings": ["Final qualification labels remain blocked pending confirmation."],
    }
    audit_path = output / "formal_development_analysis_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "audit": str(audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
