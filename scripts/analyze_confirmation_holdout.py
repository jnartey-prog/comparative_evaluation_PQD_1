"""Apply the frozen qualification analysis to the sealed confirmation hold-out."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from tfpq_qualifier.formal_analysis import (
    METHODS,
    TARGETS,
    build_metric,
    canonical_sha256,
    load_json,
    summarize_group,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/holdout_analysis_manifest.json")
    parser.add_argument("--output-dir", default="outputs/holdout_analysis")
    parser.add_argument("--bootstrap-resamples", type=int)
    args = parser.parse_args()

    holdout = load_json(ROOT / args.manifest)
    manifest_path = ROOT / holdout["source_manifest"]
    database = ROOT / holdout["source_feature_database"]
    extraction_audit_path = ROOT / holdout["source_extraction_audit"]
    audit = load_json(extraction_audit_path)
    policy = load_json(ROOT / "configs/structural_undefined_frequency_policy.json")
    validate_policy(policy)
    threshold_policy = load_json(ROOT / "configs/qualification_threshold_policy.json")
    if canonical_sha256(threshold_policy, "policy_sha256") != threshold_policy["policy_sha256"]:
        raise ValueError("qualification-threshold policy hash mismatch")
    design = load_json(ROOT / "configs/scientific_design.json")
    screening = load_json(ROOT / "outputs/method_blind_screening_summary.json")
    compound_policy = load_json(ROOT / "configs/compound_event_qualification_policy.json")

    if holdout["status"] != "FROZEN_READY":
        raise ValueError("hold-out manifest is not frozen-ready")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != holdout["source_manifest_sha256"]:
        raise ValueError("hold-out source manifest hash mismatch")
    if hashlib.sha256(database.read_bytes()).hexdigest() != holdout["source_feature_database_sha256"]:
        raise ValueError("hold-out source database hash mismatch")
    if audit["status"] != "PASS" or audit["actual_rows"] != holdout["expected_rows_all_methods"]:
        raise ValueError("hold-out extraction audit is not PASS or has unexpected row count")

    bootstrap = args.bootstrap_resamples or int(design["final_bootstrap_resamples"])
    if bootstrap < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
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

    metrics = []
    stage_counts = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        stage_counts = {
            row[0]: row[1]
            for row in connection.execute("SELECT stage,count(*) FROM feature_results GROUP BY stage")
        }
        if sum(stage_counts.values()) != holdout["expected_rows_all_methods"]:
            raise ValueError("unexpected hold-out stage counts")
        rows = connection.execute(
            "SELECT condition_id,realization,stage,stratum,method,failure,features_json,truth_json,"
            "requested_snr_db,realized_snr_db,seed FROM feature_results "
            "ORDER BY stratum,method,condition_id,realization"
        )
        compound_rows = 0
        for row in rows:
            if str(row["stratum"]) not in TARGETS:
                compound_rows += 1
                continue
            if int(row["failure"]):
                raise ValueError("hold-out database contains an extraction failure")
            key = (str(row["stratum"]), str(row["method"]))
            if key not in calibrations:
                raise ValueError(f"missing development calibration: {key}")
            row_data = dict(row)
            row_data["stage"] = "formal_repeatability"
            metrics.append(build_metric(row_data, calibrations[key], scales[key[0]], threshold_policy))

    expected_groups = len(TARGETS) * len(METHODS)
    grouped = defaultdict(list)
    for metric in metrics:
        grouped[(metric.stratum, metric.method)].append(metric)
    if len(grouped) != expected_groups:
        raise ValueError(f"expected {expected_groups} groups, found {len(grouped)}")
    summaries = [
        summarize_group(grouped[key], bootstrap_resamples=bootstrap, seed=20260819 + index * 2)
        for index, key in enumerate(sorted(grouped))
    ]
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "holdout_stratum_method_summary.json"
    summary_doc = {
        "status": "PASS",
        "scope": "independent synthetic hold-out analysis",
        "holdout_manifest": args.manifest,
        "source_manifest_sha256": holdout["source_manifest_sha256"],
        "source_database_sha256": holdout["source_feature_database_sha256"],
        "qualification_threshold_policy_sha256": threshold_policy["policy_sha256"],
        "structural_undefined_policy_sha256": policy["policy_sha256"],
        "bootstrap_resamples": bootstrap,
        "stage_counts": stage_counts,
        "records": len(metrics),
        "groups": len(summaries),
        "compound_policy": compound_policy,
        "compound_rows_descriptive_only": compound_rows,
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = {
        "status": "PASS",
        "scope": "independent synthetic hold-out analysis",
        "records_read": len(metrics),
        "expected_records": holdout["expected_rows_all_methods"],
        "groups": len(summaries),
        "expected_groups": expected_groups,
        "compound_rows_descriptive_only": compound_rows,
        "compound_policy_id": compound_policy["policy_id"],
        "stage_counts": stage_counts,
        "method_failures": {method: 0 for method in METHODS},
        "sqlite_integrity": integrity,
        "source_manifest_sha256": holdout["source_manifest_sha256"],
        "source_database_sha256": holdout["source_feature_database_sha256"],
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "undefined_primary_frequency_records": sum(
            metric.primary_failure_due_to_nonestimability for metric in metrics
        ),
        "errors": [],
    }
    (output / "holdout_analysis_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
