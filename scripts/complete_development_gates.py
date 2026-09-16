"""Complete frozen development diagnostics without reading confirmation outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tfpq_qualifier.formal_analysis import (
    FormalMetric,
    calibration_cluster_bootstrap,
    linear_response_diagnostic,
    summarize_group,
)

ROOT = Path(__file__).resolve().parents[1]


def read_metrics(path: Path) -> list[FormalMetric]:
    bool_fields = {
        "estimable",
        "structurally_undefined",
        "primary_failure_due_to_nonestimability",
        "passes_half_tolerance",
        "passes_primary_tolerance",
        "passes_double_tolerance",
    }
    int_fields = {"realization", "seed"}
    text_fields = {"condition_id", "stratum", "method", "target_name", "feature_name"}
    rows: list[FormalMetric] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            values: dict[str, Any] = {}
            for key, value in source.items():
                if key in bool_fields:
                    values[key] = value.lower() == "true"
                elif key in int_fields:
                    values[key] = int(value)
                elif key in text_fields:
                    values[key] = value
                else:
                    values[key] = float(value) if value else float("nan")
            rows.append(FormalMetric(**values))
    return rows


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[row["condition_id"]] = {
                "parameters": json.loads(row["parameters_json"]),
                "phase_deg": float(row["phase_deg"]),
                "snr_db": float(row["snr_db"]),
                "position_quantile": float(row["position_quantile"]),
            }
    return result


def condition_design(
    metrics: list[FormalMetric], manifest: dict[str, dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by_condition: dict[str, list[FormalMetric]] = defaultdict(list)
    for metric in metrics:
        if metric.estimable:
            by_condition[metric.condition_id].append(metric)
    condition_ids = sorted(by_condition)
    target = metrics[0].target_name
    parameter_names = sorted(
        {
            key
            for condition_id in condition_ids
            for key in manifest[condition_id]["parameters"]
            if key != target
        }
    )
    phase_levels = sorted({manifest[item]["phase_deg"] for item in condition_ids})
    names = [target, "snr_db", "position_quantile"]
    names += parameter_names
    names += [f"phase_{level:g}_vs_{phase_levels[0]:g}" for level in phase_levels[1:]]
    raw: list[list[float]] = []
    response: list[float] = []
    for condition_id in condition_ids:
        meta = manifest[condition_id]
        parameters = meta["parameters"]
        row = [float(parameters[target]), meta["snr_db"], meta["position_quantile"]]
        row += [float(parameters.get(name, 0.0)) for name in parameter_names]
        row += [float(meta["phase_deg"] == level) for level in phase_levels[1:]]
        raw.append(row)
        response.append(float(np.mean([item.calibrated_estimate for item in by_condition[condition_id]])))
    design = np.asarray(raw, dtype=float)
    continuous = 3 + len(parameter_names)
    for column in range(continuous):
        low, high = np.min(design[:, column]), np.max(design[:, column])
        design[:, column] = (design[:, column] - low) / (high - low) if high > low else 0.0
    return np.asarray(response, dtype=float), design, names


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if isinstance(value, float) and not math.isfinite(value) else value for key, value in row.items()}
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")

    output = ROOT / "outputs/formal_analysis"
    metrics_path = output / "record_level_metrics.csv"
    manifest_path = ROOT / "manifests/formal_repeatability_conditions.csv"
    prior_audit = json.loads((output / "formal_development_analysis_audit.json").read_text())
    if prior_audit["status"] != "PASS" or prior_audit["confirmation_rows_read"] != 0:
        raise ValueError("passed development-only analysis is required")
    if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != prior_audit["outputs_sha256"][metrics_path.name]:
        raise ValueError("record-level metrics hash mismatch")

    metrics = read_metrics(metrics_path)
    manifest = read_manifest(manifest_path)
    grouped: dict[tuple[str, str], list[FormalMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[(metric.stratum, metric.method)].append(metric)

    criteria: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(sorted(grouped.items())):
        base = summarize_group(group, bootstrap_resamples=args.bootstrap_resamples, seed=81000 + index * 4)
        calibration = calibration_cluster_bootstrap(
            group, resamples=args.bootstrap_resamples, seed=81001 + index * 4
        )
        response, design, names = condition_design(group, manifest)
        physical = linear_response_diagnostic(
            response,
            design,
            names,
            target_name=group[0].target_name,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=81002 + index * 4,
        )
        calibration_rows.append({"stratum": key[0], "method": key[1], **calibration})
        response_rows.append(
            {
                "stratum": key[0],
                "method": key[1],
                "condition_n": len(response),
                "covariates": ";".join(names),
                **physical,
            }
        )
        gates = {
            "bias": base["bias_gate_pass"],
            "rmse": base["rmse_gate_pass"],
            "all_record_95_percent": base["all_record_95_percent_gate_pass"],
            "extraction_failure": base["extraction_failure_gate_pass"],
            "calibration_ci": calibration["gate_pass"],
            "physical_response": physical["physical_response_gate_pass"],
        }
        failed = [name for name, passed in gates.items() if not passed]
        completed.append(
            {
                "stratum": key[0],
                "method": key[1],
                **{f"{name}_gate_pass": passed for name, passed in gates.items()},
                "all_development_gates_pass": not failed,
                "failed_gate_count": len(failed),
                "failed_gates": ";".join(failed),
                "final_status": "BLOCKED_PENDING_FROZEN_VALIDITY_DOMAIN_AND_CONFIRMATION",
            }
        )
        for criterion, passed in gates.items():
            criteria.append(
                {
                    "stratum": key[0],
                    "method": key[1],
                    "criterion": criterion,
                    "pass": passed,
                    "scope": "complete frozen development domain",
                }
            )

    snr_rows: list[dict[str, Any]] = []
    by_snr: dict[tuple[str, str, float], list[FormalMetric]] = defaultdict(list)
    for metric in metrics:
        by_snr[(metric.stratum, metric.method, metric.requested_snr_db)].append(metric)
    for index, (key, group) in enumerate(sorted(by_snr.items())):
        summary = summarize_group(group, bootstrap_resamples=args.bootstrap_resamples, seed=91000 + index)
        snr_rows.append(
            {
                "stratum": key[0],
                "method": key[1],
                "snr_db": key[2],
                "conditions": summary["independent_conditions"],
                "records": summary["all_record_n"],
                "availability": summary["availability_proportion"],
                "standardized_absolute_bias": summary["standardized_absolute_bias"],
                "standardized_rmse": summary["standardized_rmse"],
                "pass_proportion_half_tolerance": summary["all_record_pass_proportion_half_tolerance"],
                "pass_proportion_primary_tolerance": summary["all_record_pass_proportion_primary_tolerance"],
                "pass_proportion_double_tolerance": summary["all_record_pass_proportion_double_tolerance"],
                "numerical_gates_pass": summary["numerical_development_gates_pass"],
                "interpretation": "exploratory SNR slice; not a frozen validity domain",
            }
        )

    paths = {
        "criterion_level_diagnostics.csv": criteria,
        "calibration_cluster_bootstrap.csv": calibration_rows,
        "physical_response_diagnostics.csv": response_rows,
        "snr_tolerance_sensitivity.csv": snr_rows,
        "completed_development_gates.csv": completed,
    }
    for name, rows in paths.items():
        write_csv(output / name, rows)
    document = {
        "status": "PASS",
        "scope": "formal development only; confirmation outcomes not accessed",
        "bootstrap_resamples": args.bootstrap_resamples,
        "groups": len(completed),
        "complete_domain_all_gate_pass_groups": sum(row["all_development_gates_pass"] for row in completed),
        "validity_domain_status": "NOT_FROZEN",
        "validity_domain_note": "SNR slices are diagnostic only. Any controlled domain requires scientific review, explicit freeze, external registration, and unchanged confirmation.",
        "groups_detail": safe_json(completed),
    }
    result_path = output / "development_gate_completion.json"
    result_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in paths
    }
    hashes[result_path.name] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    audit = {
        "status": "PASS",
        "formal_records_read": len(metrics),
        "confirmation_records_read": 0,
        "groups": len(completed),
        "bootstrap_resamples": args.bootstrap_resamples,
        "outputs_sha256": hashes,
        "warnings": ["No validity domain or final qualification label was frozen by this diagnostic."],
    }
    audit_path = output / "development_gate_completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "audit": str(audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
