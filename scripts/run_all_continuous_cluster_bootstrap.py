"""Robust paired condition-cluster bootstrap sensitivity for continuous errors."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("stft", "wavelet", "s_transform", "vmd")


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_stratum(
    data: pd.DataFrame, *, resamples: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = (
        data.groupby(["condition_id", "method"], observed=True)["normalized_absolute_error"]
        .mean()
        .unstack("method")
        .reindex(columns=METHODS)
    )
    rng = np.random.default_rng(seed)
    marginal_rows: list[dict[str, Any]] = []
    for method in METHODS:
        values = matrix[method].dropna().to_numpy(dtype=float)
        selected = rng.integers(0, len(values), size=(resamples, len(values)))
        estimates = np.mean(values[selected], axis=1)
        low, high = np.quantile(estimates, [0.025, 0.975])
        marginal_rows.append(
            {
                "method": method,
                "finite_condition_n": len(values),
                "conditional_mean_normalized_absolute_error": float(np.mean(values)),
                "cluster_bootstrap_ci_low": float(low),
                "cluster_bootstrap_ci_high": float(high),
            }
        )
    contrast_rows: list[dict[str, Any]] = []
    for first, second in combinations(METHODS, 2):
        paired = matrix[[first, second]].dropna().to_numpy(dtype=float)
        selected = rng.integers(0, len(paired), size=(resamples, len(paired)))
        differences = np.mean(paired[selected, 0] - paired[selected, 1], axis=1)
        observed = float(np.mean(paired[:, 0] - paired[:, 1]))
        low, high = np.quantile(differences, [0.025, 0.975])
        lower_tail = (np.sum(differences <= 0.0) + 1) / (resamples + 1)
        upper_tail = (np.sum(differences >= 0.0) + 1) / (resamples + 1)
        contrast_rows.append(
            {
                "method_a": first,
                "method_b": second,
                "paired_finite_condition_n": len(paired),
                "conditional_mean_error_difference_a_minus_b": observed,
                "cluster_bootstrap_ci_low": float(low),
                "cluster_bootstrap_ci_high": float(high),
                "bootstrap_p_value": min(1.0, 2.0 * min(lower_tail, upper_tail)),
            }
        )
    adjusted = holm_adjust([row["bootstrap_p_value"] for row in contrast_rows])
    for row, value in zip(contrast_rows, adjusted):
        row["holm_p_value"] = value
    return marginal_rows, contrast_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    output = ROOT / "outputs/formal_analysis/inferential_models"
    metrics_path = ROOT / "outputs/formal_analysis/record_level_metrics.csv"
    audit = json.loads(
        (ROOT / "outputs/formal_analysis/formal_development_analysis_audit.json").read_text()
    )
    if audit["confirmation_rows_read"] != 0:
        raise ValueError("confirmation firewall violation")
    if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != audit["outputs_sha256"][
        metrics_path.name
    ]:
        raise ValueError("record-level metrics hash mismatch")
    metrics = pd.read_csv(metrics_path)
    metrics["method"] = pd.Categorical(metrics["method"], categories=METHODS)
    marginals: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for index, stratum in enumerate(sorted(metrics["stratum"].unique())):
        data = metrics[metrics["stratum"] == stratum]
        marginal_rows, contrast_rows = bootstrap_stratum(
            data, resamples=2000, seed=2026081600 + index
        )
        marginals.extend([{"stratum": stratum, **row} for row in marginal_rows])
        contrasts.extend([{"stratum": stratum, **row} for row in contrast_rows])
    marginal_path = output / "continuous_error_bootstrap_marginals.csv"
    contrast_path = output / "continuous_error_bootstrap_contrasts.csv"
    write_csv(marginal_path, marginals)
    write_csv(contrast_path, contrasts)
    result = {
        "status": "PASS",
        "formal_records_read": len(metrics),
        "confirmation_records_read": 0,
        "bootstrap_resamples": 2000,
        "strata": 8,
        "marginal_estimates": len(marginals),
        "pairwise_contrasts": len(contrasts),
        "estimand": "conditional finite-estimate normalized absolute error",
        "independent_resampling_unit": "formal development condition",
        "multiplicity": "Holm FWER within each disturbance family",
        "outputs_sha256": {
            marginal_path.name: hashlib.sha256(marginal_path.read_bytes()).hexdigest(),
            contrast_path.name: hashlib.sha256(contrast_path.read_bytes()).hexdigest(),
        },
    }
    audit_path = output / "continuous_error_cluster_bootstrap_audit.json"
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "audit": str(audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
