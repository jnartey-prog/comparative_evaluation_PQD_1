"""Primary development-only binary inference using paired condition-cluster bootstrap."""

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
OUTCOMES = (
    ("half_tolerance", "passes_half_tolerance"),
    ("primary_tolerance", "passes_primary_tolerance"),
    ("double_tolerance", "passes_double_tolerance"),
)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_family(
    data: pd.DataFrame,
    outcome: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    condition_method = (
        data.groupby(["condition_id", "method"], observed=True)[outcome]
        .mean()
        .unstack("method")
        .reindex(columns=METHODS)
    )
    if condition_method.isna().any().any():
        raise ValueError("paired binary analysis requires every method in every condition")
    values = condition_method.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = np.mean(values[selected], axis=1)
    marginals: list[dict[str, Any]] = []
    for index, method in enumerate(METHODS):
        low, high = np.quantile(bootstrap_means[:, index], [0.025, 0.975])
        marginals.append(
            {
                "method": method,
                "condition_n": len(values),
                "marginal_probability": float(np.mean(values[:, index])),
                "cluster_bootstrap_ci_low": float(low),
                "cluster_bootstrap_ci_high": float(high),
            }
        )
    contrasts: list[dict[str, Any]] = []
    for first, second in combinations(range(len(METHODS)), 2):
        differences = bootstrap_means[:, first] - bootstrap_means[:, second]
        observed = float(np.mean(values[:, first] - values[:, second]))
        low, high = np.quantile(differences, [0.025, 0.975])
        lower_tail = (np.sum(differences <= 0.0) + 1) / (resamples + 1)
        upper_tail = (np.sum(differences >= 0.0) + 1) / (resamples + 1)
        contrasts.append(
            {
                "method_a": METHODS[first],
                "method_b": METHODS[second],
                "condition_n": len(values),
                "probability_difference_a_minus_b": observed,
                "cluster_bootstrap_ci_low": float(low),
                "cluster_bootstrap_ci_high": float(high),
                "bootstrap_p_value": min(1.0, 2.0 * min(lower_tail, upper_tail)),
            }
        )
    adjusted = holm_adjust([row["bootstrap_p_value"] for row in contrasts])
    for row, value in zip(contrasts, adjusted):
        row["holm_p_value"] = value
    return marginals, contrasts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    output = ROOT / "outputs/formal_analysis/inferential_models"
    metrics_path = ROOT / "outputs/formal_analysis/record_level_metrics.csv"
    prior_audit_path = ROOT / "outputs/formal_analysis/formal_development_analysis_audit.json"
    prior_audit = json.loads(prior_audit_path.read_text())
    if prior_audit["status"] != "PASS" or prior_audit["confirmation_rows_read"] != 0:
        raise ValueError("passed development-only record audit is required")
    if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != prior_audit["outputs_sha256"][
        metrics_path.name
    ]:
        raise ValueError("record-level metrics hash mismatch")
    metrics = pd.read_csv(metrics_path)
    metrics["method"] = pd.Categorical(metrics["method"], categories=METHODS)
    for _, column in OUTCOMES:
        metrics[column] = metrics[column].astype(int)
    metrics["estimable_binary"] = metrics["estimable"].astype(int)

    marginal_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    family_index = 0
    for stratum in sorted(metrics["stratum"].unique()):
        data = metrics[metrics["stratum"] == stratum]
        for label, outcome in OUTCOMES:
            marginals, contrasts = bootstrap_family(
                data, outcome, resamples=2000, seed=2026081500 + family_index
            )
            prefix = {
                "stratum": stratum,
                "outcome": "qualification_pass",
                "tolerance_level": label,
                "primary_decision_level": label == "primary_tolerance",
            }
            marginal_rows.extend([{**prefix, **row} for row in marginals])
            contrast_rows.extend([{**prefix, **row} for row in contrasts])
            family_index += 1
    oscillatory = metrics[metrics["stratum"] == "oscillatory_transient"]
    marginals, contrasts = bootstrap_family(
        oscillatory,
        "estimable_binary",
        resamples=2000,
        seed=2026081500 + family_index,
    )
    prefix = {
        "stratum": "oscillatory_transient",
        "outcome": "primary_frequency_availability",
        "tolerance_level": "not_applicable",
        "primary_decision_level": False,
    }
    marginal_rows.extend([{**prefix, **row} for row in marginals])
    contrast_rows.extend([{**prefix, **row} for row in contrasts])

    marginal_path = output / "primary_binary_marginal_probabilities.csv"
    contrast_path = output / "primary_binary_method_contrasts.csv"
    write_csv(marginal_path, marginal_rows)
    write_csv(contrast_path, contrast_rows)
    audit = {
        "status": "PASS",
        "formal_records_read": len(metrics),
        "confirmation_records_read": 0,
        "bootstrap_resamples": 2000,
        "families": family_index + 1,
        "marginal_estimates": len(marginal_rows),
        "pairwise_method_contrasts": len(contrast_rows),
        "independent_resampling_unit": "formal development condition",
        "primary_binary_estimator": "paired condition-cluster bootstrap marginal inference",
        "bayesian_glmm_role": "supportive sensitivity only",
        "multiplicity": "Holm FWER within each disturbance-outcome-tolerance family",
        "outputs_sha256": {
            marginal_path.name: hashlib.sha256(marginal_path.read_bytes()).hexdigest(),
            contrast_path.name: hashlib.sha256(contrast_path.read_bytes()).hexdigest(),
        },
    }
    audit_path = output / "primary_binary_cluster_bootstrap_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({**audit, "audit": str(audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
