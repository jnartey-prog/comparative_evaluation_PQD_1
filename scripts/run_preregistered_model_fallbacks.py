"""Run paired condition-cluster bootstrap fallbacks for failed mixed-model diagnostics."""

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


def paired_condition_bootstrap(
    data: pd.DataFrame,
    outcome: str,
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    condition_method = (
        data.groupby(["condition_id", "method"], observed=True)[outcome]
        .mean()
        .unstack("method")
        .reindex(columns=METHODS)
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for first, second in combinations(METHODS, 2):
        paired = condition_method[[first, second]].dropna().to_numpy(dtype=float)
        observed = float(np.mean(paired[:, 0] - paired[:, 1]))
        estimates = np.empty(resamples, dtype=float)
        for index in range(resamples):
            selected = rng.integers(0, len(paired), len(paired))
            estimates[index] = np.mean(paired[selected, 0] - paired[selected, 1])
        low, high = np.quantile(estimates, [0.025, 0.975])
        lower_tail = (np.sum(estimates <= 0.0) + 1) / (resamples + 1)
        upper_tail = (np.sum(estimates >= 0.0) + 1) / (resamples + 1)
        rows.append(
            {
                "method_a": first,
                "method_b": second,
                "condition_n": len(paired),
                "marginal_mean_difference_a_minus_b": observed,
                "cluster_bootstrap_ci_low": float(low),
                "cluster_bootstrap_ci_high": float(high),
                "bootstrap_p_value": min(1.0, 2.0 * min(lower_tail, upper_tail)),
            }
        )
    adjusted = holm_adjust([row["bootstrap_p_value"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p_value"] = value
    return rows


def main() -> int:
    output = ROOT / "outputs/formal_analysis/inferential_models"
    diagnostics_path = output / "model_diagnostics.json"
    audit_path = output / "inferential_model_audit.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit["confirmation_records_read"] != 0 or audit["formal_records_available"] != 80800:
        raise ValueError("development-only model audit is required")
    if hashlib.sha256(diagnostics_path.read_bytes()).hexdigest() != audit["outputs_sha256"][
        diagnostics_path.name
    ]:
        raise ValueError("model diagnostics hash mismatch")
    metrics = pd.read_csv(ROOT / "outputs/formal_analysis/record_level_metrics.csv")
    metrics["method"] = pd.Categorical(metrics["method"], categories=METHODS)
    metrics["pass_primary"] = metrics["passes_primary_tolerance"].astype(int)
    fallbacks = [item for item in diagnostics["models"] if item["fallback_required"]]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(fallbacks):
        stratum = item["stratum"]
        model = item["model_name"]
        data = metrics[metrics["stratum"] == stratum]
        outcome = (
            "normalized_absolute_error" if model == "continuous_error" else "pass_primary"
        )
        contrasts = paired_condition_bootstrap(
            data, outcome, resamples=2000, seed=20260815 + index
        )
        for contrast in contrasts:
            rows.append(
                {
                    "stratum": stratum,
                    "failed_model": model,
                    "outcome": outcome,
                    "estimand": (
                        "conditional finite-estimate mean error difference"
                        if outcome == "normalized_absolute_error"
                        else "all-record qualification-pass probability difference"
                    ),
                    **contrast,
                }
            )
    path = output / "cluster_bootstrap_fallback_contrasts.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fallback_audit = {
        "status": "PASS",
        "formal_records_available": len(metrics),
        "confirmation_records_read": 0,
        "mixed_models_requiring_fallback": len(fallbacks),
        "fallback_contrasts": len(rows),
        "bootstrap_resamples": 2000,
        "pairing_unit": "formal development condition",
        "multiplicity": "Holm FWER within each stratum-model family",
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    fallback_audit_path = output / "cluster_bootstrap_fallback_audit.json"
    fallback_audit_path.write_text(json.dumps(fallback_audit, indent=2) + "\n")
    print(json.dumps({**fallback_audit, "audit": str(fallback_audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
