"""Summarize noise-free physical responsiveness without selecting on nuisance outcomes."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "outputs/development_features.sqlite"
OUTPUT = ROOT / "outputs/property_response_summary.csv"

TARGETS = {
    "sag": ("residual_voltage_pu", "fundamental_deviation_peak"),
    "swell": ("event_voltage_pu", "fundamental_deviation_peak"),
    "interruption": ("residual_voltage_pu", "fundamental_deviation_peak"),
    "harmonics": ("a3_pu", "high_band_energy_fraction"),
    "flicker": ("modulation_frequency_hz", "envelope_modulation_frequency_hz"),
    "notching": ("depth_pu", "high_band_energy_fraction"),
    "oscillatory_transient": ("frequency_hz", "high_band_peak_frequency_hz"),
    "impulsive_transient": ("absolute_amplitude_pu", "high_band_energy_fraction"),
}


def main() -> int:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    failures: dict[tuple[str, str], int] = {}
    with sqlite3.connect(DATABASE) as connection:
        rows = connection.execute(
            "SELECT stratum,method,failure,features_json,truth_json FROM feature_results WHERE stage='property_response'"
        )
        for stratum, method, failure, features_json, truth_json in rows:
            key = (stratum, method)
            failures[key] = failures.get(key, 0) + int(failure)
            if failure or stratum not in TARGETS:
                continue
            target_name, feature_name = TARGETS[stratum]
            truth = json.loads(truth_json)
            target = truth["parameters"][target_name]
            estimate = json.loads(features_json).get(feature_name)
            if estimate is not None and math.isfinite(float(estimate)):
                grouped.setdefault(key, []).append((float(target), float(estimate)))
    output = []
    for stratum in TARGETS:
        for method in ("stft", "wavelet", "s_transform", "vmd"):
            pairs = grouped.get((stratum, method), [])
            target = np.array([pair[0] for pair in pairs])
            estimate = np.array([pair[1] for pair in pairs])
            rho = float(spearmanr(target, estimate).statistic) if len(pairs) >= 3 else float("nan")
            output.append(
                {
                    "stratum": stratum,
                    "method": method,
                    "target_property": TARGETS[stratum][0],
                    "candidate_feature": TARGETS[stratum][1],
                    "sample_size": len(pairs),
                    "spearman_rho": rho,
                    "absolute_spearman_rho": abs(rho) if math.isfinite(rho) else "",
                    "failures": failures.get((stratum, method), 0),
                    "response_gate_pass": bool(math.isfinite(rho) and abs(rho) >= 0.5),
                }
            )
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "rows": len(output),
        "response_gates_passed": sum(row["response_gate_pass"] for row in output),
        "failures": sum(row["failures"] for row in output),
        "output": str(OUTPUT),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
