from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tfpq_qualifier.formal_analysis import (
    build_metric,
    calibration_cluster_bootstrap,
    classify_estimate,
    linear_response_diagnostic,
    summarize_group,
)

THRESHOLD_POLICY = {
    "primary_property_tolerances": {
        "oscillatory_transient.frequency_hz": {
            "absolute_floor_hz": 25.0,
            "relative_fraction": 0.02,
        }
    }
}


def test_structural_undefined_is_not_imputed_or_silently_dropped() -> None:
    row = {
        "condition_id": "C1",
        "realization": 0,
        "stage": "formal_repeatability",
        "stratum": "oscillatory_transient",
        "method": "vmd",
        "failure": 0,
        "features_json": json.dumps(
            {"high_band_peak_frequency_hz": float("nan"), "high_band_energy_fraction": 0.0}
        ),
        "truth_json": json.dumps({"parameters": {"frequency_hz": 500.0}}),
        "requested_snr_db": 20.0,
        "realized_snr_db": 20.1,
        "seed": 1,
    }
    metric = build_metric(row, (1.0, 0.0), 2900.0, THRESHOLD_POLICY)
    assert not metric.estimable
    assert metric.structurally_undefined
    assert metric.primary_failure_due_to_nonestimability
    assert math.isnan(metric.calibrated_estimate)
    assert math.isnan(metric.normalized_absolute_error)
    assert not metric.passes_primary_tolerance
    summary = summarize_group([metric], bootstrap_resamples=100, seed=7)
    assert summary["all_record_n"] == 1
    assert summary["estimable_n"] == 0
    assert summary["primary_failure_due_to_nonestimability_n"] == 1
    assert summary["availability_proportion"] == 0.0


def test_nonfinite_peak_with_energy_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero high-band energy"):
        classify_estimate(
            "high_band_peak_frequency_hz",
            {"high_band_peak_frequency_hz": float("nan"), "high_band_energy_fraction": 0.1},
        )


def test_confirmation_stage_is_rejected() -> None:
    row = {
        "condition_id": "H1",
        "realization": 0,
        "stage": "confirmation",
        "stratum": "oscillatory_transient",
        "method": "stft",
        "failure": 0,
        "features_json": json.dumps(
            {"high_band_peak_frequency_hz": 500.0, "high_band_energy_fraction": 0.2}
        ),
        "truth_json": json.dumps({"parameters": {"frequency_hz": 500.0}}),
        "requested_snr_db": 20.0,
        "realized_snr_db": 20.0,
        "seed": 2,
    }
    with pytest.raises(ValueError, match="formal_repeatability rows only"):
        build_metric(row, (1.0, 0.0), 2900.0, THRESHOLD_POLICY)


def test_calibration_bootstrap_recovers_identity() -> None:
    metrics = []
    for condition, truth in (("C1", 1.0), ("C2", 2.0), ("C3", 3.0), ("C4", 4.0)):
        for realization in range(3):
            row = {
                "condition_id": condition,
                "realization": realization,
                "stage": "formal_repeatability",
                "stratum": "oscillatory_transient",
                "method": "stft",
                "failure": 0,
                "features_json": json.dumps(
                    {"high_band_peak_frequency_hz": truth, "high_band_energy_fraction": 0.2}
                ),
                "truth_json": json.dumps({"parameters": {"frequency_hz": truth}}),
                "requested_snr_db": 20.0,
                "realized_snr_db": 20.0,
                "seed": realization,
            }
            metrics.append(build_metric(row, (1.0, 0.0), 3.0, THRESHOLD_POLICY))
    result = calibration_cluster_bootstrap(metrics, resamples=200, seed=4)
    assert result["slope"] == pytest.approx(1.0)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-12)
    assert result["gate_pass"]


def test_linear_response_target_dominates_nuisance() -> None:
    target = np.linspace(0.0, 1.0, 12)
    nuisance = np.tile([0.0, 1.0], 6)
    design = np.column_stack([target, nuisance])
    response = 2.0 * target + 0.1 * nuisance
    result = linear_response_diagnostic(
        response,
        design,
        ["target", "nuisance"],
        target_name="target",
        bootstrap_resamples=200,
        seed=8,
    )
    assert result["monotonic_positive"]
    assert result["target_exceeds_largest_nuisance"]
    assert result["physical_response_gate_pass"]
