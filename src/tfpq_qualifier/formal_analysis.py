"""Policy-enforcing analysis helpers for frozen formal development results."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

METHODS = ("stft", "wavelet", "s_transform", "vmd")
TARGETS: dict[str, tuple[str, str]] = {
    "sag": ("residual_voltage_pu", "fundamental_deviation_peak"),
    "swell": ("event_voltage_pu", "fundamental_deviation_peak"),
    "interruption": ("residual_voltage_pu", "fundamental_deviation_peak"),
    "harmonics": ("a3_pu", "high_band_energy_fraction"),
    "flicker": ("modulation_frequency_hz", "envelope_modulation_frequency_hz"),
    "notching": ("depth_pu", "high_band_energy_fraction"),
    "oscillatory_transient": ("frequency_hz", "high_band_peak_frequency_hz"),
    "impulsive_transient": ("absolute_amplitude_pu", "high_band_energy_fraction"),
}


@dataclass(frozen=True)
class FormalMetric:
    """One policy-classified formal development observation."""

    condition_id: str
    realization: int
    stratum: str
    method: str
    requested_snr_db: float
    realized_snr_db: float
    seed: int
    target_name: str
    feature_name: str
    truth: float
    raw_estimate: float
    calibrated_estimate: float
    normalized_absolute_error: float
    property_tolerance: float
    signed_tolerance_units: float
    estimable: bool
    structurally_undefined: bool
    primary_failure_due_to_nonestimability: bool
    passes_half_tolerance: bool
    passes_primary_tolerance: bool
    passes_double_tolerance: bool


def canonical_sha256(value: Mapping[str, Any], hash_field: str) -> str:
    """Reproduce a canonical JSON hash with its embedded hash field nulled."""
    candidate = dict(value)
    candidate[hash_field] = None
    payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Reject a changed or incomplete structural-undefined policy."""
    stored = str(policy.get("policy_sha256", ""))
    if not stored or canonical_sha256(policy, "policy_sha256") != stored:
        raise ValueError("structural-undefined policy hash mismatch")
    primary = policy["primary_analysis"]
    if primary["imputation"] != "prohibited" or primary["zero_substitution"] != "prohibited":
        raise ValueError("policy must prohibit imputation and zero substitution")
    if primary["silent_complete_case_analysis"] != "prohibited":
        raise ValueError("policy must prohibit silent complete-case primary analysis")


def classify_estimate(feature_name: str, features: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return estimability and structural-undefined status under the frozen rule."""
    estimate = float(features[feature_name])
    estimable = math.isfinite(estimate)
    if feature_name != "high_band_peak_frequency_hz" or estimable:
        return estimable, False
    high_energy = float(features["high_band_energy_fraction"])
    if high_energy != 0.0:
        raise ValueError("nonfinite high-band peak with nonzero high-band energy")
    return False, True


def property_tolerance(
    stratum: str,
    target_name: str,
    truth: float,
    truth_document: Mapping[str, Any],
    threshold_policy: Mapping[str, Any],
) -> float:
    """Evaluate the frozen record-specific physical tolerance."""
    specification = threshold_policy["primary_property_tolerances"][f"{stratum}.{target_name}"]
    relative = float(specification["relative_fraction"]) * abs(truth)
    if "absolute_floor" in specification:
        return max(float(specification["absolute_floor"]), relative)
    if "absolute_floor_hz" in specification:
        return max(float(specification["absolute_floor_hz"]), relative)
    if "record_bin_hz" in specification:
        sample_count = int(truth_document["sample_count"])
        record_bin = float(specification["low_frequency_sample_rate_hz"]) / sample_count
        return max(record_bin, relative)
    raise ValueError(f"unsupported tolerance specification: {stratum}.{target_name}")


def build_metric(
    row: Mapping[str, Any],
    calibration: tuple[float, float],
    scale: float,
    threshold_policy: Mapping[str, Any],
) -> FormalMetric:
    """Convert one successful extraction row into one policy-classified metric."""
    if row["stage"] != "formal_repeatability":
        raise ValueError("formal analysis accepts formal_repeatability rows only")
    if int(row["failure"]):
        raise ValueError("transform failure rows require a separate failure path")
    stratum = str(row["stratum"])
    target_name, feature_name = TARGETS[stratum]
    truth_document = json.loads(str(row["truth_json"]))
    truth = float(truth_document["parameters"][target_name])
    tolerance = property_tolerance(
        stratum, target_name, truth, truth_document, threshold_policy
    )
    features = json.loads(str(row["features_json"]))
    estimable, structurally_undefined = classify_estimate(feature_name, features)
    raw = float(features[feature_name])
    if estimable:
        slope, intercept = calibration
        calibrated = slope * raw + intercept
        error = abs(calibrated - truth) / scale
        tolerance_units = (calibrated - truth) / tolerance
    else:
        calibrated = float("nan")
        error = float("nan")
        tolerance_units = float("nan")
    return FormalMetric(
        condition_id=str(row["condition_id"]),
        realization=int(row["realization"]),
        stratum=stratum,
        method=str(row["method"]),
        requested_snr_db=float(row["requested_snr_db"]),
        realized_snr_db=float(row["realized_snr_db"]),
        seed=int(row["seed"]),
        target_name=target_name,
        feature_name=feature_name,
        truth=truth,
        raw_estimate=raw,
        calibrated_estimate=calibrated,
        normalized_absolute_error=error,
        property_tolerance=tolerance,
        signed_tolerance_units=tolerance_units,
        estimable=estimable,
        structurally_undefined=structurally_undefined,
        primary_failure_due_to_nonestimability=not estimable,
        passes_half_tolerance=estimable and abs(tolerance_units) <= 0.5,
        passes_primary_tolerance=estimable and abs(tolerance_units) <= 1.0,
        passes_double_tolerance=estimable and abs(tolerance_units) <= 2.0,
    )


def _cluster_bootstrap(
    metrics: list[FormalMetric],
    statistic: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    clusters: dict[str, list[FormalMetric]] = defaultdict(list)
    for metric in metrics:
        clusters[metric.condition_id].append(metric)
    names = sorted(clusters)
    totals = np.asarray([len(clusters[name]) for name in names], dtype=float)
    estimable = np.asarray(
        [sum(metric.estimable for metric in clusters[name]) for name in names], dtype=float
    )
    error_counts = estimable.copy()
    error_sums = np.asarray(
        [
            sum(
                metric.normalized_absolute_error
                for metric in clusters[name]
                if metric.estimable
            )
            for name in names
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.integers(0, len(names), len(names))
        if statistic == "availability":
            estimates[index] = np.sum(estimable[selected]) / np.sum(totals[selected])
        elif statistic == "conditional_mae":
            denominator = np.sum(error_counts[selected])
            estimates[index] = (
                np.sum(error_sums[selected]) / denominator if denominator else np.nan
            )
        else:
            raise ValueError(f"unknown bootstrap statistic: {statistic}")
    finite = estimates[np.isfinite(estimates)]
    if not finite.size:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def summarize_group(
    metrics: Iterable[FormalMetric], *, bootstrap_resamples: int, seed: int
) -> dict[str, Any]:
    """Summarize all-record availability and finite-estimate conditional accuracy."""
    group = list(metrics)
    if not group:
        raise ValueError("a non-empty metric group is required")
    finite = [metric for metric in group if metric.estimable]
    errors = np.asarray([metric.normalized_absolute_error for metric in finite], dtype=float)
    availability = len(finite) / len(group)
    availability_low, availability_high = _cluster_bootstrap(
        group, "availability", resamples=bootstrap_resamples, seed=seed
    )
    mae_low, mae_high = _cluster_bootstrap(
        group, "conditional_mae", resamples=bootstrap_resamples, seed=seed + 1
    )
    if finite and len({metric.truth for metric in finite}) > 1:
        truth = np.asarray([metric.truth for metric in finite], dtype=float)
        estimates = np.asarray([metric.calibrated_estimate for metric in finite], dtype=float)
        slope, intercept = np.polyfit(truth, estimates, 1)
        correlation = float(np.corrcoef(truth, estimates)[0, 1])
    else:
        slope = intercept = correlation = float("nan")
    signed_units = np.asarray([metric.signed_tolerance_units for metric in finite], dtype=float)
    primary_pass = np.mean([metric.passes_primary_tolerance for metric in group])
    bias_gate = bool(signed_units.size and abs(float(np.mean(signed_units))) <= 0.25)
    rmse_gate = bool(
        signed_units.size and float(np.sqrt(np.mean(signed_units**2))) <= 1.0
    )
    pass_gate = bool(primary_pass >= 0.95)
    numerical_gates_pass = bias_gate and rmse_gate and pass_gate
    return {
        "stratum": group[0].stratum,
        "method": group[0].method,
        "target_name": group[0].target_name,
        "feature_name": group[0].feature_name,
        "all_record_n": len(group),
        "independent_conditions": len({metric.condition_id for metric in group}),
        "estimable_n": len(finite),
        "structurally_undefined_n": sum(metric.structurally_undefined for metric in group),
        "primary_failure_due_to_nonestimability_n": sum(
            metric.primary_failure_due_to_nonestimability for metric in group
        ),
        "availability_proportion": availability,
        "availability_cluster_bootstrap_ci_low": availability_low,
        "availability_cluster_bootstrap_ci_high": availability_high,
        "conditional_normalized_mae": float(np.mean(errors)) if errors.size else float("nan"),
        "conditional_normalized_mae_cluster_bootstrap_ci_low": mae_low,
        "conditional_normalized_mae_cluster_bootstrap_ci_high": mae_high,
        "conditional_normalized_median_absolute_error": (
            float(np.median(errors)) if errors.size else float("nan")
        ),
        "conditional_normalized_rmse": (
            float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan")
        ),
        "conditional_calibration_slope": float(slope),
        "conditional_calibration_intercept": float(intercept),
        "conditional_pearson_correlation": correlation,
        "standardized_absolute_bias": (
            abs(float(np.mean(signed_units))) if signed_units.size else float("nan")
        ),
        "standardized_rmse": (
            float(np.sqrt(np.mean(signed_units**2))) if signed_units.size else float("nan")
        ),
        "all_record_pass_proportion_half_tolerance": float(
            np.mean([metric.passes_half_tolerance for metric in group])
        ),
        "all_record_pass_proportion_primary_tolerance": float(primary_pass),
        "all_record_pass_proportion_double_tolerance": float(
            np.mean([metric.passes_double_tolerance for metric in group])
        ),
        "bias_gate_pass": bias_gate,
        "rmse_gate_pass": rmse_gate,
        "all_record_95_percent_gate_pass": pass_gate,
        "extraction_failure_rate": 0.0,
        "extraction_failure_gate_pass": True,
        "numerical_development_gates_pass": numerical_gates_pass,
        "physical_response_gate_status": "PENDING_TARGET_VS_NUISANCE_MODEL",
        "calibration_ci_gate_status": "PENDING_CLUSTER_BOOTSTRAP_CI",
        "development_qualification_status": (
            "DEVELOPMENT_NUMERICAL_GATE_FAIL"
            if not numerical_gates_pass
            else "PENDING_CALIBRATION_AND_PHYSICAL_RESPONSE_GATES"
        ),
        "final_qualification_status": "BLOCKED_PENDING_CONFIRMATION",
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with a concrete mapping type."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def calibration_cluster_bootstrap(
    metrics: Iterable[FormalMetric], *, resamples: int, seed: int
) -> dict[str, float | bool]:
    """Estimate calibration uncertainty by resampling independent conditions."""
    group = [metric for metric in metrics if metric.estimable]
    clusters: dict[str, list[FormalMetric]] = defaultdict(list)
    for metric in group:
        clusters[metric.condition_id].append(metric)
    names = sorted(clusters)
    if len(names) < 2:
        return {
            "slope": float("nan"),
            "slope_ci_low": float("nan"),
            "slope_ci_high": float("nan"),
            "intercept": float("nan"),
            "intercept_ci_low": float("nan"),
            "intercept_ci_high": float("nan"),
            "gate_pass": False,
        }

    def fit(rows: list[FormalMetric]) -> tuple[float, float]:
        truth = np.asarray([row.truth for row in rows], dtype=float)
        estimate = np.asarray([row.calibrated_estimate for row in rows], dtype=float)
        if len(np.unique(truth)) < 2:
            return float("nan"), float("nan")
        slope, intercept = np.polyfit(truth, estimate, 1)
        return float(slope), float(intercept)

    slope, intercept = fit(group)
    rng = np.random.default_rng(seed)
    boot = np.empty((resamples, 2), dtype=float)
    for index in range(resamples):
        selected = rng.integers(0, len(names), len(names))
        rows = [row for item in selected for row in clusters[names[item]]]
        boot[index] = fit(rows)
    finite = boot[np.all(np.isfinite(boot), axis=1)]
    if not finite.size:
        slope_low = slope_high = intercept_low = intercept_high = float("nan")
    else:
        slope_low, slope_high = np.quantile(finite[:, 0], [0.025, 0.975])
        intercept_low, intercept_high = np.quantile(finite[:, 1], [0.025, 0.975])
    gate = bool(slope_low <= 1.0 <= slope_high and intercept_low <= 0.0 <= intercept_high)
    return {
        "slope": slope,
        "slope_ci_low": float(slope_low),
        "slope_ci_high": float(slope_high),
        "intercept": intercept,
        "intercept_ci_low": float(intercept_low),
        "intercept_ci_high": float(intercept_high),
        "gate_pass": gate,
    }


def linear_response_diagnostic(
    response: np.ndarray,
    design: np.ndarray,
    names: list[str],
    *,
    target_name: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Fit a condition-level response model and compare target and nuisance spans."""
    if response.ndim != 1 or design.ndim != 2 or len(response) != len(design):
        raise ValueError("response and design dimensions do not agree")
    if target_name not in names:
        raise ValueError("target covariate is missing")
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    target_index = names.index(target_name)
    nuisance_indices = [index for index, name in enumerate(names) if name != target_name]
    target_effect = float(coefficients[target_index])
    largest_nuisance = max(
        (abs(float(coefficients[index])) for index in nuisance_indices), default=0.0
    )
    rng = np.random.default_rng(seed)
    margins = np.empty(bootstrap_resamples, dtype=float)
    for index in range(bootstrap_resamples):
        selected = rng.integers(0, len(response), len(response))
        boot_coef = np.linalg.lstsq(design[selected], response[selected], rcond=None)[0]
        boot_nuisance = max(
            (abs(float(boot_coef[item])) for item in nuisance_indices), default=0.0
        )
        margins[index] = float(boot_coef[target_index]) - boot_nuisance
    low, high = np.quantile(margins[np.isfinite(margins)], [0.025, 0.975])
    return {
        "target_effect": target_effect,
        "largest_absolute_nuisance_effect": largest_nuisance,
        "target_minus_largest_nuisance": target_effect - largest_nuisance,
        "margin_bootstrap_ci_low": float(low),
        "margin_bootstrap_ci_high": float(high),
        "monotonic_positive": target_effect > 0.0,
        "target_exceeds_largest_nuisance": target_effect > largest_nuisance,
        "point_margin_pass": target_effect > 0.0 and target_effect > largest_nuisance,
        "physical_response_gate_pass": bool(target_effect > 0.0 and low > 0.0),
        "bootstrap_supportive_strict_pass": bool(target_effect > 0.0 and low > 0.0),
    }
