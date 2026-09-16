"""Fit prespecified development-only mixed models with audited fallbacks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew, spearmanr
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[1]
MODEL_ERRORS = (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in original order."""
    result = [float("nan")] * len(pvalues)
    finite = [(index, value) for index, value in enumerate(pvalues) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        result[index] = running
    return result


def load_analysis_frame() -> pd.DataFrame:
    metrics = pd.read_csv(ROOT / "outputs/formal_analysis/record_level_metrics.csv")
    manifest = pd.read_csv(ROOT / "manifests/formal_repeatability_conditions.csv")
    metadata = manifest[["condition_id", "parameters_json", "phase_deg"]].copy()
    parameters = metadata["parameters_json"].map(json.loads)
    metadata["property_level"] = [
        float(document[target])
        for document, target in zip(parameters, metadata["condition_id"].map(
            metrics.drop_duplicates("condition_id").set_index("condition_id")["target_name"]
        ))
    ]
    frame = metrics.merge(
        metadata[["condition_id", "phase_deg", "property_level"]],
        on="condition_id",
        how="left",
        validate="many_to_one",
    )
    if len(frame) != 80800 or frame["phase_deg"].isna().any():
        raise ValueError("analysis frame did not reconcile to 80,800 formal records")
    frame["method"] = pd.Categorical(
        frame["method"], categories=["stft", "wavelet", "s_transform", "vmd"]
    )
    frame["snr_scaled"] = (frame["requested_snr_db"] - frame["requested_snr_db"].mean()) / frame[
        "requested_snr_db"
    ].std()
    frame["property_scaled"] = frame.groupby("stratum")["property_level"].transform(
        lambda values: (values - values.mean()) / values.std()
    )
    frame["noise_cluster"] = frame["condition_id"] + "::" + frame["realization"].astype(str)
    frame["pass_primary"] = frame["passes_primary_tolerance"].astype(int)
    frame["estimable_binary"] = frame["estimable"].astype(int)
    return frame


def fit_continuous(data: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finite = data[np.isfinite(data["normalized_absolute_error"])].copy()
    formula = (
        "normalized_absolute_error ~ C(method, Treatment(reference='stft')) * snr_scaled "
        "+ property_scaled + C(phase_deg)"
    )
    status: dict[str, Any] = {
        "model": formula + " + (1|condition_id) + (1|noise_cluster)",
        "all_record_n": len(data),
        "finite_error_n": len(finite),
        "conditional_due_to_nonestimability": len(finite) != len(data),
        "fallback_used": False,
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = MixedLM.from_formula(
                formula,
                groups="condition_id",
                re_formula="1",
                vc_formula={"noise_cluster": "0 + C(noise_cluster)"},
                data=finite,
            )
            fit = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        status.update(
            {
                "estimator": "maximum-likelihood linear mixed model",
                "converged": bool(fit.converged),
                "warnings": [str(item.message) for item in caught],
                "condition_random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
                "noise_seed_variance_component": float(fit.vcomp[0]),
                "residual_variance": float(fit.scale),
            }
        )
        fitted = np.asarray(model.exog @ fit.fe_params.to_numpy(), dtype=float)
        residuals = np.asarray(model.endog, dtype=float) - fitted
        scale = math.sqrt(float(fit.scale))
        standardized = residuals / scale
        heteroscedasticity_rho, heteroscedasticity_p = spearmanr(
            np.abs(residuals), fitted
        )
        status["residual_diagnostics"] = {
            "residual_type": "marginal fixed-effect residual observed minus X beta",
            "standardized_residual_mean": float(np.mean(standardized)),
            "standardized_residual_sd": float(np.std(standardized, ddof=1)),
            "standardized_residual_skew": float(skew(standardized)),
            "standardized_residual_excess_kurtosis": float(kurtosis(standardized)),
            "standardized_residual_q01": float(np.quantile(standardized, 0.01)),
            "standardized_residual_q99": float(np.quantile(standardized, 0.99)),
            "absolute_residual_fitted_spearman_rho": float(heteroscedasticity_rho),
            "absolute_residual_fitted_spearman_p": float(heteroscedasticity_p),
        }
        rows = [
            {
                "term": name,
                "estimate": float(fit.fe_params[name]),
                "standard_error": float(fit.bse_fe[name]),
                "z": float(fit.fe_params[name] / fit.bse_fe[name]),
                "p_value": float(2 * norm.sf(abs(fit.fe_params[name] / fit.bse_fe[name]))),
            }
            for name in fit.fe_params.index
        ]
        status["diagnostic_pass"] = bool(
            fit.converged
            and np.all(np.isfinite(fit.fe_params))
            and fit.scale > 0
            and not any("singular" in item.lower() for item in status["warnings"])
        )
        status["fallback_required"] = not status["diagnostic_pass"]
        return rows, status
    except MODEL_ERRORS as error:
        status.update(
            {
                "estimator": "linear mixed model failed",
                "converged": False,
                "diagnostic_pass": False,
                "fallback_required": True,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return [], status


def fit_binomial(
    data: pd.DataFrame, outcome: str, *, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formula = (
        f"{outcome} ~ C(method, Treatment(reference='stft')) * snr_scaled "
        "+ property_scaled + C(phase_deg)"
    )
    status: dict[str, Any] = {
        "model": formula + " + (1|condition_id)",
        "all_record_n": len(data),
        "outcome": label,
        "fallback_used": False,
    }
    if data[outcome].nunique() < 2:
        status.update(
            {
                "estimator": "not estimable: constant binary outcome",
                "converged": False,
                "diagnostic_pass": False,
                "fallback_required": True,
                "error": "binary outcome has no variation",
            }
        )
        return [], status
    try:
        model = BinomialBayesMixedGLM.from_formula(
            formula, {"condition": "0 + C(condition_id)"}, data
        )
        fit = model.fit_map(method="BFGS", minim_opts={"maxiter": 500})
        names = model.exog_names
        means = np.asarray(fit.fe_mean, dtype=float)
        standard_errors = np.asarray(fit.fe_sd, dtype=float)
        rows = []
        for name, estimate, standard_error in zip(names, means, standard_errors):
            z = estimate / standard_error
            rows.append(
                {
                    "term": name,
                    "estimate": float(estimate),
                    "standard_error": float(standard_error),
                    "z": float(z),
                    "p_value": float(2 * norm.sf(abs(z))),
                }
            )
        converged = bool(fit.optim_retvals.get("success", False))
        gradient = np.asarray(fit.optim_retvals.get("jac", []), dtype=float)
        status.update(
            {
                "estimator": "Laplace-approximation binomial mixed model",
                "converged": converged,
                "diagnostic_pass": bool(
                    converged and np.all(np.isfinite(means)) and np.all(standard_errors > 0)
                ),
                "fallback_required": not converged,
                "optimizer_message": str(fit.optim_retvals.get("message", "")),
                "gradient_max_abs": (
                    float(np.max(np.abs(gradient))) if gradient.size else float("nan")
                ),
                "condition_random_log_sd": float(fit.vcp_mean[0]),
            }
        )
        return rows, status
    except MODEL_ERRORS as error:
        status.update(
            {
                "estimator": "binomial mixed model failed",
                "converged": False,
                "diagnostic_pass": False,
                "fallback_required": True,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return [], status


def write_terms(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("stratum,model,term,estimate,standard_error,z,p_value,holm_p_value\n")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stratum", choices=[
        "sag", "swell", "interruption", "harmonics", "flicker", "notching",
        "oscillatory_transient", "impulsive_transient",
    ])
    parser.add_argument("--model", choices=["all", "continuous", "binary"], default="all")
    args = parser.parse_args()
    output = ROOT / "outputs/formal_analysis/inferential_models"
    output.mkdir(parents=True, exist_ok=True)
    frame = load_analysis_frame()
    strata = [args.stratum] if args.stratum else sorted(frame["stratum"].unique())
    statuses: list[dict[str, Any]] = []
    terms: list[dict[str, Any]] = []
    for stratum in strata:
        data = frame[frame["stratum"] == stratum].copy()
        model_names = {
            "all": ("continuous_error", "qualification_pass"),
            "continuous": ("continuous_error",),
            "binary": ("qualification_pass",),
        }[args.model]
        for model_name in model_names:
            if model_name == "continuous_error":
                model_terms, status = fit_continuous(data)
            else:
                model_terms, status = fit_binomial(
                    data, "pass_primary", label="qualification pass"
                )
            pvalues = [row["p_value"] for row in model_terms]
            adjusted = holm_adjust(pvalues)
            for row, holm in zip(model_terms, adjusted):
                terms.append(
                    {"stratum": stratum, "model": model_name, **row, "holm_p_value": holm}
                )
            statuses.append({"stratum": stratum, "model_name": model_name, **status})
        if stratum == "oscillatory_transient" and args.model in {"all", "binary"}:
            model_terms, status = fit_binomial(
                data, "estimable_binary", label="primary-frequency availability"
            )
            adjusted = holm_adjust([row["p_value"] for row in model_terms])
            for row, holm in zip(model_terms, adjusted):
                terms.append(
                    {"stratum": stratum, "model": "frequency_availability", **row, "holm_p_value": holm}
                )
            statuses.append(
                {"stratum": stratum, "model_name": "frequency_availability", **status}
            )

    suffix_parts = [item for item in (args.stratum, args.model if args.model != "all" else None) if item]
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    terms_path = output / f"fixed_effects{suffix}.csv"
    status_path = output / f"model_diagnostics{suffix}.json"
    audit_path = output / f"inferential_model_audit{suffix}.json"
    write_terms(terms_path, terms)
    status_document = {
        "scope": "formal development only; no confirmation outcomes accessed",
        "models": json_safe(statuses),
    }
    status_path.write_text(json.dumps(status_document, indent=2, allow_nan=False) + "\n")
    audit = {
        "status": "PASS" if statuses and all("error" not in item for item in statuses) else "WARN",
        "formal_records_available": len(frame),
        "confirmation_records_read": 0,
        "strata": strata,
        "model_count": len(statuses),
        "diagnostic_pass_count": sum(bool(item.get("diagnostic_pass")) for item in statuses),
        "fallback_required_count": sum(bool(item.get("fallback_required")) for item in statuses),
        "outputs_sha256": {
            terms_path.name: hashlib.sha256(terms_path.read_bytes()).hexdigest(),
            status_path.name: hashlib.sha256(status_path.read_bytes()).hexdigest(),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({**audit, "audit": str(audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
