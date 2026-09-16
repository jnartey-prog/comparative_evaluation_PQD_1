"""Typed domain models for the descriptor-qualification workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SignalSpec:
    """Configuration for one analytical power-quality signal."""

    disturbance: str = "sag"
    sample_rate: float = 3200.0
    duration: float = 0.2
    fundamental_frequency: float = 50.0
    nominal_voltage: float = 1.0
    magnitude: float = 0.5
    event_start: float = 0.06
    event_duration: float = 0.06
    component_frequency: float = 250.0
    modulation_depth: float = 0.1
    damping: float = 40.0
    noise_std: float = 0.0
    seed: int = 0
    partition: str = "development"


@dataclass(frozen=True)
class SignalRecord:
    """Generated samples and exact provenance."""

    samples: np.ndarray
    time: np.ndarray
    sample_rate: float
    disturbance: str
    properties: dict[str, float]
    conditions: dict[str, Any]
    partition: str
    seed: int

    def __repr__(self) -> str:
        return f"SignalRecord({self.disturbance!r}, n={len(self.samples)}, partition={self.partition!r})"


@dataclass(frozen=True)
class DatasetBundle:
    """Disjoint development and hold-out records."""

    development: tuple[SignalRecord, ...]
    holdout: tuple[SignalRecord, ...]

    def __repr__(self) -> str:
        return f"DatasetBundle(development={len(self.development)}, holdout={len(self.holdout)})"


@dataclass(frozen=True)
class Representation:
    """Time-frequency or modal transform output."""

    method: str
    values: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    config: dict[str, float]


@dataclass(frozen=True)
class FeatureRecord:
    """Feature values, units, and intended interpretation."""

    values: dict[str, float]
    units: dict[str, str]
    intended_properties: dict[str, str]


@dataclass(frozen=True)
class Evidence:
    """Criterion-level statistical evidence."""

    bias: float
    mae: float
    rmse: float
    correlation: float
    slope: float
    intercept: float
    ci_lower: float
    ci_upper: float
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return asdict(self)


@dataclass(frozen=True)
class QualificationPolicy:
    """Frozen engineering thresholds used before hold-out inspection."""

    max_mae_qualified: float = 0.10
    max_mae_conditional: float = 0.25
    min_abs_correlation: float = 0.80
    min_abs_slope: float = 0.50


@dataclass(frozen=True)
class Qualification:
    """Property-specific qualification decision."""

    status: str
    failed_criteria: tuple[str, ...]
    validity_conditions: tuple[str, ...]
    evidence: Evidence


@dataclass(frozen=True)
class HoldoutConfirmation:
    """Frozen-policy comparison between development and hold-out evidence."""

    development_statuses: tuple[str, ...]
    holdout_statuses: tuple[str, ...]
    decision_agreement: float
    mean_rmse_retention: float
    policy_fingerprint: str


@dataclass
class StudyConfig:
    """Small reproducible study configuration."""

    disturbances: tuple[str, ...] = (
        "sag",
        "swell",
        "interruption",
        "harmonics",
        "flicker",
        "notching",
        "oscillatory_transient",
        "impulsive_transient",
    )
    sample_rate: float = 3200.0
    duration: float = 0.2
    seed: int = 20260813
    development_repetitions: int = 1
    holdout_repetitions: int = 1


@dataclass
class StudyResult:
    """Pipeline result with paths and scientific summaries."""

    run_id: str
    dataset: DatasetBundle
    features: list[dict[str, Any]] = field(default_factory=list)
    qualifications: list[dict[str, Any]] = field(default_factory=list)
    output_dir: str = "outputs"

    def summary(self) -> str:
        """Return a compact terminal summary."""
        return f"Run {self.run_id}: {len(self.dataset.development)} development, {len(self.dataset.holdout)} hold-out, {len(self.qualifications)} decisions"

    def __str__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        return f"<strong>TFPQ study</strong><br>{self.summary()}"
