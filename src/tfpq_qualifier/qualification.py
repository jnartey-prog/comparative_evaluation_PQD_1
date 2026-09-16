"""Frozen property-specific qualification decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from .models import Evidence, HoldoutConfirmation, Qualification, QualificationPolicy


def qualify(evidence: Evidence, policy: QualificationPolicy | None = None) -> Qualification:
    """Apply transparent criterion-level rules."""
    p = policy or QualificationPolicy()
    failed: list[str] = []
    if abs(evidence.correlation) < p.min_abs_correlation:
        failed.append("association")
    if abs(evidence.slope) < p.min_abs_slope:
        failed.append("sensitivity")
    if evidence.mae <= p.max_mae_qualified and not failed:
        status = "Qualified"
    elif evidence.mae <= p.max_mae_conditional and len(failed) <= 1:
        status = "Conditionally qualified"
    else:
        status = "Not qualified"
    conditions = (
        f"MAE <= {p.max_mae_conditional}",
        f"|correlation| >= {p.min_abs_correlation}",
        "controlled synthetic domain only",
    )
    return Qualification(status, tuple(failed), conditions, evidence)


def confirm_holdout(
    development: list[Evidence], holdout: list[Evidence], policy: QualificationPolicy | None = None
) -> HoldoutConfirmation:
    """Apply one frozen policy to paired development and hold-out evidence."""
    if not development or len(development) != len(holdout):
        raise ValueError("paired non-empty development and holdout evidence are required")
    p = policy or QualificationPolicy()
    dev = tuple(qualify(item, p).status for item in development)
    test = tuple(qualify(item, p).status for item in holdout)
    agreement = float(np.mean([a == b for a, b in zip(dev, test)]))
    retention = float(
        np.mean([h.rmse / max(d.rmse, np.finfo(float).eps) for d, h in zip(development, holdout)])
    )
    fingerprint = hashlib.sha256(json.dumps(asdict(p), sort_keys=True).encode()).hexdigest()
    return HoldoutConfirmation(dev, test, agreement, retention, fingerprint)
