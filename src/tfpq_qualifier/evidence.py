"""Criterion-level evidence calculations."""

from __future__ import annotations

import numpy as np

from .models import Evidence


def evaluate_feature(
    values: list[float], references: list[float], *, bootstrap_samples: int = 200, seed: int = 0
) -> Evidence:
    """Evaluate fidelity, calibration, association, and bootstrap uncertainty."""
    x = np.asarray(references, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.shape != y.shape or x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(
            "values and references must be equal-length finite sequences of at least two"
        )
    residual = y - x
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(10, bootstrap_samples)):
        idx = rng.integers(0, x.size, x.size)
        means.append(float(np.mean(residual[idx])))
    low, high = np.quantile(means, [0.025, 0.975])
    return Evidence(
        float(np.mean(residual)),
        float(np.mean(np.abs(residual))),
        float(np.sqrt(np.mean(residual * residual))),
        correlation,
        float(slope),
        float(intercept),
        float(low),
        float(high),
        int(x.size),
    )
