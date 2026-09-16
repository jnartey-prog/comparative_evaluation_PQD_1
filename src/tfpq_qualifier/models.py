"""Domain records shared by transform and descriptor extraction code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
class Representation:
    """Time-frequency or modal transform output."""

    method: str
    values: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    config: dict[str, float]
