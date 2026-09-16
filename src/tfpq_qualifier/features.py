"""Physically motivated feature extraction."""

from __future__ import annotations

import numpy as np

from .models import FeatureRecord, Representation, SignalRecord


def extract_features(
    signal: SignalRecord, representation: Representation | None = None
) -> FeatureRecord:
    """Extract time, frequency, energy, entropy, envelope, and support features."""
    x = np.asarray(signal.samples, dtype=float)
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(abs_x))
    energy = float(np.sum(x * x) / signal.sample_rate)
    threshold = max(np.finfo(float).eps, 0.1 * peak)
    indices = np.flatnonzero(abs_x >= threshold)
    onset = float(indices[0] / signal.sample_rate) if indices.size else float("nan")
    support = float((indices[-1] - indices[0] + 1) / signal.sample_rate) if indices.size else 0.0
    fft = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / signal.sample_rate)
    dominant = float(freqs[int(np.argmax(fft[1:]) + 1)]) if len(fft) > 1 else 0.0
    fundamental = float(signal.conditions.get("fundamental_frequency", 50.0))
    high_frequency_mask = freqs >= 2.5 * fundamental
    high_frequency_peak = (
        float(freqs[high_frequency_mask][int(np.argmax(fft[high_frequency_mask]))])
        if np.any(high_frequency_mask)
        else 0.0
    )
    probabilities = (fft * fft) / max(float(np.sum(fft * fft)), np.finfo(float).eps)
    entropy = float(-np.sum(probabilities * np.log(probabilities + np.finfo(float).eps)))
    concentration = float(np.max(probabilities))
    if representation is not None and representation.values.size:
        concentration = float(
            np.max(representation.values)
            / max(float(np.sum(representation.values)), np.finfo(float).eps)
        )
    values = {
        "rms": rms,
        "peak": peak,
        "energy": energy,
        "onset": onset,
        "support_duration": support,
        "dominant_frequency": dominant,
        "high_frequency_peak": high_frequency_peak,
        "spectral_entropy": entropy,
        "energy_concentration": concentration,
    }
    units = {
        "rms": "p.u.",
        "peak": "p.u.",
        "energy": "p.u.^2 s",
        "onset": "s",
        "support_duration": "s",
        "dominant_frequency": "Hz",
        "high_frequency_peak": "Hz",
        "spectral_entropy": "dimensionless",
        "energy_concentration": "dimensionless",
    }
    intended = {
        "rms": "magnitude",
        "peak": "magnitude",
        "energy": "energy",
        "onset": "event_start",
        "support_duration": "event_duration",
        "dominant_frequency": "component_frequency",
        "high_frequency_peak": "component_frequency",
        "spectral_entropy": "spectral_dispersion",
        "energy_concentration": "localization",
    }
    return FeatureRecord(values, units, intended)
