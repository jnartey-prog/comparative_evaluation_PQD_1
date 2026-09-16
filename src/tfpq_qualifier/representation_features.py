"""Comparable descriptors derived from time-frequency or modal representations."""

from __future__ import annotations

import numpy as np

from .models import Representation


def extract_representation_features(
    representation: Representation,
    *,
    fundamental_frequency_hz: float = 50.0,
    clean_pre_event_s: float = 0.1,
) -> dict[str, float]:
    """Return common deployable summaries on the frozen 0-3 kHz support."""
    magnitude = np.abs(np.asarray(representation.values, dtype=float))
    if magnitude.ndim != 2 or not magnitude.size or not np.all(np.isfinite(magnitude)):
        raise ValueError("a finite two-dimensional representation is required")
    power = magnitude**2
    total = max(float(np.sum(power)), np.finfo(float).eps)
    frequency_energy = np.sum(power, axis=1)
    temporal_energy = np.sum(power, axis=0)
    frequencies = np.asarray(representation.frequencies, dtype=float)
    times = np.asarray(representation.times, dtype=float)
    if len(frequencies) != power.shape[0] or len(times) != power.shape[1]:
        raise ValueError("representation axes do not match its values")
    dominant_index = int(np.argmax(frequency_energy))
    centroid = float(np.sum(frequencies * frequency_energy) / np.sum(frequency_energy))
    spread = float(
        np.sqrt(
            np.sum(((frequencies - centroid) ** 2) * frequency_energy) / np.sum(frequency_energy)
        )
    )
    normalized = power / total
    entropy = float(-np.sum(normalized * np.log(normalized + np.finfo(float).eps)))
    threshold = 0.1 * float(np.max(temporal_energy))
    support = np.flatnonzero(temporal_energy >= threshold)
    onset = float(times[support[0]]) if support.size else float("nan")
    duration = float(times[support[-1]] - times[support[0]]) if support.size > 1 else 0.0
    fundamental_index = int(np.argmin(np.abs(frequencies - fundamental_frequency_hz)))
    fundamental_envelope = magnitude[fundamental_index]
    baseline_mask = times < clean_pre_event_s
    baseline = (
        float(np.median(fundamental_envelope[baseline_mask]))
        if np.any(baseline_mask)
        else float(fundamental_envelope[0])
    )
    deviation = np.abs(fundamental_envelope - baseline) / max(baseline, np.finfo(float).eps)
    envelope_support = np.flatnonzero(deviation >= 0.05)
    envelope_onset = float(times[envelope_support[0]]) if envelope_support.size else float("nan")
    envelope_duration = (
        float(times[envelope_support[-1]] - times[envelope_support[0]])
        if envelope_support.size > 1
        else 0.0
    )
    high_mask = frequencies >= 2.5 * fundamental_frequency_hz
    high_energy = frequency_energy[high_mask]
    high_frequencies = frequencies[high_mask]
    high_peak = (
        float(high_frequencies[int(np.argmax(high_energy))]) if high_energy.size else float("nan")
    )
    high_temporal = np.sum(power[high_mask], axis=0) if np.any(high_mask) else np.zeros_like(times)
    high_baseline = float(np.median(high_temporal[baseline_mask])) if np.any(baseline_mask) else 0.0
    high_threshold = max(high_baseline * 5.0, 0.1 * float(np.max(high_temporal)))
    high_support = np.flatnonzero(high_temporal >= high_threshold)
    high_onset = float(times[high_support[0]]) if high_support.size else float("nan")
    high_duration = (
        float(times[high_support[-1]] - times[high_support[0]]) if high_support.size > 1 else 0.0
    )
    if len(times) > 2:
        spacing = float(np.median(np.diff(times)))
        modulation_spectrum = np.abs(
            np.fft.rfft(fundamental_envelope - np.mean(fundamental_envelope))
        )
        modulation_frequencies = np.fft.rfftfreq(len(fundamental_envelope), spacing)
        modulation_mask = (modulation_frequencies >= 0.25) & (modulation_frequencies <= 30.0)
        modulation_peak = (
            float(
                modulation_frequencies[modulation_mask][
                    np.argmax(modulation_spectrum[modulation_mask])
                ]
            )
            if np.any(modulation_mask)
            else float("nan")
        )
    else:
        modulation_peak = float("nan")
    return {
        "tf_dominant_frequency_hz": float(frequencies[dominant_index]),
        "tf_frequency_centroid_hz": centroid,
        "tf_frequency_spread_hz": spread,
        "tf_onset_s": onset,
        "tf_support_duration_s": duration,
        "tf_energy": total,
        "tf_entropy": entropy,
        "tf_concentration": float(np.max(normalized)),
        "fundamental_baseline_amplitude": baseline,
        "fundamental_deviation_peak": float(np.max(deviation)),
        "fundamental_deviation_onset_s": envelope_onset,
        "fundamental_deviation_duration_s": envelope_duration,
        "high_band_peak_frequency_hz": high_peak,
        "high_band_energy_fraction": float(np.sum(high_energy) / total),
        "high_band_onset_s": high_onset,
        "high_band_duration_s": high_duration,
        "envelope_modulation_frequency_hz": modulation_peak,
    }
