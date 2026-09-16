"""Analytical controlled power-quality signal generation."""

from __future__ import annotations

import numpy as np

from .models import DatasetBundle, SignalRecord, SignalSpec, StudyConfig

SUPPORTED = {
    "sag",
    "swell",
    "interruption",
    "harmonics",
    "flicker",
    "notching",
    "oscillatory_transient",
    "impulsive_transient",
}


def generate_signal(spec: SignalSpec) -> SignalRecord:
    """Generate an analytical signal with exact reference metadata."""
    if spec.disturbance not in SUPPORTED:
        raise ValueError(f"Unsupported disturbance: {spec.disturbance}")
    if spec.sample_rate <= 0 or spec.duration <= 0:
        raise ValueError("sample_rate and duration must be positive")
    n = max(8, round(spec.sample_rate * spec.duration))
    t = np.arange(n, dtype=float) / spec.sample_rate
    base = spec.nominal_voltage * np.sin(2 * np.pi * spec.fundamental_frequency * t)
    mask = (t >= spec.event_start) & (t < spec.event_start + spec.event_duration)
    x = base.copy()
    if spec.disturbance == "sag":
        x[mask] *= spec.magnitude
    elif spec.disturbance == "swell":
        x[mask] *= 1.0 + spec.magnitude
    elif spec.disturbance == "interruption":
        x[mask] *= min(spec.magnitude, 0.05)
    elif spec.disturbance == "harmonics":
        x += spec.magnitude * np.sin(2 * np.pi * spec.component_frequency * t)
    elif spec.disturbance == "flicker":
        x *= 1.0 + spec.modulation_depth * np.sin(2 * np.pi * 8.8 * t)
    elif spec.disturbance == "notching":
        phase = np.mod(t * spec.fundamental_frequency, 1.0)
        x[(phase < 0.025) | ((phase > 0.5) & (phase < 0.525))] -= spec.magnitude
    elif spec.disturbance == "oscillatory_transient":
        tau = np.maximum(t - spec.event_start, 0.0)
        x += (
            mask
            * spec.magnitude
            * np.exp(-spec.damping * tau)
            * np.sin(2 * np.pi * spec.component_frequency * tau)
        )
    elif spec.disturbance == "impulsive_transient":
        width = max(1, int(0.001 * spec.sample_rate))
        index = min(n - 1, int(spec.event_start * spec.sample_rate))
        x[index : index + width] += spec.magnitude
    if spec.noise_std:
        x += np.random.default_rng(spec.seed).normal(0.0, spec.noise_std, size=n)
    properties = {
        "magnitude": spec.magnitude,
        "event_start": spec.event_start,
        "event_duration": spec.event_duration,
        "component_frequency": spec.component_frequency,
        "modulation_depth": spec.modulation_depth,
        "damping": spec.damping,
    }
    conditions = {
        "fundamental_frequency": spec.fundamental_frequency,
        "nominal_voltage": spec.nominal_voltage,
        "noise_std": spec.noise_std,
    }
    return SignalRecord(
        x, t, spec.sample_rate, spec.disturbance, properties, conditions, spec.partition, spec.seed
    )


def build_dataset(config: StudyConfig | None = None) -> DatasetBundle:
    """Build deterministic disjoint development and hold-out sets."""
    cfg = config or StudyConfig()
    dev: list[SignalRecord] = []
    hold: list[SignalRecord] = []
    for i, name in enumerate(cfg.disturbances):
        for rep in range(cfg.development_repetitions):
            seed = cfg.seed + i * 100 + rep
            dev.append(
                generate_signal(
                    SignalSpec(
                        name,
                        cfg.sample_rate,
                        cfg.duration,
                        magnitude=0.25 + 0.05 * (i % 4),
                        seed=seed,
                        partition="development",
                    )
                )
            )
        for rep in range(cfg.holdout_repetitions):
            seed = cfg.seed + 10000 + i * 100 + rep
            spec = SignalSpec(
                name,
                cfg.sample_rate,
                cfg.duration,
                magnitude=0.275 + 0.05 * (i % 4),
                event_start=0.065,
                seed=seed,
                partition="holdout",
            )
            hold.append(generate_signal(spec))
    return DatasetBundle(tuple(dev), tuple(hold))
