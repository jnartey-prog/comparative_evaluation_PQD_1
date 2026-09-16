"""Manifest-driven research-grade synthetic power-quality data generation.

The module deliberately separates condition enumeration from waveform generation.
It never opens the sealed-confirmation manifest unless the caller supplies that path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

EQUATION_VERSION = "tfpq-analytical-v1.1.0"
TRANSIENT_STRATA = {"notching", "oscillatory_transient", "impulsive_transient"}
COMPOUND_COMPONENTS = {
    "sag+harmonics": ("sag", "harmonics"),
    "swell+harmonics": ("swell", "harmonics"),
    "sag+oscillatory_transient": ("sag", "oscillatory_transient"),
    "flicker+harmonics": ("flicker", "harmonics"),
    "interruption+transient": ("interruption", "oscillatory_transient"),
}


@dataclass(frozen=True)
class ManifestCondition:
    """One immutable row from a frozen condition manifest."""

    condition_id: str
    partition: str
    stage: str
    stratum: str
    parameters: Mapping[str, float]
    phase_deg: float | str
    snr_db: float | None | str
    position_quantile: float
    seed_namespace: int
    noise_realizations: int


@dataclass(frozen=True)
class SyntheticRecord:
    """One realization with clean components, noise and exact provenance."""

    condition_id: str
    realization: int
    partition: str
    disturbance: str
    sample_rate_hz: float
    time_s: np.ndarray
    samples_pu: np.ndarray
    clean_pu: np.ndarray
    noise_pu: np.ndarray
    components_pu: Mapping[str, np.ndarray]
    truth: Mapping[str, Any]
    requested_snr_db: float | None
    realized_snr_db: float | None
    seed: int
    checksum_sha256: str
    design_sha256: str
    generator_sha256: str
    equation_version: str = EQUATION_VERSION

    def validate(self) -> None:
        """Raise if numerical or provenance invariants are violated."""
        n = len(self.time_s)
        if n < 8 or any(len(x) != n for x in (self.samples_pu, self.clean_pu, self.noise_pu)):
            raise ValueError("signal arrays must have one common non-trivial length")
        if not all(np.all(np.isfinite(x)) for x in (self.samples_pu, self.clean_pu, self.noise_pu)):
            raise ValueError("signal arrays must be finite")
        np.testing.assert_allclose(
            self.samples_pu, self.clean_pu + self.noise_pu, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            self.clean_pu,
            np.sum(np.stack(list(self.components_pu.values())), axis=0),
            rtol=0,
            atol=1e-12,
        )
        if hashlib.sha256(self.samples_pu.tobytes()).hexdigest() != self.checksum_sha256:
            raise ValueError("sample checksum mismatch")


def load_conditions(path: str | Path) -> Iterator[ManifestCondition]:
    """Read conditions lazily from an explicitly selected manifest."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            snr_raw = row["snr_db"]
            snr: float | None | str
            if snr_raw == "noise_free":
                snr = None
            elif snr_raw in {"partition_grid", ""}:
                snr = snr_raw
            else:
                snr = float(snr_raw)
            phase_raw = row["phase_deg"]
            phase: float | str = (
                phase_raw if phase_raw == "stratified_by_component" else float(phase_raw)
            )
            yield ManifestCondition(
                condition_id=row["condition_id"],
                partition=row["partition"],
                stage=row.get("stage", "legacy"),
                stratum=row["stratum"],
                parameters=json.loads(row["parameters_json"]),
                phase_deg=phase,
                snr_db=snr,
                position_quantile=float(row["position_quantile"]),
                seed_namespace=int(row["seed_namespace"]),
                noise_realizations=int(row["noise_realizations"]),
            )


class SyntheticDataGenerator:
    """Generate approved analytical signals from frozen manifest conditions."""

    def __init__(self, design_path: str | Path = "configs/scientific_design.json") -> None:
        self.design_path = Path(design_path)
        self.design = json.loads(self.design_path.read_text(encoding="utf-8"))
        if not self.design.get("frozen"):
            raise ValueError("synthetic generation requires a frozen scientific design")
        self.design_sha256 = str(self.design["configuration_sha256"])
        self.generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.f0 = float(self.design["signal_context"]["fundamental_frequency_hz"])
        self.nominal = float(self.design["signal_context"]["nominal_voltage_pu"])
        if self.design.get("generator_sha256") != self.generator_sha256:
            raise ValueError("generator source hash differs from the frozen scientific design")
        if self.design.get("generator_equation_version") != EQUATION_VERSION:
            raise ValueError("generator equation version differs from the frozen scientific design")

    def verify_manifest(
        self, manifest_path: str | Path, lock_path: str | Path = "manifests/PUBLIC_RELEASE_LOCK.json"
    ) -> str:
        """Verify an explicitly selected manifest against the public release lock."""
        path = Path(manifest_path)
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        expected_key = (
            "sealed_confirmation_manifest_sha256"
            if "confirmation" in path.name.lower()
            else "development_manifest_sha256"
        )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != lock[expected_key]:
            raise ValueError(f"manifest hash mismatch for {path}")
        if lock["scientific_design_canonical_sha256"] != self.design_sha256:
            raise ValueError("freeze lock and scientific design hashes disagree")
        return actual

    def validate_condition(self, condition: ManifestCondition) -> None:
        """Check one row against frozen domains and partition factor grids."""
        if condition.partition not in {"development", "confirmation"}:
            raise ValueError("partition must be development or confirmation")
        if not 0 <= condition.position_quantile <= 1:
            raise ValueError("position quantile must be in [0, 1]")
        if condition.noise_realizations < 1:
            raise ValueError("noise realizations must be positive")
        if condition.stratum in COMPOUND_COMPONENTS:
            required = {"overlap_fraction", "component_scale_quantile", "onset_order_quantile"}
            if set(condition.parameters) != required or not all(
                0 <= float(value) <= 1 for value in condition.parameters.values()
            ):
                raise ValueError("compound quantiles must be complete and in [0, 1]")
        else:
            domains = self.design["parameter_domains"]
            if condition.stratum not in domains:
                raise ValueError(f"stratum is absent from frozen domains: {condition.stratum}")
            if set(condition.parameters) != set(domains[condition.stratum]):
                raise ValueError("condition parameter names do not match the frozen domain")
            for name, value in condition.parameters.items():
                lower, upper = domains[condition.stratum][name]
                if not float(lower) <= float(value) <= float(upper):
                    raise ValueError(f"{name} is outside the frozen domain")
        phase_grid = self.design[
            "phase_confirmation_deg"
            if condition.partition == "confirmation"
            else "phase_development_deg"
        ]
        if (
            not isinstance(condition.phase_deg, str)
            and float(condition.phase_deg) not in phase_grid
        ):
            raise ValueError("phase is outside the frozen partition grid")
        if isinstance(condition.snr_db, float):
            snr_grid = self.design[
                "snr_confirmation_db"
                if condition.partition == "confirmation"
                else "snr_development_db"
            ]
            numeric_snr = [float(value) for value in snr_grid if value != "noise_free"]
            if float(condition.snr_db) not in numeric_snr:
                raise ValueError("SNR is outside the frozen partition grid")
        elif condition.snr_db not in {None, "partition_grid"}:
            raise ValueError("unsupported SNR manifest value")

    def generate(self, condition: ManifestCondition, realization: int = 0) -> SyntheticRecord:
        """Generate one deterministic realization and validate it."""
        self.validate_condition(condition)
        if realization < 0 or realization >= condition.noise_realizations:
            raise ValueError("realization is outside the manifest allocation")
        rng_seed = (condition.seed_namespace + 1_000_003 * realization) % (2**32)
        rng = np.random.default_rng(rng_seed)
        factor_rng = np.random.default_rng(condition.seed_namespace)
        sample_rate = self._sample_rate(condition.stratum)
        parameters = dict(condition.parameters)
        event_cycles = self._event_cycles(condition.stratum, parameters)
        pre = int(self.design["signal_context"]["pre_event_cycles"])
        post = int(self.design["signal_context"]["post_event_cycles"])
        slack_cycles = float(self.design["record_policies"]["positioning_slack_cycles"])
        duration = max(
            float(self.design["signal_context"]["minimum_record_duration_s"]),
            (pre + event_cycles + post + slack_cycles) / self.f0,
        )
        flicker_frequency: float | None = None
        if condition.stratum == "flicker":
            flicker_frequency = float(parameters["modulation_frequency_hz"])
        elif condition.stratum == "flicker+harmonics":
            flicker_frequency = 0.5 + 24.5 * float(parameters["overlap_fraction"])
        if flicker_frequency is not None:
            duration = max(
                duration,
                float(self.design["record_policies"]["flicker_minimum_modulation_cycles"])
                / flicker_frequency,
            )
        n = math.ceil(duration * sample_rate)
        time = np.arange(n, dtype=np.float64) / sample_rate
        event_duration = event_cycles / self.f0
        earliest = pre / self.f0
        latest = max(earliest, duration - post / self.f0 - event_duration)
        event_start = earliest + condition.position_quantile * (latest - earliest)
        mask = (time >= event_start) & (time < event_start + event_duration)
        phase_deg = self._phase(condition, factor_rng)
        phase = math.radians(phase_deg)
        base = self.nominal * np.sin(2 * np.pi * self.f0 * time + phase)
        components = {"fundamental": base.copy()}
        clean, truth = self._compose(
            condition.stratum, time, base, mask, event_start, event_duration, parameters, factor_rng
        )
        components.update(truth.pop("components"))
        requested_snr = self._snr(condition, factor_rng)
        if requested_snr is None:
            noise = np.zeros_like(clean)
            realized_snr = None
        else:
            signal_power = float(np.mean(clean**2))
            sigma = math.sqrt(signal_power / 10 ** (requested_snr / 10))
            noise = rng.normal(0.0, sigma, size=n)
            noise_power = float(np.mean(noise**2))
            realized_snr = 10 * math.log10(signal_power / noise_power)
        samples = clean + noise
        truth.update(
            {
                "event_start_s": event_start,
                "event_duration_s": event_duration,
                "event_end_s": event_start + event_duration,
                "fundamental_frequency_hz": self.f0,
                "fundamental_phase_deg": phase_deg,
                "parameters": parameters,
                "stage": condition.stage,
                "sample_count": n,
                "event_start_sample": int(np.flatnonzero(mask)[0]) if np.any(mask) else None,
                "event_end_sample_exclusive": int(np.flatnonzero(mask)[-1] + 1)
                if np.any(mask)
                else None,
                "sampled_event_start_s": float(time[np.flatnonzero(mask)[0]])
                if np.any(mask)
                else None,
                "sampled_event_end_s_exclusive": float((np.flatnonzero(mask)[-1] + 1) / sample_rate)
                if np.any(mask)
                else None,
            }
        )
        record = SyntheticRecord(
            condition.condition_id,
            realization,
            condition.partition,
            condition.stratum,
            sample_rate,
            time,
            samples,
            clean,
            noise,
            components,
            truth,
            requested_snr,
            realized_snr,
            rng_seed,
            hashlib.sha256(samples.tobytes()).hexdigest(),
            self.design_sha256,
            self.generator_sha256,
        )
        record.validate()
        return record

    def iter_manifest(
        self,
        manifest_path: str | Path,
        *,
        max_conditions: int | None = None,
        realizations: int | None = None,
    ) -> Iterator[SyntheticRecord]:
        """Stream records without retaining the complete experiment in memory."""
        for index, condition in enumerate(load_conditions(manifest_path)):
            if max_conditions is not None and index >= max_conditions:
                break
            count = (
                condition.noise_realizations
                if realizations is None
                else min(realizations, condition.noise_realizations)
            )
            for realization in range(count):
                yield self.generate(condition, realization)

    def _sample_rate(self, stratum: str) -> float:
        transient = stratum in TRANSIENT_STRATA or "transient" in stratum
        key = (
            "nominal_sampling_rate_transient_hz"
            if transient
            else "nominal_sampling_rate_low_frequency_hz"
        )
        return float(self.design["signal_context"][key])

    def _event_cycles(self, stratum: str, parameters: Mapping[str, float]) -> float:
        if "duration_cycles" in parameters:
            return float(parameters["duration_cycles"])
        if stratum == "oscillatory_transient":
            return float(parameters["support_ms"]) * self.f0 / 1000
        if stratum == "impulsive_transient":
            return float(parameters["width_ms"]) * self.f0 / 1000
        return 10.0

    def _phase(self, condition: ManifestCondition, rng: np.random.Generator) -> float:
        if isinstance(condition.phase_deg, str):
            grid = self.design[
                "phase_confirmation_deg"
                if condition.partition == "confirmation"
                else "phase_development_deg"
            ]
            return float(rng.choice(grid))
        return float(condition.phase_deg)

    def _snr(self, condition: ManifestCondition, rng: np.random.Generator) -> float | None:
        if condition.snr_db == "partition_grid":
            grid = self.design[
                "snr_confirmation_db"
                if condition.partition == "confirmation"
                else "snr_development_db"
            ]
            grid = [float(value) for value in grid if value != "noise_free"]
            return float(rng.choice(grid))
        return None if condition.snr_db is None else float(condition.snr_db)

    def _compose(
        self,
        stratum: str,
        time: np.ndarray,
        base: np.ndarray,
        mask: np.ndarray,
        event_start: float,
        event_duration: float,
        parameters: Mapping[str, float],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if stratum in COMPOUND_COMPONENTS:
            return self._compound(
                stratum, time, base, mask, event_start, event_duration, parameters, rng
            )
        clean, component, truth = self._single(
            stratum, time, base, mask, event_start, parameters, rng
        )
        truth["components"] = {stratum: component}
        return clean, truth

    def _single(
        self,
        stratum: str,
        time: np.ndarray,
        base: np.ndarray,
        mask: np.ndarray,
        event_start: float,
        parameters: Mapping[str, float],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        clean = base.copy()
        component = np.zeros_like(base)
        truth: dict[str, Any] = {}
        if stratum in {"sag", "swell", "interruption"}:
            key = "event_voltage_pu" if stratum == "swell" else "residual_voltage_pu"
            level = float(parameters[key])
            # Preserve the original phase exactly by scaling the existing fundamental.
            event_wave = level / self.nominal * base
            component[mask] = event_wave[mask] - base[mask]
            clean += component
            truth["event_voltage_pu"] = level
        elif stratum == "harmonics":
            amplitudes = {order: float(parameters[f"a{order}_pu"]) for order in (3, 5, 7)}
            phases = {order: float(parameters[f"phi{order}_deg"]) for order in (3, 5, 7)}
            for order, value in amplitudes.items():
                component += value * np.sin(
                    2 * np.pi * order * self.f0 * time + math.radians(phases[order])
                )
            clean += component
            truth["harmonic_amplitudes_pu"] = amplitudes
            truth["harmonic_phases_deg"] = phases
        elif stratum == "flicker":
            depth = float(parameters["modulation_depth"])
            frequency = float(parameters["modulation_frequency_hz"])
            component = base * depth * np.sin(2 * np.pi * frequency * time)
            clean += component
            truth.update({"modulation_depth": depth, "modulation_frequency_hz": frequency})
        elif stratum == "notching":
            depth = float(parameters["depth_pu"])
            width = float(parameters["width_ms"]) / 1000
            cycle_phase = np.mod(time * self.f0, 1.0)
            notch = (
                (cycle_phase < width * self.f0)
                | ((cycle_phase >= 0.5) & (cycle_phase < 0.5 + width * self.f0))
            ) & mask
            component[notch] = -depth
            clean += component
            truth.update({"notch_depth_pu": depth, "notch_width_s": width, "pulses_per_cycle": 2})
        elif stratum == "oscillatory_transient":
            amplitude = float(parameters["amplitude_pu"])
            frequency = float(parameters["frequency_hz"])
            time_constant = float(parameters["time_constant_ms"]) / 1000
            tau = np.maximum(time - event_start, 0.0)
            component = (
                mask
                * amplitude
                * np.exp(-tau / time_constant)
                * np.sin(2 * np.pi * frequency * tau)
            )
            clean += component
            truth.update(
                {
                    "transient_amplitude_pu": amplitude,
                    "transient_frequency_hz": frequency,
                    "time_constant_s": time_constant,
                }
            )
        elif stratum == "impulsive_transient":
            amplitude = float(parameters["absolute_amplitude_pu"])
            polarity = -1.0 if float(parameters["polarity"]) < 0 else 1.0
            width = float(parameters["width_ms"]) / 1000
            pulse = (time >= event_start) & (time < event_start + width)
            component[pulse] = polarity * amplitude
            clean += component
            truth.update({"impulse_amplitude_pu": polarity * amplitude, "impulse_width_s": width})
        else:
            raise ValueError(f"unsupported frozen stratum: {stratum}")
        return clean, component, truth

    def _compound(
        self,
        stratum: str,
        time: np.ndarray,
        base: np.ndarray,
        mask: np.ndarray,
        event_start: float,
        event_duration: float,
        parameters: Mapping[str, float],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        first, second = COMPOUND_COMPONENTS[stratum]
        scale = 0.2 + 0.6 * float(parameters["component_scale_quantile"])
        common: dict[str, float] = {}
        if first == "sag":
            common = {"residual_voltage_pu": 1.0 - 0.7 * scale}
        elif first == "swell":
            common = {"event_voltage_pu": 1.0 + 0.7 * scale}
        elif first == "interruption":
            common = {"residual_voltage_pu": 0.1 * (1 - scale)}
        elif first == "flicker":
            common = {
                "modulation_depth": 0.01 + 0.09 * scale,
                "modulation_frequency_hz": 0.5 + 24.5 * float(parameters["overlap_fraction"]),
            }
        clean1, comp1, truth1 = self._single(first, time, base, mask, event_start, common, rng)
        if second == "harmonics":
            second_parameters = {
                "a3_pu": 0.01 + 0.09 * scale,
                "a5_pu": 0.01 + 0.09 * float(parameters["overlap_fraction"]),
                "a7_pu": 0.01 + 0.09 * float(parameters["onset_order_quantile"]),
                "phi3_deg": 360 * float(parameters["onset_order_quantile"]),
                "phi5_deg": 360 * float(parameters["overlap_fraction"]),
                "phi7_deg": 360 * float(parameters["component_scale_quantile"]),
            }
        else:
            second_parameters = {
                "amplitude_pu": 0.1 + 0.7 * scale,
                "frequency_hz": 300 + 2700 * float(parameters["overlap_fraction"]),
                "time_constant_ms": 0.5 + 9.5 * scale,
            }
        _, comp2, truth2 = self._single(
            second, time, base, mask, event_start, second_parameters, rng
        )
        return clean1 + comp2, {
            "constituents": {first: truth1, second: truth2},
            "components": {first: comp1, second: comp2},
        }


def record_metadata(record: SyntheticRecord) -> dict[str, Any]:
    """Return JSON-serializable metadata without duplicating arrays."""
    return {
        "condition_id": record.condition_id,
        "realization": record.realization,
        "partition": record.partition,
        "stage": record.truth.get("stage"),
        "disturbance": record.disturbance,
        "sample_rate_hz": record.sample_rate_hz,
        "sample_count": len(record.samples_pu),
        "truth": record.truth,
        "requested_snr_db": record.requested_snr_db,
        "realized_snr_db": record.realized_snr_db,
        "seed": record.seed,
        "checksum_sha256": record.checksum_sha256,
        "design_sha256": record.design_sha256,
        "generator_sha256": record.generator_sha256,
        "equation_version": record.equation_version,
        "component_names": sorted(record.components_pu),
    }


def save_record(record: SyntheticRecord, directory: str | Path) -> tuple[Path, Path]:
    """Persist one record as compressed arrays plus readable metadata."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{record.condition_id}__r{record.realization:03d}"
    array_path = destination / f"{stem}.npz"
    metadata_path = destination / f"{stem}.json"
    arrays = {
        "time_s": record.time_s,
        "samples_pu": record.samples_pu,
        "clean_pu": record.clean_pu,
        "noise_pu": record.noise_pu,
    }
    arrays.update({f"component__{name}": values for name, values in record.components_pu.items()})
    cast(Any, np.savez_compressed)(array_path, **arrays)
    metadata_path.write_text(json.dumps(record_metadata(record), indent=2) + "\n", encoding="utf-8")
    return array_path, metadata_path


def save_condition_batch(
    records: list[SyntheticRecord], directory: str | Path
) -> tuple[Path, Path]:
    """Store all realizations of one condition in two files for scalable execution."""
    if not records or len({record.condition_id for record in records}) != 1:
        raise ValueError("a non-empty single-condition record list is required")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    condition_id = records[0].condition_id
    array_path = destination / f"{condition_id}.npz"
    metadata_path = destination / f"{condition_id}.json"
    arrays: dict[str, np.ndarray] = {
        "time_s": records[0].time_s,
        "clean_pu": records[0].clean_pu,
        "samples_pu": np.stack([record.samples_pu for record in records]),
        "noise_pu": np.stack([record.noise_pu for record in records]),
    }
    arrays.update(
        {f"component__{name}": values for name, values in records[0].components_pu.items()}
    )
    cast(Any, np.savez_compressed)(array_path, **arrays)
    metadata = {
        "condition": record_metadata(records[0]),
        "realizations": [
            {
                "realization": record.realization,
                "seed": record.seed,
                "requested_snr_db": record.requested_snr_db,
                "realized_snr_db": record.realized_snr_db,
                "checksum_sha256": record.checksum_sha256,
            }
            for record in records
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return array_path, metadata_path
