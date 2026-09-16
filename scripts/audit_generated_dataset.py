"""Independently validate persisted bundles against a locked manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tfpq_qualifier.synthetic import SyntheticDataGenerator, load_conditions

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/development_conditions.csv")
    parser.add_argument("--data", default="outputs/synthetic/development")
    parser.add_argument("--output", default="outputs/generated_dataset_audit.json")
    args = parser.parse_args()
    generator = SyntheticDataGenerator(ROOT / "configs/scientific_design.json")
    manifest = ROOT / args.manifest
    data = ROOT / args.data
    if "formal_repeatability" in manifest.name.lower():
        lock = json.loads((ROOT / "manifests/PUBLIC_RELEASE_LOCK.json").read_text(encoding="utf-8"))
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest_hash != lock["formal_repeatability_manifest_sha256"]:
            raise ValueError("formal-repeatability manifest hash mismatch")
    else:
        manifest_hash = generator.verify_manifest(manifest)
    conditions = list(load_conditions(manifest))
    errors: list[str] = []
    stage_records: dict[str, int] = {}
    stratum_records: dict[str, int] = {}
    realized_snr_max_error = 0.0
    total_records = 0
    observed_seeds: set[int] = set()
    duplicate_seeds = 0
    for condition in conditions:
        array_path = data / f"{condition.condition_id}.npz"
        metadata_path = data / f"{condition.condition_id}.json"
        if not array_path.exists() or not metadata_path.exists():
            errors.append(f"missing files for {condition.condition_id}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["condition"]["condition_id"] != condition.condition_id:
            errors.append(f"condition identifier mismatch: {condition.condition_id}")
        if metadata["condition"]["partition"] != condition.partition:
            errors.append(f"partition mismatch: {condition.condition_id}")
        if metadata["condition"]["disturbance"] != condition.stratum:
            errors.append(f"disturbance mismatch: {condition.condition_id}")
        if len(metadata["realizations"]) != condition.noise_realizations:
            errors.append(f"metadata realization count mismatch: {condition.condition_id}")
        if metadata["condition"]["design_sha256"] != generator.design_sha256:
            errors.append(f"design hash mismatch: {condition.condition_id}")
        if metadata["condition"]["generator_sha256"] != generator.generator_sha256:
            errors.append(f"generator hash mismatch: {condition.condition_id}")
        if metadata["condition"]["stage"] != condition.stage:
            errors.append(f"stage mismatch: {condition.condition_id}")
        with np.load(array_path) as arrays:
            required_arrays = {"time_s", "samples_pu", "noise_pu", "clean_pu"}
            if not required_arrays <= set(arrays.files):
                errors.append(f"required arrays missing: {condition.condition_id}")
                continue
            time_axis = arrays["time_s"]
            samples = arrays["samples_pu"]
            noise = arrays["noise_pu"]
            clean = arrays["clean_pu"]
            components = [arrays[name] for name in arrays.files if name.startswith("component__")]
            if samples.shape[0] != condition.noise_realizations:
                errors.append(f"realization count mismatch: {condition.condition_id}")
            if samples.shape != noise.shape or samples.shape[1] != len(clean):
                errors.append(f"array shape mismatch: {condition.condition_id}")
            if not np.all(np.isfinite(samples)):
                errors.append(f"non-finite samples: {condition.condition_id}")
            if not np.all(np.isfinite(time_axis)) or not np.all(np.diff(time_axis) > 0):
                errors.append(f"invalid time axis: {condition.condition_id}")
            if not np.allclose(np.sum(np.stack(components), axis=0), clean, rtol=0, atol=1e-12):
                errors.append(f"component reconstruction mismatch: {condition.condition_id}")
            if not np.allclose(samples, clean[None, :] + noise, rtol=0, atol=1e-12):
                errors.append(f"noise reconstruction mismatch: {condition.condition_id}")
            for index, realization in enumerate(metadata["realizations"]):
                if int(realization["realization"]) != index:
                    errors.append(f"realization index mismatch: {condition.condition_id}/{index}")
                seed = int(realization["seed"])
                if seed in observed_seeds:
                    duplicate_seeds += 1
                observed_seeds.add(seed)
                checksum = hashlib.sha256(samples[index].tobytes()).hexdigest()
                if checksum != realization["checksum_sha256"]:
                    errors.append(f"checksum mismatch: {condition.condition_id}/{index}")
                requested = realization["requested_snr_db"]
                realized = realization["realized_snr_db"]
                if requested is None:
                    if np.any(noise[index] != 0) or realized is not None:
                        errors.append(
                            f"noise-free invariant mismatch: {condition.condition_id}/{index}"
                        )
                else:
                    signal_power = float(np.mean(clean**2))
                    noise_power = float(np.mean(noise[index] ** 2))
                    recomputed = 10 * math.log10(signal_power / noise_power)
                    if not math.isclose(recomputed, float(realized), abs_tol=1e-10):
                        errors.append(f"realized SNR mismatch: {condition.condition_id}/{index}")
                    realized_snr_max_error = max(
                        realized_snr_max_error, abs(float(realized) - float(requested))
                    )
        total_records += condition.noise_realizations
        stage_records[condition.stage] = (
            stage_records.get(condition.stage, 0) + condition.noise_realizations
        )
        stratum_records[condition.stratum] = (
            stratum_records.get(condition.stratum, 0) + condition.noise_realizations
        )
    expected_names = {
        f"{condition.condition_id}{suffix}"
        for condition in conditions
        for suffix in (".npz", ".json")
    }
    expected_names.add("generation_summary.json")
    extras = {path.name for path in data.iterdir() if path.is_file()} - expected_names
    if extras:
        errors.append(f"unexpected files: {sorted(extras)}")
    missing_names = expected_names - {path.name for path in data.iterdir() if path.is_file()}
    if missing_names:
        errors.append(f"missing expected files: {len(missing_names)}")
    if duplicate_seeds:
        errors.append(f"duplicate realization seeds: {duplicate_seeds}")
    generation_summary_path = data / "generation_summary.json"
    if generation_summary_path.exists():
        generation_summary = json.loads(generation_summary_path.read_text(encoding="utf-8"))
        if generation_summary["manifest_sha256"] != manifest_hash:
            errors.append("generation summary manifest hash mismatch")
        if generation_summary["design_sha256"] != generator.design_sha256:
            errors.append("generation summary design hash mismatch")
        if int(generation_summary["conditions_written"]) != len(conditions):
            errors.append("generation summary condition count mismatch")
        if int(generation_summary["realizations_written"]) != total_records:
            errors.append("generation summary realization count mismatch")
    else:
        errors.append("generation summary missing")
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "manifest_sha256": manifest_hash,
        "design_sha256": generator.design_sha256,
        "generator_sha256": generator.generator_sha256,
        "condition_bundles": len(conditions),
        "records": total_records,
        "unique_realization_seeds": len(observed_seeds),
        "duplicate_realization_seeds": duplicate_seeds,
        "stage_records": stage_records,
        "stratum_records": stratum_records,
        "realized_snr_max_absolute_error_db": realized_snr_max_error,
        "stored_bytes": sum(path.stat().st_size for path in data.iterdir() if path.is_file()),
        "errors": errors[:100],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
