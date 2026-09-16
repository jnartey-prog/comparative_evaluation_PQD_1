"""Independent audit of development transform extraction checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from tfpq_qualifier.synthetic import SyntheticDataGenerator, load_conditions

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "outputs/development_features.sqlite"
METHODS = {"stft", "wavelet", "s_transform", "vmd"}
FEATURES = {
    "tf_dominant_frequency_hz",
    "tf_frequency_centroid_hz",
    "tf_frequency_spread_hz",
    "tf_onset_s",
    "tf_support_duration_s",
    "tf_energy",
    "tf_entropy",
    "tf_concentration",
    "fundamental_baseline_amplitude",
    "fundamental_deviation_peak",
    "fundamental_deviation_onset_s",
    "fundamental_deviation_duration_s",
    "high_band_peak_frequency_hz",
    "high_band_energy_fraction",
    "high_band_onset_s",
    "high_band_duration_s",
    "envelope_modulation_frequency_hz",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="outputs/development_features.sqlite")
    parser.add_argument("--manifest", default="manifests/development_conditions.csv")
    parser.add_argument("--data", default="outputs/synthetic/development")
    parser.add_argument("--output", default="outputs/development_extraction_audit.json")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    args = parser.parse_args()
    selected_methods = set(args.methods)
    database = ROOT / args.database
    data = ROOT / args.data
    generator = SyntheticDataGenerator(ROOT / "configs/scientific_design.json")
    manifest_path = ROOT / args.manifest
    if "formal_repeatability" in manifest_path.name.lower():
        lock = json.loads((ROOT / "manifests/PUBLIC_RELEASE_LOCK.json").read_text(encoding="utf-8"))
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_hash != lock["formal_repeatability_manifest_sha256"]:
            raise ValueError("formal-repeatability manifest hash mismatch")
    else:
        manifest_hash = generator.verify_manifest(manifest_path)
    manifest = list(load_conditions(manifest_path))
    selected = list(manifest)
    condition_map = {c.condition_id: c for c in selected}
    expected_keys = {
        (condition.condition_id, realization, method)
        for condition in selected
        for realization in range(condition.noise_realizations)
        for method in selected_methods
    }
    errors: list[str] = []
    warnings: list[str] = []
    actual_keys: set[tuple[str, int, str]] = set()
    method_rows = {method: 0 for method in selected_methods}
    stage_rows: dict[str, int] = {}
    method_runtime = {method: 0.0 for method in selected_methods}
    method_failures = {method: 0 for method in selected_methods}
    nonfinite_counts = {method: 0 for method in selected_methods}
    undefined_high_band_peak = {method: 0 for method in selected_methods}
    feature_key_mismatches = 0
    with sqlite3.connect(database) as connection:
        database_integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if database_integrity != "ok":
            errors.append(f"database integrity failure: {database_integrity}")
        duplicate_count = connection.execute(
            "SELECT COUNT(*) FROM (SELECT condition_id,realization,method,COUNT(*) n FROM feature_results GROUP BY condition_id,realization,method HAVING n>1)"
        ).fetchone()[0]
        if duplicate_count:
            errors.append(f"duplicate primary keys: {duplicate_count}")
        placeholders = ",".join("?" for _ in selected_methods)
        rows = connection.execute(
            "SELECT condition_id,realization,stage,stratum,method,runtime_s,failure,"
            "failure_message,features_json,truth_json,requested_snr_db,realized_snr_db,seed "
            f"FROM feature_results WHERE method IN ({placeholders})",
            sorted(selected_methods),
        )
        for row in rows:
            (
                condition_id,
                realization,
                stage,
                stratum,
                method,
                runtime,
                failure,
                message,
                features_json,
                truth_json,
                requested_snr,
                realized_snr,
                seed,
            ) = row
            if condition_id not in condition_map:
                errors.append(f"unexpected or non-audited condition row: {condition_id}")
                continue
            condition = condition_map[condition_id]
            key = (condition_id, int(realization), method)
            actual_keys.add(key)
            if method not in selected_methods or stage != condition.stage or stratum != condition.stratum:
                errors.append(f"identity mismatch: {key}")
            if not 0 <= int(realization) < condition.noise_realizations:
                errors.append(f"realization outside allocation: {key}")
            metadata = json.loads((data / f"{condition_id}.json").read_text(encoding="utf-8"))
            source = metadata["realizations"][int(realization)]
            if int(seed) != int(source["seed"]):
                errors.append(f"seed mismatch: {key}")
            if (
                requested_snr != source["requested_snr_db"]
                or realized_snr != source["realized_snr_db"]
            ):
                errors.append(f"SNR provenance mismatch: {key}")
            if json.loads(truth_json) != metadata["condition"]["truth"]:
                errors.append(f"truth mismatch: {key}")
            method_rows[method] += 1
            stage_rows[stage] = stage_rows.get(stage, 0) + 1
            method_runtime[method] += float(runtime)
            method_failures[method] += int(failure)
            if not math.isfinite(float(runtime)) or runtime < 0:
                errors.append(f"invalid runtime: {key}")
            if failure:
                if not message or features_json != "{}":
                    errors.append(f"malformed failure row: {key}")
                continue
            features: dict[str, Any] = json.loads(features_json)
            if set(features) != FEATURES:
                feature_key_mismatches += 1
            for name, value in features.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"non-numeric feature {name}: {key}")
                elif not math.isfinite(float(value)) and name == "high_band_peak_frequency_hz":
                    if float(features["high_band_energy_fraction"]) != 0.0:
                        nonfinite_counts[method] += 1
                    else:
                        undefined_high_band_peak[method] += 1
                elif not math.isfinite(float(value)) and name not in {
                    "fundamental_deviation_onset_s",
                    "high_band_onset_s",
                    "envelope_modulation_frequency_hz",
                }:
                    nonfinite_counts[method] += 1
            for name in (
                "tf_dominant_frequency_hz",
                "tf_frequency_centroid_hz",
                "high_band_peak_frequency_hz",
            ):
                value = features[name]
                if math.isfinite(float(value)) and not 0 <= float(value) <= 3000.000001:
                    errors.append(f"frequency support violation {name}: {key}")
            if not 0 <= float(features["high_band_energy_fraction"]) <= 1.0000001:
                errors.append(f"energy fraction outside [0,1]: {key}")
    missing = expected_keys - actual_keys
    extras = actual_keys - expected_keys
    if missing:
        errors.append(f"missing expected rows: {len(missing)}")
    if extras:
        errors.append(f"unexpected rows: {len(extras)}")
    if feature_key_mismatches:
        errors.append(f"feature schema mismatches: {feature_key_mismatches}")
    if any(nonfinite_counts.values()):
        errors.append(f"non-permitted non-finite feature values: {nonfinite_counts}")
    if any(method_failures.values()):
        warnings.append(f"transform failures are scientific outcomes: {method_failures}")
    if any(undefined_high_band_peak.values()):
        warnings.append(
            "undefined high-band peak denotes no transform component above 125 Hz: "
            f"{undefined_high_band_peak}"
        )
    database_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "manifest_sha256": manifest_hash,
        "database_sha256": database_hash,
        "database_integrity": database_integrity,
        "conditions_expected": len(selected),
        "expected_rows": len(expected_keys),
        "actual_rows": len(actual_keys),
        "stage_rows": stage_rows,
        "method_rows": method_rows,
        "method_failures": method_failures,
        "method_runtime_seconds": method_runtime,
        "method_runtime_hours": {
            method: seconds / 3600 for method, seconds in method_runtime.items()
        },
        "nonfinite_nonpermitted_counts": nonfinite_counts,
        "undefined_high_band_peak_counts": undefined_high_band_peak,
        "confirmation_rows": sum(
            1 for condition_id, _, _ in actual_keys if condition_id.startswith("CON-")
        ),
        "errors": errors[:100],
        "warnings": warnings,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
