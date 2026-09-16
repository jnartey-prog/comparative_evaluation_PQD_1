"""Resumable parallel extraction of nominal transform features from development bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from tfpq_qualifier.models import SignalRecord
from tfpq_qualifier.representation_features import extract_representation_features
from tfpq_qualifier.synthetic import ManifestCondition, SyntheticDataGenerator, load_conditions
from tfpq_qualifier.transforms import transform

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("stft", "wavelet", "s_transform", "vmd")


def method_config(method: str, design: dict[str, Any]) -> dict[str, float | str]:
    common: dict[str, float | str] = {"frequency_min": 10.0, "frequency_max": 3000.0}
    if method == "stft":
        common.update({"window_name": "hann"})
    elif method == "wavelet":
        common.update({"wavelet": "cmor1.5-1.0", "frequency_bins": 128.0})
    elif method == "s_transform":
        common.update(
            {
                "frequency_bins": float(design["transforms"]["s_transform"]["frequency_bins"]),
                "gaussian_width_factor": 1.0,
            }
        )
    else:
        vmd = design["transforms"]["vmd"]
        common.update(
            {
                "modes": float(vmd["K"]),
                "alpha": float(vmd["alpha"]),
                "tau": float(vmd["tau"]),
                "dc": float(vmd["dc"]),
                "tolerance": float(vmd["tolerance"]),
            }
        )
    return common


def analyze_chunk(
    payload: tuple[ManifestCondition, tuple[tuple[int, tuple[str, ...]], ...], str, str],
) -> list[tuple[Any, ...]]:
    condition, work_items, design_path, data_path = payload
    design = json.loads(Path(design_path).read_text(encoding="utf-8"))
    data = Path(data_path)
    metadata = json.loads((data / f"{condition.condition_id}.json").read_text(encoding="utf-8"))
    rows: list[tuple[Any, ...]] = []
    with np.load(data / f"{condition.condition_id}.npz") as arrays:
        time_axis = arrays["time_s"]
        samples = arrays["samples_pu"]
        for realization_index, methods in work_items:
            realization = metadata["realizations"][realization_index]
            record = SignalRecord(
                samples[realization_index],
                time_axis,
                float(metadata["condition"]["sample_rate_hz"]),
                condition.stratum,
                dict(metadata["condition"]["truth"]),
                {"fundamental_frequency": 50.0},
                condition.partition,
                int(realization["seed"]),
            )
            for method in methods:
                started = time.perf_counter()
                try:
                    representation = transform(record, method, method_config(method, design))
                    features = extract_representation_features(representation)
                    failure, message = 0, ""
                except Exception as exc:  # noqa: BLE001 - failure is a prespecified outcome
                    features, failure, message = {}, 1, f"{type(exc).__name__}: {exc}"
                rows.append(
                    (
                        condition.condition_id,
                        realization_index,
                        condition.stage,
                        condition.stratum,
                        method,
                        time.perf_counter() - started,
                        failure,
                        message,
                        json.dumps(features, sort_keys=True, separators=(",", ":")),
                        json.dumps(
                            metadata["condition"]["truth"], sort_keys=True, separators=(",", ":")
                        ),
                        realization["requested_snr_db"],
                        realization["realized_snr_db"],
                        int(realization["seed"]),
                    )
                )
    return rows


def initialize(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS feature_results (
        condition_id TEXT NOT NULL, realization INTEGER NOT NULL, stage TEXT NOT NULL,
        stratum TEXT NOT NULL, method TEXT NOT NULL, runtime_s REAL NOT NULL,
        failure INTEGER NOT NULL, failure_message TEXT NOT NULL, features_json TEXT NOT NULL,
        truth_json TEXT NOT NULL, requested_snr_db REAL, realized_snr_db REAL, seed INTEGER NOT NULL,
        PRIMARY KEY (condition_id, realization, method))""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry only durable rows previously recorded with failure=1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=(os.cpu_count() or 1),
        help="Worker processes (defaults to all visible CPU cores)",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "property_response",
            "broad_nuisance_screen",
            "compound_screen",
            "formal_repeatability",
            "sealed_confirmation",
        ),
    )
    parser.add_argument("--max-conditions", type=int)
    parser.add_argument("--stratum", help="Process only one manifest stratum")
    parser.add_argument("--max-chunks", type=int, help="Process only the first N pending chunks")
    parser.add_argument(
        "--chunk-realizations",
        type=int,
        default=100,
        help="Maximum realizations per atomic worker result and database commit",
    )
    parser.add_argument("--manifest", default="manifests/development_conditions.csv")
    parser.add_argument("--data", default="outputs/synthetic/development")
    parser.add_argument("--database", default="outputs/development_features.sqlite")
    args = parser.parse_args()
    if args.chunk_realizations < 1:
        parser.error("--chunk-realizations must be at least 1")
    generator = SyntheticDataGenerator(ROOT / "configs/scientific_design.json")
    manifest = ROOT / args.manifest
    if "formal_repeatability" in manifest.name.lower():
        lock = json.loads((ROOT / "manifests/PUBLIC_RELEASE_LOCK.json").read_text(encoding="utf-8"))
        if (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            != lock["formal_repeatability_manifest_sha256"]
        ):
            raise ValueError("formal-repeatability manifest hash mismatch")
    else:
        generator.verify_manifest(manifest)
    conditions = [
        condition
        for condition in load_conditions(manifest)
        if (args.stage is None or condition.stage == args.stage)
        and (args.stratum is None or condition.stratum == args.stratum)
    ]
    if args.max_conditions is not None:
        conditions = conditions[: args.max_conditions]
    database = ROOT / args.database
    database.parent.mkdir(exist_ok=True)
    with sqlite3.connect(database) as connection:
        initialize(connection)
        placeholders = ",".join("?" for _ in args.methods)
        existing = {
            (condition_id, int(realization), method)
            for condition_id, realization, method in connection.execute(
                f"SELECT condition_id,realization,method FROM feature_results WHERE method IN ({placeholders})",
                args.methods,
            )
        }
        if args.retry_failures:
            failed = {
                (condition_id, int(realization), method)
                for condition_id, realization, method in connection.execute(
                    f"SELECT condition_id,realization,method FROM feature_results "
                    f"WHERE failure=1 AND method IN ({placeholders})",
                    args.methods,
                )
            }
            # Treat failed keys as pending, while retaining their durable
            # failure records until a successful retry replaces the key.
            existing.difference_update(failed)
        completed = {
            condition.condition_id
            for condition in conditions
            if all(
                (condition.condition_id, realization, method) in existing
                for realization in range(condition.noise_realizations)
                for method in args.methods
            )
        }
        pending = [condition for condition in conditions if condition.condition_id not in completed]
        payloads = []
        for condition in pending:
            # Build work from exact missing primary keys. A partially completed
            # realization is sent only with its missing methods; completed rows
            # are never recomputed or overwritten.
            missing_items = [
                (
                    realization,
                    tuple(
                        method
                        for method in args.methods
                        if (condition.condition_id, realization, method) not in existing
                    ),
                )
                for realization in range(condition.noise_realizations)
            ]
            missing_items = [item for item in missing_items if item[1]]
            for start in range(0, len(missing_items), args.chunk_realizations):
                payloads.append(
                    (
                        condition,
                        tuple(missing_items[start : start + args.chunk_realizations]),
                        str(ROOT / "configs/scientific_design.json"),
                        str(ROOT / args.data),
                    )
                )
        if args.max_chunks is not None:
            payloads = payloads[: args.max_chunks]
        # Workers never touch SQLite.  The parent process is the sole writer,
        # and commits in bounded batches to minimize lock duration while
        # retaining resumability after interruption.
        written = 0
        write_buffer: list[tuple[Any, ...]] = []

        def flush_buffer() -> None:
            nonlocal written
            if not write_buffer:
                return
            connection.executemany(
                "INSERT OR REPLACE INTO feature_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                write_buffer,
            )
            connection.commit()
            written += len(write_buffer)
            write_buffer.clear()

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(analyze_chunk, payload) for payload in payloads]
            for future in as_completed(futures):
                rows = future.result()
                write_buffer.extend(rows)
                while len(write_buffer) >= 500:
                    batch = write_buffer[:500]
                    del write_buffer[:500]
                    connection.executemany(
                        "INSERT OR REPLACE INTO feature_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    connection.commit()
                    written += len(batch)
            flush_buffer()
        summary = {
            "database": str(database),
            "conditions_requested": len(conditions),
            "conditions_previously_complete": len(completed),
            "conditions_processed": len(pending),
            "chunks_processed": len(payloads),
            "chunk_realizations": args.chunk_realizations,
            "rows_written": written,
            "methods": args.methods,
            "stage": args.stage,
            "stratum": args.stratum,
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
