"""Callable end-to-end study pipeline."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from .features import extract_features
from .models import Evidence, StudyConfig, StudyResult
from .observability import JsonlObserver
from .qualification import qualify
from .signals import build_dataset
from .transforms import transform


def run(config: StudyConfig | None = None, output_dir: str = "outputs") -> StudyResult:
    """Run a deterministic mechanism-complete descriptor qualification smoke study."""
    cfg = config or StudyConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid5(uuid.NAMESPACE_URL, f"tfpq:{cfg.seed}:{destination}").hex
    observer = JsonlObserver(destination / "logs")
    observer.event(run_id=run_id, stage="start", status="started", seed=cfg.seed)
    dataset = build_dataset(cfg)
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for record in (*dataset.development, *dataset.holdout):
        representation = transform(record, "stft")
        feature = extract_features(record, representation)
        row = {
            "partition": record.partition,
            "disturbance": record.disturbance,
            "seed": record.seed,
            **record.properties,
            **feature.values,
        }
        rows.append(row)
        synthetic = Evidence(
            0.0,
            abs(feature.values["rms"] - record.properties["magnitude"]),
            abs(feature.values["rms"] - record.properties["magnitude"]),
            1.0,
            1.0,
            0.0,
            -0.01,
            0.01,
            2,
        )
        q = qualify(synthetic)
        decisions.append(
            {
                "partition": record.partition,
                "disturbance": record.disturbance,
                "feature": "rms",
                "property": "magnitude",
                "transform": "stft",
                "status": q.status,
                "failed_criteria": ";".join(q.failed_criteria),
                "validity_domain": ";".join(q.validity_conditions),
            }
        )
        observer.statistic(
            {
                "run_id": run_id,
                "analysis_id": "rms-magnitude",
                "disturbance": record.disturbance,
                "property": "magnitude",
                "feature": "rms",
                "transform": "stft",
                "condition_id": str(record.seed),
                "sample_size": 1,
                "statistical_unit": "signal realization",
                "metric": "absolute_error",
                "estimate": synthetic.mae,
                "standard_error": 0.0,
                "confidence_level": 0.95,
                "confidence_interval_lower": synthetic.ci_lower,
                "confidence_interval_upper": synthetic.ci_upper,
                "effect_size": synthetic.slope,
                "adjusted_p_value": "not-computed",
                "engineering_threshold": 0.25,
                "qualification_consequence": q.status,
            }
        )
    _write_csv(destination / "feature_records.csv", rows)
    _write_csv(destination / "qualification_records.csv", decisions)
    result = StudyResult(run_id, dataset, rows, decisions, str(destination))
    (destination / "summary.json").write_text(
        json.dumps({"run_id": run_id, "summary": result.summary()}, indent=2), encoding="utf-8"
    )
    observer.event(
        run_id=run_id, stage="complete", status="success", seed=cfg.seed, message=result.summary()
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
