"""Redacted structured JSONL persistence."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _redact(value: Any) -> Any:
    text = str(value)
    text = re.sub(r"(?i)(token|password|secret)=\S+", r"\1=[REDACTED]", text)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\]+", "[HOME]", text)
    return text


class JsonlObserver:
    """Append schema-controlled run and statistics events."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_path = self.directory / "run.jsonl"
        self.statistics_path = self.directory / "statistics.jsonl"

    def event(
        self,
        *,
        run_id: str,
        stage: str,
        status: str,
        task_id: str = "PIPELINE.CORE.001",
        seed: int = 0,
        message: str = "",
        duration_ms: int = 0,
    ) -> None:
        record = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "task_id": task_id,
            "dataset_partition": "all",
            "configuration_hash": "runtime",
            "random_seed": seed,
            "status": status,
            "duration_ms": duration_ms,
            "warning_count": 0,
            "error_type": "",
            "message": _redact(message),
        }
        self._append(self.run_path, record)

    def statistic(self, record: dict[str, Any]) -> None:
        self._append(self.statistics_path, {k: _redact(v) for k, v in record.items()})

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
