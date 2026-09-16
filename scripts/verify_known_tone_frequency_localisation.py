"""Audit 50-Hz localisation for the four nominal Figure 2 representations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tfpq_qualifier.models import SignalRecord
from tfpq_qualifier.transforms import transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "verification"


def main() -> int:
    sample_rate = 12_800.0
    duration = 1.0
    tone_frequency = 50.0
    tolerance = 5.0
    time = np.arange(0.0, duration, 1.0 / sample_rate)
    samples = np.sin(2.0 * np.pi * tone_frequency * time)
    record = SignalRecord(
        samples=samples, time=time, sample_rate=sample_rate,
        disturbance="unit_tone", properties={"frequency_hz": tone_frequency},
        conditions={"duration_s": duration}, partition="test", seed=0,
    )
    configurations = {
        "STFT": ("stft", {"window_name": "hann", "window": 256, "hop": 64}),
        "CWT": ("wavelet", {"wavelet": "cmor1.5-1.0", "frequency_bins": 128,
                              "frequency_min": 10.0, "frequency_max": 3000.0}),
        "S-transform": ("s_transform", {"frequency_bins": 128}),
        "VMD": ("vmd", {"modes": 5, "alpha": 2000, "tau": 0, "dc": 0,
                          "init": 1, "tolerance": 1e-7}),
    }

    results = []
    for label, (method, configuration) in configurations.items():
        representation = transform(record, method, configuration)
        magnitude = np.abs(np.asarray(representation.values))
        if label == "CWT":
            # Preserve the recovery rule used by the existing audited CWT test.
            peak_row = int(np.unravel_index(np.argmax(magnitude), magnitude.shape)[0])
            recovery_rule = "frequency row containing the maximum-magnitude coefficient"
        else:
            # Suppress isolated boundary maxima for gridded transforms and
            # select the dominant-energy VMD mode without using the known truth.
            peak_row = int(np.argmax(np.mean(magnitude * magnitude, axis=1)))
            recovery_rule = (
                "final centre frequency of the maximum-energy mode"
                if label == "VMD"
                else "frequency row with maximum time-averaged squared magnitude"
            )
        recovered = float(np.asarray(representation.frequencies)[peak_row])
        error = abs(recovered - tone_frequency)
        results.append({
            "representation": label,
            "input_frequency_hz": tone_frequency,
            "recovered_frequency_hz": recovered,
            "absolute_error_hz": error,
            "acceptance_tolerance_hz": tolerance,
            "status": "PASS" if error <= tolerance else "FAIL",
            "recovery_rule": recovery_rule,
            "configuration": configuration,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_signal": {"waveform": "sin(2*pi*50*t)",
                        "sampling_rate_hz": sample_rate, "duration_s": duration},
        "all_pass": all(row["status"] == "PASS" for row in results),
        "results": results,
    }
    (OUT / "known_tone_frequency_localisation_all_transforms.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "known_tone_frequency_localisation_all_transforms.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["representation", "input_frequency_hz", "recovered_frequency_hz",
                      "absolute_error_hz", "acceptance_tolerance_hz", "status",
                      "recovery_rule"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
