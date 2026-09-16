"""Verify CWT scale-to-frequency localisation with a known 50-Hz tone."""
from pathlib import Path
import json
import numpy as np
from tfpq_qualifier.models import SignalRecord
from tfpq_qualifier.transforms import transform

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    fs = 12800.0
    tone_hz = 50.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    x = np.sin(2.0 * np.pi * tone_hz * t)
    rec = SignalRecord(samples=x, time=t, sample_rate=fs, disturbance="unit_tone", properties={}, conditions={}, partition="test", seed=0)
    rep = transform(rec, "wavelet", {"wavelet": "cmor1.5-1.0", "frequency_bins": 128, "frequency_min": 10.0, "frequency_max": 3000.0})
    magnitude = np.abs(np.asarray(rep.values))
    peak_frequency = float(np.asarray(rep.frequencies)[np.unravel_index(np.argmax(magnitude), magnitude.shape)[0]])
    error_hz = abs(peak_frequency - tone_hz)
    result = {"status": "PASS" if error_hz <= 5.0 else "FAIL", "tone_frequency_hz": tone_hz, "detected_peak_frequency_hz": peak_frequency, "absolute_error_hz": error_hz, "acceptance_tolerance_hz": 5.0, "wavelet": "cmor1.5-1.0", "frequency_bins": 128, "sampling_rate_hz": fs}
    out = ROOT / "outputs/verification/cwt_frequency_localisation_unit_test.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
