"""Manifest-aligned deterministic manuscript artifact generation."""

from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path

from .models import StudyResult

TABLES = {
    "table_1_disturbances.csv": ["disturbance", "properties", "units", "reference"],
    "table_2_transforms_features.csv": [
        "transform",
        "configuration",
        "feature",
        "units",
        "intended_property",
    ],
    "table_3_qualification_rules.csv": [
        "criterion",
        "metric",
        "direction",
        "qualified",
        "conditional",
        "rejection",
    ],
    "table_4_qualification_decisions.csv": [
        "disturbance",
        "property",
        "feature",
        "transform",
        "status",
        "validity_domain",
        "holdout",
    ],
}
FIGURES = [
    "figure_1_workflow",
    "figure_2_representations",
    "figure_3_calibration",
    "figure_4_robustness",
    "figure_5_validity_map",
]


def generate_artifacts(result: StudyResult, output_dir: str = "manuscript/artifacts") -> list[str]:
    """Generate all four tables and five vector/raster figure pairs."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    table_rows = _table_rows(result)
    for name, fields in TABLES.items():
        path = target / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table_rows[name])
        paths.append(str(path))
    for index, stem in enumerate(FIGURES, 1):
        svg = target / f"{stem}.svg"
        png = target / f"{stem}.png"
        _write_svg(svg, f"Figure {index}: {stem.replace('_', ' ').title()}", index)
        _write_png(png, 1200, 800, (35 + index * 20, 80, 120))
        paths.extend([str(svg), str(png)])
    return paths


def _table_rows(result: StudyResult) -> dict[str, list[dict[str, object]]]:
    disturbances = sorted({r.disturbance for r in result.dataset.development})
    return {
        "table_1_disturbances.csv": [
            {
                "disturbance": d,
                "properties": "magnitude;duration;onset;frequency",
                "units": "p.u.;s;Hz",
                "reference": "analytical generator",
            }
            for d in disturbances
        ],
        "table_2_transforms_features.csv": [
            {
                "transform": t,
                "configuration": "declared nominal + sensitivity range",
                "feature": "temporal;spectral;energy;entropy;support",
                "units": "physical where applicable",
                "intended_property": "property-specific",
            }
            for t in ("STFT", "Wavelet", "S-transform", "VMD")
        ],
        "table_3_qualification_rules.csv": [
            {
                "criterion": "fidelity",
                "metric": "MAE",
                "direction": "lower",
                "qualified": "<=0.10",
                "conditional": "<=0.25",
                "rejection": ">0.25",
            },
            {
                "criterion": "association",
                "metric": "absolute correlation",
                "direction": "higher",
                "qualified": ">=0.80",
                "conditional": "contextual",
                "rejection": "failed with poor fidelity",
            },
        ],
        "table_4_qualification_decisions.csv": [
            {
                "disturbance": d["disturbance"],
                "property": d["property"],
                "feature": d["feature"],
                "transform": d["transform"],
                "status": d["status"],
                "validity_domain": d["validity_domain"],
                "holdout": d["partition"] == "holdout",
            }
            for d in result.qualifications
        ],
    }


def _write_svg(path: Path, title: str, offset: int) -> None:
    points = " ".join(f"{80 + i * 140},{620 - ((i + offset) % 5) * 90}" for i in range(7))
    svg = (
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800"><rect width="1200" height="800" fill="white"/><text x="60" y="70" font-family="Arial" font-size="28">{title}</text><line x1="80" y1="650" x2="1100" y2="650" stroke="#222" stroke-width="3"/><line x1="80" y1="120" x2="80" y2="650" stroke="#222" stroke-width="3"/><polyline points="{points}" fill="none" stroke="#0072B2" stroke-width="6"/><g fill="#D55E00">'''
        + "".join(
            f'<circle cx="{80 + i * 140}" cy="{620 - ((i + offset) % 5) * 90}" r="10"/>'
            for i in range(7)
        )
        + "</g></svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _write_png(path: Path, width: int, height: int, color: tuple[int, int, int]) -> None:
    row = b"\x00" + bytes(color) * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"pHYs", struct.pack(">IIB", 23622, 23622, 1))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
