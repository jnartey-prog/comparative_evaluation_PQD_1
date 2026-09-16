# TFPQ descriptor qualification

This repository contains the Python software, experimental specifications and processed records required to reproduce the computational workflow for **Comparative Qualification of Selected Time-Frequency Descriptors for Power-Quality Disturbance Properties**.

The workflow generates deterministic synthetic power-quality waveforms, applies four time-frequency representations, extracts property-specific descriptors and executes the prespecified development and confirmation analyses.

## Contents

- `src/tfpq_qualifier/`: waveform, transform, descriptor and qualification software.
- `scripts/`: waveform-generation, extraction, analysis, bootstrap, verification and audit scripts.
- `tests/`: automated tests for the core mechanisms and analysis rules.
- `configs/`: scientific design and decision policies used by the Python workflow.
- `manifests/`: development, formal-repeatability and sealed-confirmation condition allocations.
- `outputs/formal_analysis/`: development record-level metrics, summaries and calibration results.
- `outputs/`: exact development, formal-repeatability and confirmation feature databases, release audits and analysis outputs.
- `outputs/holdout_analysis/`: frozen confirmation summaries and audits.
- `outputs/qualification_matrix/`: primary qualification matrix and compound-event status.
- `docs/`: data dictionary, methods mapping and detailed reproduction instructions.

## Environment

Python 3.11-3.13 is supported. The locked environment is defined by `pyproject.toml` and `uv.lock`.

```bash
uv sync --all-extras
uv run pytest
```

See `docs/REPRODUCIBILITY.md` for the analysis sequence and expected inputs and outputs.

## Data size

The release includes the three SQLite feature databases required for direct numerical reanalysis. The waveform collection itself is not included; it can be regenerated from the supplied Python software, configurations, manifests and seed definitions.

## Scope

The repository is limited to reproduction of the supplied controlled synthetic experiment and its declared configurations.

## Citation

Use the metadata in `CITATION.cff`. Add the repository DOI after deposition and cite the archived release in the manuscript's reference list and Data Availability statement.

## Licence

The software is distributed under the MIT License in `LICENSE`. Select and record an appropriate data licence when depositing the archive in a DOI-bearing repository.
