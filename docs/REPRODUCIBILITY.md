# Reproducibility instructions

## Retrieve the published databases

The three feature databases are managed with Git LFS. After cloning the repository, ensure
that Git LFS is installed and materialise the database files:

```bash
git lfs install
git lfs pull
```

## Rapid release verification

Run the read-only release verifier before undertaking a complete reproduction:

```bash
uv sync --all-extras
uv run python scripts/verify_release.py
```

This command verifies the release checksums, SQLite integrity and record allocations, the
32-combination qualification matrix and the reported STFT flicker development and confirmation
metrics. It uses the supplied feature databases and does not regenerate waveforms or repeat
transform extraction.

## Installation and tests

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests scripts
```

The exact dependency resolution used for the release is stored in `uv.lock`.

## Supplied datasets

The repository includes the exact extracted development, formal-repeatability and confirmation feature databases used by the reported analysis. This allows numerical reanalysis without first regenerating the full waveform collection.

The condition manifests retain the controlled parameters, acquisition conditions, partition assignments and deterministic seeds. Waveforms can be regenerated with `scripts/generate_synthetic_data.py` using the supplied configurations and manifests.

## Analysis sequence

1. Validate the synthetic generator and manifests.
2. Extract the representation-specific descriptor records.
3. Evaluate property response using development conditions.
4. Fit and freeze development calibration relationships.
5. Calculate the development fidelity metrics and six numerical-gate decisions.
6. Run the prespecified repeated-condition and bootstrap analyses.
7. Apply the frozen calibration functions and decision rules to the sealed confirmation partition without refitting.

Representative commands are:

```bash
uv run python scripts/audit_synthetic_generator.py
uv run python scripts/audit_generated_dataset.py
uv run python scripts/analyze_property_response.py
uv run python scripts/analyze_formal_development.py
uv run python scripts/complete_development_gates.py
uv run python scripts/analyze_confirmation_holdout.py
uv run python scripts/audit_qualification_matrix.py
```

Some analysis scripts expose additional command-line arguments. Run a script with `--help` before execution and retain the supplied files as the immutable reference release.

## Expected confirmation results

- Expected and observed feature rows: 318,400.
- Records per representation: 79,600.
- Computational extraction failures: 0.

## Expected primary decision

The primary qualification matrix contains 32 disturbance-representation combinations. Only the STFT envelope-modulation-frequency descriptor for flicker meets all six evaluated development-stage numerical gates. Confirmation retention is recorded separately and is not treated as a seventh gate.

## Integrity

`FILE_MANIFEST_SHA256.csv` contains a path, byte count and SHA-256 digest for every release file. Verify the archive after download before conducting a reproduction run.
