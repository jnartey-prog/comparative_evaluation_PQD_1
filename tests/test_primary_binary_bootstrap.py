from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_primary_binary_cluster_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("primary_binary_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_bootstrap_retains_condition_as_resampling_unit() -> None:
    rows = []
    methods = ("stft", "wavelet", "s_transform", "vmd")
    for condition in ("C1", "C2", "C3", "C4"):
        for method_index, method in enumerate(methods):
            for realization in range(3):
                rows.append(
                    {
                        "condition_id": condition,
                        "method": method,
                        "pass": int(method_index == 0 or realization == 0),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["method"] = pd.Categorical(frame["method"], categories=methods)
    marginals, contrasts = MODULE.bootstrap_family(
        frame, "pass", resamples=200, seed=9
    )
    assert len(marginals) == 4
    assert len(contrasts) == 6
    assert all(row["condition_n"] == 4 for row in marginals + contrasts)
    assert marginals[0]["marginal_probability"] == pytest.approx(1.0)
    assert marginals[1]["marginal_probability"] == pytest.approx(1 / 3)


def test_bootstrap_rejects_unpaired_methods() -> None:
    frame = pd.DataFrame(
        {
            "condition_id": ["C1", "C1", "C1"],
            "method": pd.Categorical(
                ["stft", "wavelet", "s_transform"], categories=MODULE.METHODS
            ),
            "pass": [1, 1, 0],
        }
    )
    with pytest.raises(ValueError, match="every method"):
        MODULE.bootstrap_family(frame, "pass", resamples=100, seed=1)
