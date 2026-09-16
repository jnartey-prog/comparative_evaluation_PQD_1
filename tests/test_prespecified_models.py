from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fit_prespecified_development_models.py"
SPEC = importlib.util.spec_from_file_location("fit_prespecified_models", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holm_adjustment_preserves_order_and_monotonicity() -> None:
    adjusted = MODULE.holm_adjust([0.03, 0.01, 0.20])
    assert adjusted == pytest.approx([0.06, 0.03, 0.20])


def test_holm_adjustment_retains_nonfinite_marker() -> None:
    adjusted = MODULE.holm_adjust([0.01, float("nan")])
    assert adjusted[0] == pytest.approx(0.01)
    assert adjusted[1] != adjusted[1]
