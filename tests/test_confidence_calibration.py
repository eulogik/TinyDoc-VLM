"""Unit tests for grouped isotonic calibration (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from run_confidence_calibration import (
    calibration_report,
    cross_val_predict,
    fit_isotonic,
    grouped_folds,
    predict_isotonic,
)


def test_isotonic_is_monotone():
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ys = [0, 1, 0, 1, 0, 1]
    knots = fit_isotonic(xs, ys)
    values = [y for _, y in knots]
    assert values == sorted(values)
    assert all(0.0 <= y <= 1.0 for y in values)


def test_isotonic_preserves_monotone_signal():
    knots = fit_isotonic([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1])
    assert knots == [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 1.0)]
    assert predict_isotonic(knots, 2.5) == 1.0
    assert predict_isotonic(knots, -1.0) == 0.0
    assert predict_isotonic(knots, 10.0) == 1.0


def test_isotonic_pools_violators():
    # The 1 then 0 pair violates monotonicity and must pool to mean 0.5.
    knots = fit_isotonic([0.0, 0.5, 1.0], [0, 1, 0])
    assert len(knots) == 2
    assert knots[-1][1] == 0.5


def test_grouped_folds_keep_documents_together():
    ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    folds = grouped_folds(ids, k=2)
    for doc_id in set(ids):
        assigned = {f for i, f in zip(ids, folds) if i == doc_id}
        assert len(assigned) == 1
    assert len(set(folds)) == 2
    assert folds == grouped_folds(ids, k=2)


def test_cross_val_separates_overconfident_groups():
    ids = [f"doc{i // 4}" for i in range(40)]
    xs = [0.9 if i % 2 == 0 else 0.1 for i in range(40)]
    ys = [i % 2 == 0 for i in range(40)]
    calibrated, folds = cross_val_predict(xs, ys, ids, k=5)
    assert folds == 5
    assert all(0.0 <= c <= 1.0 for c in calibrated)
    # Perfectly separable signal should stay separated after calibration.
    assert all((c > 0.5) == y for c, y in zip(calibrated, ys))


def _synthetic_rows(n_docs: int = 20) -> list[dict]:
    rows = []
    for d in range(n_docs):
        hit = d < n_docs // 2
        fields = {
            name: {"hit": hit, "pred": "x", "gold": "x", "anls": 1.0 if hit else 0.0}
            for name in ("company", "date", "address", "total")
        }
        conf = 0.9 if hit else 0.1
        rows.append(
            {
                "id": f"doc{d:03d}",
                "confidence": conf,
                "field_confidences": {name: conf for name in fields},
                "score": {"fields": fields},
            }
        )
    return rows


def test_calibration_report_reduces_ece():
    report = calibration_report(_synthetic_rows(), k=5)
    raw = report["fields"]["raw"]
    calibrated = report["fields"]["isotonic_oof"]
    assert raw["ece_10bin"] > 0.05
    assert calibrated["ece_10bin"] < raw["ece_10bin"]
    assert calibrated["ece_10bin"] < 0.02
    assert calibrated["brier"] < raw["brier"]
    assert report["n_field_rows"] == 80
    assert report["n_docs"] == 20
    assert report["n_docs_with_field_confidences"] == 20
    assert report["constant_confidence"] is False
    assert set(report["by_field"]) == {"company", "date", "address", "total"}
    for metrics in report["by_field"].values():
        assert metrics["isotonic_oof"]["ece_10bin"] <= metrics["raw"]["ece_10bin"]
