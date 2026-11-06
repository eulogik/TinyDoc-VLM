"""Unit tests for the confidence probe (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from run_confidence_probe import (
    aurc,
    brier,
    ece,
    probe,
    top_band_precision,
)


def test_perfect_confidence():
    conf = [1.0, 1.0, 1.0]
    ok = [True, True, True]
    assert ece(conf, ok) == 0.0
    assert brier(conf, ok) == 0.0
    assert aurc(conf, ok) == 0.0
    assert top_band_precision(conf, ok) == 1.0


def test_overconfident_wrong():
    conf = [0.9, 0.9]
    ok = [False, False]
    assert abs(ece(conf, ok) - 0.9) < 1e-9
    assert abs(brier(conf, ok) - 0.81) < 1e-9
    assert abs(aurc(conf, ok) - 0.75) < 1e-9
    assert top_band_precision(conf, ok) == 0.0


def test_mixed_ordering():
    conf = [0.9, 0.1]
    ok = [True, False]
    assert abs(ece(conf, ok) - 0.1) < 1e-9
    assert abs(brier(conf, ok) - 0.01) < 1e-9
    assert abs(aurc(conf, ok) - 0.125) < 1e-9
    assert top_band_precision(conf, ok, 0.5) == 1.0


def test_probe_falls_back_to_doc_confidence():
    row = {
        "confidence": 0.4,
        "score": {
            "fields": {
                "company": {"hit": True},
                "date": {"hit": False},
                "address": {"hit": False},
                "total": {"hit": True},
            }
        },
    }
    report = probe([row])
    assert report["used_field_confidences"] is False
    assert report["n_field_rows"] == 4
    assert report["field_level"]["mean_confidence"] == 0.4
    assert report["doc_exact4"]["accuracy"] == 0.0


def test_probe_prefers_field_confidences():
    row = {
        "confidence": 0.4,
        "field_confidences": {"company": 0.9, "date": 0.2, "address": 0.3, "total": 0.8},
        "score": {
            "fields": {
                "company": {"hit": True},
                "date": {"hit": False},
                "address": {"hit": False},
                "total": {"hit": True},
            }
        },
    }
    report = probe([row])
    assert report["used_field_confidences"] is True
    assert abs(report["field_level"]["mean_confidence"] - 0.55) < 1e-9
