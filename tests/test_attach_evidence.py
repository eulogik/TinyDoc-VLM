"""Unit tests for evidence replay on saved predictions (no OCR, no model)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

import attach_evidence_from_preds as replay
from tinydoc.evidence import Evidence


def _row():
    return {
        "id": "doc1",
        "image_path": "unused.jpg",
        "pred": {
            "company": "ACME",
            "date": "01/02/2026",
            "address": "1 MAIN STREET 50000 KUALA LUMPUR",
            "total": "12.50",
        },
        "schema_valid_pipeline": True,
        "confidence": 0.4025,
        "evidence_coverage": 0.0,
    }


def _fake_words(monkeypatch):
    monkeypatch.setattr(replay, "ocr_words", lambda path: ([{"text": "ACME"}], "rapidocr"))


def test_replay_with_full_evidence(monkeypatch):
    _fake_words(monkeypatch)

    def fake_locate(value, image_path, words=None):
        return Evidence(kind="ocr_span", quote=value, bbox=[0, 0, 1, 1], page=1, score=1.0)

    monkeypatch.setattr(replay, "locate_field_evidence", fake_locate)
    out = replay.enrich_row(_row())

    assert out["id"] == "doc1"
    assert out["evidence_engine"] == "rapidocr"
    assert out["evidence_coverage"] == 1.0
    assert out["field_confidences"]["company"] == 0.9
    assert out["field_confidences"]["total"] == 0.95
    assert out["confidence"] == 0.925
    assert out["field_evidence"]["company"]["kind"] == "ocr_span"


def test_replay_without_evidence_reproduce_constant(monkeypatch):
    monkeypatch.setattr(replay, "ocr_words", lambda path: ([], "tesseract"))
    monkeypatch.setattr(replay, "locate_field_evidence", lambda *a, **k: None)
    out = replay.enrich_row(_row())

    assert out["evidence_engine"] == "tesseract"
    assert out["evidence_coverage"] == 0.0
    assert out["field_confidences"]["company"] == 0.55
    assert out["confidence"] == 0.4025


def test_replay_survives_ocr_failure(monkeypatch):
    def broken(path):
        raise RuntimeError("ocr unavailable")

    monkeypatch.setattr(replay, "ocr_words", broken)
    monkeypatch.setattr(replay, "locate_field_evidence", lambda *a, **k: None)
    out = replay.enrich_row(_row())

    assert out["evidence_error"] == "RuntimeError: ocr unavailable"
    assert out["evidence_engine"] == "none"
    assert out["confidence"] == 0.4025
    assert out["evidence_coverage"] == 0.0


def test_replay_handles_schema_invalid(monkeypatch):
    _fake_words(monkeypatch)

    def fake_locate(value, image_path, words=None):
        return Evidence(kind="ocr_span", quote=value, bbox=None, page=1, score=0.5)

    row = _row()
    row["schema_valid_pipeline"] = False
    monkeypatch.setattr(replay, "locate_field_evidence", fake_locate)
    out = replay.enrich_row(row)

    # 0.35 + 0.55*0.5 = 0.625 for company/address; +0.05 for date/total = 0.675.
    assert out["confidence"] == round(0.65 * 0.75, 4)
    assert out["field_confidences"]["address"] == 0.625
