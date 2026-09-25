"""Unit tests for evidence anchoring: locate/attach, engine routing, to_dict.

No OCR binaries or images required — words are injected, engine selection is
monkeypatched. These cover the evidence layer that README/CLI advertise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from tinydoc import evidence as ev_mod
from tinydoc.evidence import (
    Evidence,
    attach_evidence,
    locate_field_evidence,
)


def _word(text: str, left: float, top: float = 10.0, width: float = 40.0, height: float = 12.0):
    return {"text": text, "left": float(left), "top": float(top), "width": float(width), "height": float(height)}


@pytest.fixture
def receipt_words():
    return [
        _word("GARDENIA", 10),
        _word("BAKERIES", 80),
        _word("(KL)", 160),
        _word("SDN", 200),
        _word("BHD", 240),
        _word("22/10/2017", 10, top=40.0),
        _word("TOTAL", 10, top=70.0),
        _word("34.21", 120, top=70.0, width=30.0),
    ]


class TestLocate:
    def test_exact_multitoken_match(self, receipt_words):
        ev = locate_field_evidence(
            "GARDENIA BAKERIES (KL) SDN BHD", "unused.png", words=receipt_words
        )
        assert isinstance(ev, Evidence)
        assert ev.kind == "ocr_span"
        assert ev.page == 1
        assert ev.score > 0.9
        # bbox unions exactly the five matched words: x0=10..x1=240+40, y=10..22
        assert ev.bbox == [10.0, 10.0, 280.0, 22.0]
        assert "GARDENIA" in ev.quote

    def test_single_token_total(self, receipt_words):
        ev = locate_field_evidence("34.21", "unused.png", words=receipt_words)
        assert ev is not None
        assert ev.bbox == [120.0, 70.0, 150.0, 82.0]
        assert ev.score >= 0.34

    def test_currency_normalization(self, receipt_words):
        ev = locate_field_evidence("$34.21", "unused.png", words=receipt_words)
        assert ev is not None
        assert "34.21" in ev.quote

    def test_no_match_returns_none(self, receipt_words):
        assert locate_field_evidence("QQQZZZ", "unused.png", words=receipt_words) is None

    def test_empty_value_returns_none(self, receipt_words):
        assert locate_field_evidence("", "unused.png", words=receipt_words) is None
        assert locate_field_evidence("   ", "unused.png", words=receipt_words) is None

    def test_no_words_returns_none(self):
        assert locate_field_evidence("ACME", "unused.png", words=[]) is None

    def test_to_dict_roundtrip(self, receipt_words):
        ev = locate_field_evidence("34.21", "unused.png", words=receipt_words)
        d = ev.to_dict()
        assert set(d) == {"kind", "quote", "bbox", "page", "score"}
        assert d["kind"] == "ocr_span"


class TestEngineRouting:
    def test_auto_falls_back_to_tesseract(self, monkeypatch):
        monkeypatch.setattr(ev_mod, "_rapidocr_engine", lambda: None)
        monkeypatch.setattr(ev_mod, "_tesseract_words", lambda p: [{"text": "T"}])
        assert ev_mod._ocr_words("x.png", engine="auto") == [{"text": "T"}]

    def test_explicit_rapidocr_never_touches_tesseract(self, monkeypatch):
        monkeypatch.setattr(ev_mod, "_rapidocr_engine", lambda: None)
        monkeypatch.setattr(
            ev_mod, "_tesseract_words", lambda p: pytest.fail("tesseract called")
        )
        assert ev_mod._ocr_words("x.png", engine="rapidocr") == []

    def test_explicit_tesseract_never_touches_rapidocr(self, monkeypatch):
        monkeypatch.setattr(
            ev_mod, "_rapidocr_engine", lambda: pytest.fail("rapidocr called")
        )
        monkeypatch.setattr(ev_mod, "_tesseract_words", lambda p: [{"text": "T"}])
        assert ev_mod._ocr_words("x.png", engine="tesseract") == [{"text": "T"}]

    def test_auto_prefers_rapidocr_when_available(self, monkeypatch):
        monkeypatch.setattr(ev_mod, "_rapidocr_words", lambda p: [{"text": "R"}])
        monkeypatch.setattr(
            ev_mod, "_tesseract_words", lambda p: pytest.fail("tesseract called")
        )
        assert ev_mod._ocr_words("x.png", engine="auto") == [{"text": "R"}]

    def test_locate_threads_ocr_engine(self, monkeypatch, receipt_words):
        seen = {}

        def fake_words(path, engine="auto"):
            seen["engine"] = engine
            return receipt_words

        monkeypatch.setattr(ev_mod, "_ocr_words", fake_words)
        ev = locate_field_evidence("34.21", "x.png", ocr_engine="tesseract")
        assert ev is not None
        assert seen["engine"] == "tesseract"


class TestAttach:
    def test_attach_maps_all_fields(self, monkeypatch, receipt_words):
        monkeypatch.setattr(ev_mod, "_ocr_words", lambda p, engine="auto": receipt_words)
        fields = {
            "company": "GARDENIA BAKERIES (KL) SDN BHD",
            "date": "22/10/2017",
            "address": "",
            "total": "34.21",
            "evidence": "should-be-skipped",
        }
        out = attach_evidence(fields, "x.png")
        assert set(out) == {"company", "date", "address", "total"}
        assert out["company"] is not None
        assert out["total"] is not None
        assert out["date"] is not None
        assert out["address"] is None  # empty value → no evidence

    def test_attach_engine_param_passes_through(self, monkeypatch):
        seen = {}

        def fake_words(path, engine="auto"):
            seen["engine"] = engine
            return []

        monkeypatch.setattr(ev_mod, "_ocr_words", fake_words)
        attach_evidence({"total": "1.00"}, "x.png", ocr_engine="rapidocr")
        assert seen["engine"] == "rapidocr"
