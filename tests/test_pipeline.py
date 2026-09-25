"""Unit tests for ReceiptPipeline: schema, sanitize, router, confidence.

No network / no Ollama required — engines are stubbed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

EVAL = ROOT / "evaluation" / "phase0"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from metrics import address_metrics
from tinydoc.pipeline import (
    FIELDS,
    BaseEngine,
    DocumentResult,
    FieldResult,
    OcrRegexEngine,
    ReceiptPipeline,
    RoutedEngine,
    _extract_json,
    _field_confidence,
    build_engine,
    collapse_repetition,
    load_schema,
    sanitize_fields,
    schema_is_valid,
)


@pytest.fixture
def receipt_image(tmp_path: Path) -> Path:
    p = tmp_path / "receipt.png"
    Image.new("RGB", (200, 300), (255, 255, 255)).save(p)
    return p


class StubEngine(BaseEngine):
    name = "stub"

    def __init__(self, fields=None, name="stub"):
        self.name = name
        self._fields = fields or {}
        self.calls = 0

    def extract_fields(self, image_path: str):
        self.calls += 1
        return dict(self._fields)


class TestSchema:
    def test_load_schema_has_required_fields(self):
        schema = load_schema()
        assert schema is not None
        assert set(schema["required"]) == set(FIELDS)

    def test_schema_is_valid_ok(self):
        obj = {
            "company": "ACME",
            "date": "2024-01-15",
            "address": "1 Main St",
            "total": "12.34",
        }
        assert schema_is_valid(obj) is True

    def test_schema_is_valid_missing_key(self):
        obj = {"company": "ACME", "date": "2024-01-15", "address": "1 Main St"}
        assert schema_is_valid(obj) is False

    def test_schema_is_valid_empty_required(self):
        obj = {"company": "", "date": "2024-01-15", "address": "1 Main St", "total": "1"}
        assert schema_is_valid(obj) is False

    def test_schema_is_valid_not_dict(self):
        assert schema_is_valid(None) is False
        assert schema_is_valid([]) is False

    def test_schema_is_valid_maxlength(self):
        obj = {
            "company": "C",
            "date": "2024-01-15",
            "address": "x" * 401,
            "total": "1",
        }
        assert schema_is_valid(obj) is False


class TestSanitize:
    def test_sanitize_collapses_repetition(self):
        raw = "PERAK, " * 50
        out = sanitize_fields({"address": raw, "company": "A", "date": "d", "total": "1"})
        assert "PERAK" in out["address"]
        assert out["address"].count("PERAK") <= 3

    def test_sanitize_truncates_maxlen(self):
        out = sanitize_fields(
            {"company": "x" * 300, "date": "d", "address": "a", "total": "1"}
        )
        assert len(out["company"]) <= 200

    def test_sanitize_missing_keys_become_empty(self):
        out = sanitize_fields({})
        assert set(out) == set(FIELDS)
        assert all(out[k] == "" for k in FIELDS)

    def test_collapse_repetition_ngram_loop(self):
        unit = "TAMPO 81200 PERAK"
        text = " ".join([unit] * 10)
        out = collapse_repetition(text)
        assert out.count("TAMPO") <= 3

    def test_collapse_repetition_single_token_run(self):
        out = collapse_repetition("hello " * 10)
        assert out.split().count("hello") <= 3

    def test_collapse_repetition_adjacent_duplicate(self):
        out = collapse_repetition("PERAK, PERAK,")
        assert out.count("PERAK") <= 2


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a":"1"}') == {"a": "1"}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"a":"1"}\n```') == {"a": "1"}

    def test_truncated_salvage(self):
        obj = _extract_json('{"company":"ACME","date":"2024"')
        assert obj is not None
        assert obj.get("company") == "ACME"

    def test_empty(self):
        assert _extract_json("") is None
        assert _extract_json("no json here") is None

    def test_multiple_objects_keeps_first(self):
        obj = _extract_json('{"a":"1"} {"b":"2"}')
        assert obj == {"a": "1"}


class TestRouter:
    def test_build_engine_ocr_regex(self):
        eng = build_engine("ocr_regex")
        assert isinstance(eng, OcrRegexEngine)

    def test_build_engine_unknown(self):
        with pytest.raises(ValueError):
            build_engine("nope")

    def test_routed_fills_empty_only_when_schema_fails(self):
        primary = StubEngine(
            {"company": "", "date": "", "address": "", "total": ""}, name="pri"
        )
        fallback = StubEngine(
            {"company": "FB", "date": "2024-01-01", "address": "1 St", "total": "9"},
            name="fb",
        )
        routed = RoutedEngine(primary, fallback)
        out = routed.extract_fields("x.png")
        # schema was invalid (empties) → fallback fills empties
        assert out["company"] == "FB"
        assert primary.calls == 1 and fallback.calls == 1

    def test_routed_never_overwrites_nonempty_primary(self):
        primary = StubEngine(
            {
                "company": "GOOD",
                "date": "2024-01-15",
                "address": "1 Main",
                "total": "10.00",
            },
            name="pri",
        )
        fallback = StubEngine(
            {"company": "BAD", "date": "x", "address": "y", "total": "z"},
            name="fb",
        )
        routed = RoutedEngine(primary, fallback)
        out = routed.extract_fields("x.png")
        assert out["company"] == "GOOD"
        # schema already valid → fallback not called
        assert fallback.calls == 0


class TestConfidence:
    def test_empty_value_zero(self):
        assert _field_confidence("total", "", None, True) == 0.0

    def test_schema_ok_base(self):
        c = _field_confidence("company", "ACME", None, True)
        assert 0.0 < c <= 1.0

    def test_schema_fail_lowers(self):
        ok = _field_confidence("company", "ACME", None, True)
        bad = _field_confidence("company", "ACME", None, False)
        assert bad < ok

    def test_total_with_digits_bonus(self):
        c = _field_confidence("total", "12.34", None, True)
        assert c >= 0.55


class TestReceiptPipeline:
    def test_extract_with_stub(self, receipt_image: Path):
        stub = StubEngine(
            {
                "company": "ACME",
                "date": "2024-01-15",
                "address": "1 Main St, 40000 KL",
                "total": "12.34",
            }
        )
        pipe = ReceiptPipeline(stub)
        # evidence off (no ocr dependency in unit test)
        r = pipe.extract(receipt_image, with_evidence=False)
        assert isinstance(r, DocumentResult)
        assert r.schema_valid is True
        assert r.fields["company"] == "ACME"
        assert r.confidence > 0
        assert r.latency_ms >= 0
        assert r.engine == "stub"
        assert len(r.field_results) == len(FIELDS)
        names = [fr.name for fr in r.field_results]
        assert names == list(FIELDS)

    def test_extract_missing_file(self):
        pipe = ReceiptPipeline(StubEngine())
        with pytest.raises(FileNotFoundError):
            pipe.extract("/nonexistent/image.png", with_evidence=False)

    def test_to_dict_roundtrip_keys(self, receipt_image: Path):
        stub = StubEngine(
            {"company": "A", "date": "d", "address": "a", "total": "1"}
        )
        pipe = ReceiptPipeline(stub)
        d = pipe.extract(receipt_image, with_evidence=False).to_dict()
        for k in (
            "image_path",
            "engine",
            "fields",
            "schema_valid",
            "confidence",
            "evidence_coverage",
            "latency_ms",
            "fields_detail",
        ):
            assert k in d
        # JSON-serializable
        json.dumps(d)

    def test_field_result_dataclass(self):
        fr = FieldResult(name="total", value="1.00", present=True, confidence=0.7)
        assert fr.evidence is None
        assert fr.present is True

    def test_extract_folder_limit(self, tmp_path: Path):
        for i in range(3):
            Image.new("RGB", (50, 50), (255, 0, 0)).save(tmp_path / f"i{i}.png")
        stub = StubEngine(
            {"company": "A", "date": "d", "address": "a", "total": "1"}
        )
        pipe = ReceiptPipeline(stub)
        rs = pipe.extract_folder(tmp_path, limit=2, with_evidence=False)
        assert len(rs) == 2


class TestAddressMetrics:
    def test_exact_address(self):
        out = address_metrics("12 Main St", "12 Main St")
        assert out == {"raw_cer": 0.0, "normalized_f1": 1.0, "raw_exact": 1.0}

    def test_single_character_error(self):
        out = address_metrics("12 Main St", "13 Main St")
        assert out["raw_cer"] == pytest.approx(0.1)
        assert out["normalized_f1"] == pytest.approx(2 / 3)
        assert out["raw_exact"] == 0.0

    def test_punctuation_only_difference(self):
        out = address_metrics("12 Main St", "12 Main St.")
        assert out["raw_cer"] > 0
        assert out["normalized_f1"] == 1.0
        assert out["raw_exact"] == 0.0

    def test_empty_addresses(self):
        assert address_metrics("", "") == {
            "raw_cer": 0.0,
            "normalized_f1": 1.0,
            "raw_exact": 1.0,
        }
        assert address_metrics("12 Main St", "") == {
            "raw_cer": 1.0,
            "normalized_f1": 0.0,
            "raw_exact": 0.0,
        }
