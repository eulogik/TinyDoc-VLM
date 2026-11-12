"""Tests for the real Portuguese probe scorer (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation" / "invoice"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from pt_score import (
    norm_date,
    norm_number,
    score_document,
    score_manifest,
    vendor_match,
)


def _expected(**overrides):
    expected = {
        "vendor_name": "carvalho, batista, lda",
        "invoice_number": "FT 2019A17/679",
        "invoice_date": "2019-01-30",
        "currency": "EUR",
        "tax": 33.5,
        "total": 179.17,
        "scored": ["vendor_name", "invoice_number", "invoice_date", "currency", "tax", "total"],
    }
    expected.update(overrides)
    return expected


def test_vendor_match_ignores_case_punctuation_and_legal_forms():
    assert vendor_match("CARVALHO, BATISTA, LDA", "carvalho, batista, lda")[0]
    assert vendor_match("Carvalho Batista", "carvalho, batista, lda")[0]
    assert vendor_match("Loja do Bairro, Lda.", "loja do bairro")[0]
    assert not vendor_match("Padaria Central", "carvalho, batista, lda")[0]


def test_norm_number_and_date():
    assert norm_number(" FT 2019A17/679 ") == "FT2019A17/679"
    assert norm_number("") == ""
    assert norm_date("30/01/2019") == "2019-01-30"
    assert norm_date("2019-01-30") == "2019-01-30"
    assert norm_date(None) is None


def test_score_document_all_six_correct():
    prediction = {
        "vendor_name": "Carvalho, Batista, Lda",
        "invoice_number": "FT2019A17/679",
        "invoice_date": "2019-01-30",
        "currency": "EUR",
        "tax": 33.5,
        "total": 179.17,
    }
    result = score_document(_expected(), prediction)
    assert result["fully_correct"] is True
    assert result["all_six_scored"] is True
    assert result["fully_correct_six"] is True


def test_score_document_reads_total_amount_alias():
    prediction = {
        "vendor_name": "carvalho, batista, lda",
        "invoice_number": "FT 2019A17/679",
        "invoice_date": "2019-01-30",
        "currency": "EUR",
        "tax": 33.5,
        "total_amount": 179.17,
    }
    result = score_document(_expected(), prediction)
    assert result["fields"]["total"]["correct"] is True


def test_score_document_cent_band_and_missing_tax():
    prediction = {
        "vendor_name": "carvalho, batista, lda",
        "invoice_number": "FT 2019A17/679",
        "invoice_date": "2019-01-30",
        "currency": "EUR",
        "tax": 33.51,
        "total": 179.17,
    }
    result = score_document(_expected(), prediction)
    assert result["fields"]["tax"]["correct"] is False
    assert result["fields"]["total"]["correct"] is True
    assert result["fully_correct"] is False

    expected = _expected(scored=["vendor_name", "invoice_number", "invoice_date", "currency", "total"])
    result = score_document(expected, prediction)
    assert "tax" not in result["fields"]
    assert result["fully_correct"] is True
    assert result["all_six_scored"] is False
    assert result["fully_correct_six"] is None


def test_zero_tax_missing_prediction_counts_correct():
    expected = _expected(tax=0.0)
    prediction = {
        "vendor_name": "carvalho, batista, lda",
        "invoice_number": "FT 2019A17/679",
        "invoice_date": "2019-01-30",
        "currency": "EUR",
        "total": 179.17,
    }
    result = score_document(expected, prediction)
    assert result["fields"]["tax"]["correct"] is True


def test_score_manifest_aggregates_fields_and_documents():
    rows = [
        {
            "id": "a",
            "category": "real_pt_company",
            "difficulty": "real",
            "expected": _expected(),
        },
        {
            "id": "b",
            "category": "real_pt_company",
            "difficulty": "real",
            "expected": _expected(scored=["vendor_name", "total"]),
        },
    ]
    predictions = {
        "a": {
            "vendor_name": "carvalho, batista, lda",
            "invoice_number": "FT 2019A17/679",
            "invoice_date": "2019-01-30",
            "currency": "EUR",
            "tax": 33.5,
            "total": 179.17,
        },
        "b": {"vendor_name": "wrong", "total": 1.0},
    }
    scores, summary = score_manifest(rows, predictions, system="test", run_date="2026-11-12")
    assert summary["n_docs"] == 2
    assert summary["field_correct"] == 6
    assert summary["field_scored"] == 8
    assert summary["fully_correct"] == 1
    assert summary["fully_correct_six"] == 1
    assert summary["six_scored"] == 1
    assert summary["per_field"]["vendor_name"] == {"correct": 1, "scored": 2, "accuracy": 0.5}
    assert scores["documents"][0]["doc_id"] == "a"
