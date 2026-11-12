"""Guard the eval-provenance manifest against drift.

The provenance file is the public statement of what each set is and is not.
Hashes make silent edits to the eval sets visible; optional datasets (FUNSD)
are skipped when the external data directory is absent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "evaluation/phase0/results/eval_provenance.json"
sys.path.insert(0, str(ROOT / "evaluation" / "docvqa"))

from run_docvqa_probe import anls_score, normalize_answer


def test_provenance_hashes_and_labels():
    data = json.loads(PROVENANCE.read_text())

    for name, entry in data["datasets"].items():
        path = ROOT / entry["path"]
        if not path.exists():
            # FUNSD lives behind the untracked evaluation/data symlink.
            assert name == "funsd", f"missing tracked eval set: {name}"
            continue
        raw = path.read_bytes()
        assert len(raw) == entry["bytes"], f"{name}: size changed"
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], f"{name}: hash changed"

    original = data["datasets"]["sroie_eval"]
    assert original["leakage"]["public_train_overlap"] == 61
    assert original["leakage"]["fraction"] == 0.61

    clean = data["datasets"]["sroie_eval_clean"]
    assert clean["source"] == "sroie_public_train_clean"
    assert clean["leakage"]["external_test"] is False

    assert data["multi_vertical"]["invoices"]["status"] == "synthetic_probe"
    assert data["multi_vertical"]["invoices"]["caveat"].startswith("synthetic")
    assert data["multi_vertical"]["invoices"]["real_invoices"] == "missing"
    assert (ROOT / data["claim_audit"]).exists()


def test_probe_provenance_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    probes = data["probes"]

    grounding = probes["grounding"]
    assert grounding["status"] == "failed"
    assert grounding["seed"] == 20261106
    assert grounding["num_docs"] == 30
    for name, expected in grounding["results"].items():
        result = json.loads((ROOT / expected["path"]).read_text())
        assert result["n"] == expected["n"], name
        assert abs(result["parse_rate"] - expected["parse_rate"]) < 1e-9, name
        assert abs(result["mean_iou"] - expected["mean_iou"]) < 5e-4, name
        assert abs(result["hit_at_0.5"] - expected["hit_at_0.5"]) < 1e-9, name
        # The product gate is 0.70; these results must not accidentally pass.
        assert result["hit_at_0.5"] < 0.70, name

    confidence = probes["confidence"]
    assert confidence["status"] == "uncalibrated"
    for name, expected in confidence["results"].items():
        result = json.loads((ROOT / expected["path"]).read_text())
        assert result["n_docs"] == confidence["n_docs"], name
        assert result["used_field_confidences"] is False, name
        field = result["field_level"]
        assert abs(field["ece_10bin"] - expected["ece_10bin"]) < 5e-4, name
        assert abs(field["brier"] - expected["brier"]) < 5e-4, name
        assert abs(field["mean_confidence"] - expected["mean_confidence"]) < 1e-9, name
        assert abs(field["accuracy"] - expected["accuracy"]) < 1e-9, name

    manifest = Path(grounding.get("manifest_path", ""))
    if manifest.exists():
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == grounding["manifest_sha256"]


def test_evidence_on_calibration_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    evidence = data["probes"]["confidence_evidence_on"]
    assert evidence["path"].startswith("evidence_on")
    assert evidence["n_docs"] == 100
    assert evidence["calibration"]["folds"] == 5
    assert evidence["calibration"]["grouping"].startswith("document")

    for name, expected in evidence["results"].items():
        for key in ("preds", "probe", "calibration"):
            assert (ROOT / expected[key]).exists(), f"{name}: missing {key}"
        report = json.loads((ROOT / expected["calibration"]).read_text())
        assert report["n_docs"] == evidence["n_docs"], name
        assert report["folds"] == evidence["calibration"]["folds"], name
        assert report["constant_confidence"] is False, name

        field = report["fields"]
        assert abs(field["raw"]["ece_10bin"] - expected["field"]["ece_raw"]) < 5e-4, name
        assert (
            abs(field["isotonic_oof"]["ece_10bin"] - expected["field"]["ece_isotonic_oof"])
            < 5e-4
        ), name
        # Field-level calibration must improve ECE.
        assert field["isotonic_oof"]["ece_10bin"] < field["raw"]["ece_10bin"], name

        doc = report["doc_exact4"]
        assert abs(doc["raw"]["ece_10bin"] - expected["doc_exact4"]["ece_raw"]) < 5e-4, name
        assert (
            abs(doc["isotonic_oof"]["ece_10bin"] - expected["doc_exact4"]["ece_isotonic_oof"])
            < 5e-4
        ), name

    # Document-level isotonic is not reliably better; the 2B got worse.
    two_b = evidence["results"]["qwen3_vl_2b_instruct"]["doc_exact4"]
    assert two_b["ece_isotonic_oof"] > two_b["ece_raw"]


def test_invoice_probe_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    invoice = data["probes"]["invoice_transfer"]
    assert invoice["set"] == "invoice_extraction_benchmark_v1"
    assert invoice["n_docs"] == 181
    assert invoice["page"].startswith("first page")
    assert "synthetic" in invoice["caveat"]
    assert invoice["real_invoices"] == "missing"

    artifacts = invoice["artifacts"]
    summary = json.loads((ROOT / artifacts["summary"]).read_text())
    assert summary["n_docs"] == invoice["n_docs"]
    for key in ("key_header_accuracy", "fully_correct_rate"):
        assert abs(summary[key] - invoice["results"][key]) < 1e-5, key
    assert abs(summary["line_items"]["f1_mean"] - invoice["results"]["line_f1_mean"]) < 1e-5
    assert summary["key_header_correct"] == invoice["results"]["key_header_correct"]
    assert summary["key_header_scored"] == invoice["results"]["key_header_scored"]
    assert summary["fully_correct"] == invoice["results"]["fully_correct"]
    assert summary["parse_errors"] == 0
    key_ci = invoice["results"]["key_header_accuracy_ci95"]
    full_ci = invoice["results"]["fully_correct_rate_ci95"]
    assert key_ci[0] <= summary["key_header_accuracy"] <= key_ci[1]
    assert full_ci[0] <= summary["fully_correct_rate"] <= full_ci[1]

    scores = json.loads((ROOT / artifacts["scores"]).read_text())
    assert len(scores["documents"]) == invoice["n_docs"]
    assert scores["summary"]["key_fields"]["correct"] == invoice["results"]["key_header_correct"]
    assert scores["summary"]["key_fields"]["scored"] == invoice["results"]["key_header_scored"]
    flags = invoice["results"]["flags"]
    assert flags["inconsistent_invoices"] == {"total": 5, "flagged": 0}
    assert flags["multi_invoice_files"] == {"total": 2, "flagged": 0}

    categories = invoice["results"]["per_category"]
    assert categories["clean"]["n"] == 31
    assert categories["multipage"]["n"] == 10
    assert categories["multipage"]["fully_correct_rate"] == 0.0
    assert categories["photo"]["n"] == 45


def test_invoice_tier_comparison_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    invoice = data["probes"]["invoice_transfer"]
    assert invoice["model"] == "qwen2.5vl:3b"

    artifacts = invoice["artifacts"]
    summary_2b = json.loads((ROOT / artifacts["summary_2b"]).read_text())
    two_b = invoice["results_2b"]
    for key in ("key_header_accuracy", "fully_correct_rate"):
        assert abs(summary_2b[key] - two_b[key]) < 1e-5, key
    assert summary_2b["key_header_correct"] == two_b["key_header_correct"]
    assert summary_2b["fully_correct"] == two_b["fully_correct"]
    assert summary_2b["parse_errors"] == 0
    assert abs(summary_2b["line_items"]["f1_mean"] - two_b["line_f1_mean"]) < 1e-5

    comparison = json.loads((ROOT / artifacts["comparison"]).read_text())
    recorded = invoice["tier_comparison"]
    assert abs(comparison["key"]["diff"] - recorded["key_header_accuracy"]["diff"]) < 1e-5
    assert abs(comparison["fully_correct"]["diff"] - recorded["fully_correct_rate"]["diff"]) < 1e-5
    assert abs(comparison["key"]["sign_p"] - recorded["key_header_accuracy"]["sign_p"]) < 1e-5
    assert comparison["n_docs"] == 181
    assert recorded["key_header_accuracy"]["b_better_docs"] == 32
    assert recorded["key_header_accuracy"]["a_better_docs"] == 19

    # The point estimate favors 2B, but the paired test is not significant.
    assert recorded["key_header_accuracy"]["diff"] > 0
    assert recorded["key_header_accuracy"]["sign_p"] > 0.05
    assert recorded["fully_correct_rate"]["ci95"][0] <= 0.0


def test_multipage_probe_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    mp = data["probes"]["multipage"]
    assert mp["n_docs"] == 10
    assert mp["pages"] == 46
    assert mp["page_render_px"] == 1400
    assert mp["model"] == "qwen3-vl:2b-instruct"
    assert "descriptive" in mp["caveat"]

    artifacts = mp["artifacts"]
    summary = json.loads((ROOT / artifacts["summary"]).read_text())
    scores = json.loads((ROOT / artifacts["scores"]).read_text())
    assert len(scores["documents"]) == 10
    assert scores["summary"]["key_fields"] == {"correct": 64, "scored": 90}
    rows = [
        json.loads(line)
        for line in (ROOT / artifacts["rows"]).read_text().splitlines()
        if line.strip()
    ]
    comparison = json.loads((ROOT / artifacts["compare"]).read_text())

    assert summary["n_docs"] == 10
    assert summary["parse_errors"] == 2
    for source, expected in ((summary, mp["results"]["all_pages"]),):
        for key in ("key_header_accuracy", "fully_correct_rate"):
            assert abs(source[key] - expected[key]) < 1e-5, key
        assert source["key_header_correct"] == expected["key_header_correct"]
        assert source["fully_correct"] == expected["fully_correct"]
        assert abs(source["line_items"]["f1_mean"] - expected["line_f1_mean"]) < 1e-5

    expected_failures = {"multipage-8p-us-9", "multipage-8p-na-10"}
    failed = {row["id"] for row in rows if row.get("error")}
    assert failed == expected_failures
    for row in rows:
        if row.get("error"):
            assert row["done_reason"] == "length"
        else:
            assert row["done_reason"] == "stop"
            assert (row["prompt_eval_count"] + row["eval_count"]) <= 16384

    assert comparison["n_docs"] == 10
    assert comparison["page1"]["key_correct"] == 57
    assert comparison["page1"]["fully_correct"] == 0
    assert comparison["all_pages"]["key_correct"] == 64
    assert comparison["all_pages"]["fully_correct"] == 4
    assert comparison["transitions"]["key_sign_p"] > 0.05
    assert comparison["transitions"]["fully_correct_sign_p"] > 0.05
    # Line-item F1 gets worse when every page is sent at 1400 px; do not hide it.
    assert comparison["all_pages"]["mean_line_f1"] < comparison["page1"]["mean_line_f1"]
    assert comparison["all_pages"]["key_accuracy"] > comparison["page1"]["key_accuracy"]

    valid = mp["results"]["valid_only"]
    assert valid["n_docs"] == 8
    assert valid["key_header_correct"] == 62
    assert valid["fully_correct"] == 4


def test_docvqa_probe_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    docvqa = data["multi_vertical"]["docvqa"]
    assert docvqa["status"] == "measured_probe"
    assert docvqa["n"] == 200
    assert docvqa["single_reference"] is True
    assert "not comparable" in docvqa["caveat"]

    probe = data["probes"]["docvqa_reading"]
    assert probe["model"] == "qwen3-vl:2b-instruct"
    assert probe["image_max_edge_px"] == 768

    artifacts = probe["artifacts"]
    manifest = [
        json.loads(line)
        for line in (ROOT / artifacts["manifest"]).read_text().splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in (ROOT / artifacts["rows"]).read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((ROOT / artifacts["summary"]).read_text())
    scores = json.loads((ROOT / artifacts["scores"]).read_text())

    assert len(manifest) == 200
    assert len({item["image"] for item in manifest}) == 200
    assert all(item["question"].strip() and item["target"].strip() for item in manifest)
    assert len(rows) == 200
    assert sum(1 for row in rows if row.get("error")) == 0

    anls = sum(anls_score(row["answer"], row["target"]) for row in rows) / len(rows)
    exact = sum(
        1 for row in rows
        if normalize_answer(row["answer"]) == normalize_answer(row["target"])
    ) / len(rows)
    assert abs(anls - 0.829397) < 1e-6
    assert abs(exact - 0.765) < 1e-9
    assert abs(summary["anls_mean"] - anls) < 1e-9
    assert abs(summary["exact_rate"] - exact) < 1e-9
    assert summary["errors"] == 0
    assert summary["stored_anls_mismatches"] == 0
    assert summary["single_reference"] is True
    assert abs(scores["anls_mean"] - summary["anls_mean"]) < 1e-9
    assert abs(scores["exact_rate"] - summary["exact_rate"]) < 1e-9
    assert "not comparable" in probe["caveat"]


def test_pagewise_probe_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    assert data["multi_vertical"]["invoices"]["multipage_pagewise"] == "measured_separately"
    probe = data["probes"]["multipage_pagewise"]
    assert probe["n_docs"] == 10
    assert probe["pages"] == 46
    assert probe["model"] == "qwen3-vl:2b-instruct"
    assert probe["context"] == 8192
    assert probe["output_budget"] == 2048
    assert "no gold" in probe["method"]
    assert "descriptive" in probe["caveat"]

    artifacts = probe["artifacts"]
    rows = [
        json.loads(line)
        for line in (ROOT / artifacts["rows"]).read_text().splitlines()
        if line.strip()
    ]
    raw_rows = [
        json.loads(line)
        for line in (ROOT / artifacts["rows_raw"]).read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((ROOT / artifacts["summary"]).read_text())
    scores = json.loads((ROOT / artifacts["scores"]).read_text())
    comparison = json.loads((ROOT / artifacts["compare"]).read_text())
    audit = json.loads((ROOT / artifacts["audit"]).read_text())
    rebuild = json.loads((ROOT / artifacts["rebuild"]).read_text())

    assert len(rows) == 10
    assert len(raw_rows) == 10
    assert sum(len(row["pages"]) for row in rows) == 46
    assert all(row.get("error") is None for row in rows)
    assert all(row.get("error") is None for row in raw_rows)
    for row in rows:
        for page in row["pages"]:
            assert page["done_reason"] == "stop"
            assert page["prompt_eval_count"] + page["eval_count"] <= 8192

    assert summary["n_docs"] == 10
    assert summary["parse_errors"] == 0
    assert summary["key_header_correct"] == 84
    assert summary["key_header_scored"] == 90
    assert abs(summary["key_header_accuracy"] - 0.933333) < 1e-6
    assert summary["fully_correct"] == 6
    assert abs(summary["line_items"]["f1_mean"] - 0.848) < 1e-9
    assert summary["per_field"]["currency"] == {"correct": 7, "scored": 10}
    assert scores["summary"]["key_fields"] == {"correct": 84, "scored": 90}
    assert scores["summary"]["fully_correct"] == 6

    assert comparison["before"]["all_pages"]["key_correct"] == 64
    assert comparison["before"]["all_pages"]["fully_correct"] == 4
    assert comparison["after"]["pagewise"]["key_correct"] == 84
    assert comparison["after"]["pagewise"]["fully_correct"] == 6
    assert abs(comparison["after"]["pagewise"]["mean_line_f1"] - 0.848) < 1e-9
    assert comparison["transitions"]["key_sign_p"] > 0.05
    assert comparison["transitions"]["fully_correct_sign_p"] > 0.05

    assert audit["ok"] is True
    assert rebuild["corrections"] == 2
    assert rebuild["review_flags"] == 8
    corrections = {
        row["id"]: row["pred"].get("corrections")
        for row in rows
        if (row.get("pred") or {}).get("corrections")
    }
    assert corrections == {
        "multipage-3p-na-3": ["subtotal_from_total_tax_residual"],
        "multipage-6p-za-8": ["tax_from_total_subtotal_truncated"],
    }


def test_pagewise_ocr_probe_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    assert data["multi_vertical"]["invoices"]["multipage_pagewise_ocr"] == "measured"
    assert data["multi_vertical"]["invoices"]["multipage_pagewise_prompt_v2"] == "tested_no_gain"
    probe = data["probes"]["multipage_pagewise_ocr"]
    assert probe["n_docs"] == 10
    assert probe["pages"] == 46
    assert probe["model"] == "qwen3-vl:2b-instruct"
    assert probe["prompt_version"] == 1
    assert probe["ocr"]["consistent_blocks"] == 10
    assert "no gold" in probe["method"]

    artifacts = probe["artifacts"]
    rows = [
        json.loads(line)
        for line in (ROOT / artifacts["rows"]).read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((ROOT / artifacts["summary"]).read_text())
    scores = json.loads((ROOT / artifacts["scores"]).read_text())
    audit = json.loads((ROOT / artifacts["audit"]).read_text())
    rebuild = json.loads((ROOT / artifacts["rebuild"]).read_text())

    assert len(rows) == 10
    assert sum(len(row["pages"]) for row in rows) == 46
    assert all(row.get("error") is None for row in rows)
    assert all((row.get("ocr_totals") or {}).get("consistent") is True for row in rows)
    assert summary["key_header_correct"] == 90
    assert summary["key_header_scored"] == 90
    assert summary["fully_correct"] == 10
    assert summary["parse_errors"] == 0
    assert abs(summary["line_items"]["f1_mean"] - 0.848) < 1e-9
    assert summary["line_items"]["all_lines_right"] == 1
    assert 0.69 < summary["ci95_exact"]["fully_correct_rate"][0] < 0.70
    assert scores["summary"]["key_fields"] == {"correct": 90, "scored": 90}
    assert scores["summary"]["fully_correct"] == 10
    assert audit["ok"] is True
    assert rebuild["ocr_totals_consistent"] == 10
    assert rebuild["corrections"] == 3
    assert rebuild["review_flags"] == 8

    corrections = {
        row["id"]: row["pred"]["corrections"]
        for row in rows
        if row["pred"].get("corrections")
    }
    assert corrections == {
        "multipage-2p-za-1": ["totals_from_ocr:subtotal+tax+total"],
        "multipage-3p-na-3": [
            "subtotal_from_total_tax_residual",
            "currency_from_ocr_totals",
        ],
        "multipage-6p-za-8": ["tax_from_total_subtotal_truncated"],
    }

    experiment = probe["prompt_v2_experiment"]
    assert experiment["raw_results"]["key_header"] == "81/90"
    assert experiment["raw_results"]["fully_correct"] == "5/10"
    assert "did not reliably fix currency" in experiment["conclusion"]
    assert experiment["after_same_reconciliation"]["fully_correct"] == "10/10"


def test_degraded_multipage_probe_matches_artifacts():
    data = json.loads(PROVENANCE.read_text())
    assert data["multi_vertical"]["invoices"]["multipage_pagewise_degraded"] == "measured"
    probe = data["probes"]["multipage_pagewise_degraded"]
    assert probe["n_docs"] == 10
    assert probe["pages"] == 46
    assert probe["documents_complete"] == 8
    assert probe["documents_failed"] == 2
    assert probe["model"] == "qwen3-vl:2b-instruct"
    assert probe["prompt_version"] == 1
    assert probe["degradation"]["seed"] == 42
    assert probe["degradation"]["jpeg_quality"] == 58
    assert probe["degradation"]["recipe"] == "downscale+rotate+blur+brightness+noise+jpeg"
    assert set(probe["failures"]) == {"multipage-3p-gb-4", "multipage-6p-za-8"}

    artifacts = probe["artifacts"]
    rows = [
        json.loads(line)
        for line in (ROOT / artifacts["rows"]).read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((ROOT / artifacts["summary"]).read_text())
    scores = json.loads((ROOT / artifacts["scores"]).read_text())
    audit = json.loads((ROOT / artifacts["audit"]).read_text())
    compare = json.loads((ROOT / artifacts["compare_clean_vs_degraded"]).read_text())
    recipe = json.loads((ROOT / probe["degradation"]["recipe_report"]).read_text())

    assert len(rows) == 10
    assert sum(len(row["pages"]) for row in rows) == 46
    assert sorted(row["id"] for row in rows if row.get("error")) == [
        "multipage-3p-gb-4",
        "multipage-6p-za-8",
    ]
    assert all(
        (row.get("ocr_totals") or {}).get("consistent") is True for row in rows
    )
    assert summary["n_docs"] == 8
    assert summary["key_header_correct"] == 71
    assert summary["key_header_scored"] == 72
    assert summary["fully_correct"] == 7
    assert summary["parse_errors"] == 2
    assert abs(summary["line_items"]["f1_mean"] - 0.795) < 1e-9
    assert summary["line_items"]["all_lines_right"] == 0
    assert 0.47 < summary["ci95_exact"]["fully_correct_rate"][0] < 0.48
    assert scores["summary"]["key_fields"] == {"correct": 71, "scored": 72}
    assert scores["summary"]["fully_correct"] == 7
    assert scores["summary"]["documents"] == 8
    assert audit["ok"] is False
    assert compare["n_docs"] == 8
    assert sorted(compare["missing_from_after"]) == [
        "multipage-3p-gb-4",
        "multipage-6p-za-8",
    ]
    assert abs(compare["delta"]["fully_correct_rate"] + 0.125) < 1e-9
    assert recipe["documents"] == 10
    assert recipe["pages"] == 46
    assert recipe["seed"] == 42
    assert recipe["recipe"]["quality"] == 58
