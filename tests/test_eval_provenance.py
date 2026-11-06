"""Guard the eval-provenance manifest against drift.

The provenance file is the public statement of what each set is and is not.
Hashes make silent edits to the eval sets visible; optional datasets (FUNSD)
are skipped when the external data directory is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "evaluation/phase0/results/eval_provenance.json"


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
