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

    assert data["multi_vertical"]["invoices"]["status"] == "missing"
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
