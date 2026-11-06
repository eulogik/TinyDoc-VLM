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
