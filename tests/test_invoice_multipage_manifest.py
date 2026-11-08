"""Tests for the invoice manifest builder's page and id handling (no model calls)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "evaluation" / "invoice" / "build_manifest.py"


def _make_corpus(tmp_path: Path, count: int = 3) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    documents = []
    for index in range(count):
        name = f"doc_{index}.png"
        Image.new("RGB", (40, 60), "white").save(corpus / name)
        documents.append(
            {
                "doc_id": f"doc-{index}",
                "file": name,
                "category": "clean",
                "difficulty": "easy",
                "kind": "invoice",
                "pages": 1,
                "render": "image",
                "currency": "USD",
                "expected": {"vendor_name": f"Vendor {index}"},
            }
        )
    (corpus / "manifest.json").write_text(json.dumps({"documents": documents}))
    return corpus


def test_ids_filter_and_image_paths_list(tmp_path):
    corpus = _make_corpus(tmp_path)
    out = tmp_path / "manifest.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--corpus",
            str(corpus),
            "--render-dir",
            str(tmp_path / "rendered"),
            "--out",
            str(out),
            "--ids",
            "doc-0",
            "doc-2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert [row["id"] for row in rows] == ["doc-0", "doc-2"]
    for row in rows:
        assert row["image_paths"] == [row["image_path"]]
        assert row["expected"]["vendor_name"].startswith("Vendor")
        assert row["pages"] == 1


def test_all_pages_keeps_one_image_for_non_pdf(tmp_path):
    corpus = _make_corpus(tmp_path, count=1)
    out = tmp_path / "manifest.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--corpus",
            str(corpus),
            "--render-dir",
            str(tmp_path / "rendered"),
            "--out",
            str(out),
            "--all-pages",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(out.read_text().splitlines()[0])
    assert row["image_paths"] == [row["image_path"]]
    assert row["rendered_pages"] == 1
