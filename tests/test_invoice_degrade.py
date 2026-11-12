"""Tests for the deterministic degraded-page renderer (no model calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation" / "invoice"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from degrade_pages import degrade_image, degrade_manifest


def _manifest(tmp_path: Path) -> Path:
    paths = []
    for index in range(1, 3):
        path = tmp_path / f"p{index}.png"
        Image.new("RGB", (400, 500), (255 - index * 10, 250, 240)).save(path)
        paths.append(str(path))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "image_path": paths[0],
                "image_paths": paths,
                "source_path": "original.pdf",
            }
        )
        + "\n"
    )
    return manifest


def test_degrade_image_is_deterministic_and_keeps_dimensions(tmp_path):
    source = _manifest(tmp_path)
    row = json.loads(source.read_text().splitlines()[0])
    out_a = tmp_path / "a.jpg"
    out_b = tmp_path / "b.jpg"
    params_a = degrade_image(
        row["image_paths"][0], str(out_a), doc_id="doc-1", page=1, seed=42
    )
    params_b = degrade_image(
        row["image_paths"][0], str(out_b), doc_id="doc-1", page=1, seed=42
    )
    assert out_a.read_bytes() == out_b.read_bytes()
    assert params_a == params_b
    with Image.open(out_a) as image:
        assert image.size == (400, 500)
    assert params_a["jpeg_quality"] == 58
    assert params_a["max_px"] == 1400


def test_degrade_seed_changes_output(tmp_path):
    source = _manifest(tmp_path)
    row = json.loads(source.read_text().splitlines()[0])
    out = tmp_path / "seed.jpg"
    degrade_image(row["image_paths"][0], str(out), seed=7, page=1)
    seed_bytes = out.read_bytes()
    out2 = tmp_path / "seed2.jpg"
    degrade_image(row["image_paths"][0], str(out2), seed=8, page=1)
    assert seed_bytes != out2.read_bytes()


def test_degrade_manifest_preserves_rows_and_records_recipe(tmp_path):
    source = _manifest(tmp_path)
    out_dir = tmp_path / "degraded"
    out_manifest = out_dir / "manifest.jsonl"
    report = degrade_manifest(
        source, out_dir, out_manifest, seed=42, downscale_px=300, quality=50
    )
    assert report["documents"] == 1
    assert report["pages"] == 2
    assert report["recipe"]["quality"] == 50
    rows = [
        json.loads(line)
        for line in out_manifest.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "doc-1"
    assert row["source_path"] == "original.pdf"
    assert len(row["image_paths"]) == 2
    assert all(Path(path).exists() for path in row["image_paths"])
    assert row["degradation"]["seed"] == 42
    assert len(row["degradation"]["pages"]) == 2
    assert row["degradation"]["pages"][0]["jpeg_quality"] == 50
    assert row["image_path"] == row["image_paths"][0]
