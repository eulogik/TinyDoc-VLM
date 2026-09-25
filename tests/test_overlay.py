"""Unit tests for evidence overlay rendering (tiny synthetic images only)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from tinydoc.overlay import draw_overlay


def _page(tmp_path: Path, size=(200, 80), color=(255, 255, 255)) -> Path:
    p = tmp_path / "page.png"
    Image.new("RGB", size, color).save(p)
    return p


def test_overlay_writes_and_draws_bbox(tmp_path: Path):
    page = _page(tmp_path)
    fields = [
        {
            "name": "total",
            "value": "34.21",
            "confidence": 0.9,
            "evidence": {
                "kind": "ocr_span",
                "quote": "TOTAL 34.21",
                "bbox": [20.0, 30.0, 90.0, 50.0],
                "page": 1,
                "score": 0.95,
            },
        }
    ]
    out = tmp_path / "overlay.png"
    ret = draw_overlay(page, fields, out)
    assert ret == out and out.exists()

    orig = Image.open(page).convert("RGB")
    drawn = Image.open(out).convert("RGB")
    assert drawn.size == orig.size

    # outline drawn on (inflated by pad=3) → border pixel differs from white
    assert drawn.getpixel((17, 30)) != (255, 255, 255)
    # untouched area far from the box stays white
    assert drawn.getpixel((190, 70)) == (255, 255, 255)


def test_overlay_handles_missing_evidence(tmp_path: Path):
    page = _page(tmp_path)
    fields = [
        {"name": "company", "value": "ACME", "confidence": 0.5, "evidence": None},
        {"name": "date", "value": "", "confidence": 0.0, "evidence": None},
    ]
    out = tmp_path / "legend.png"
    draw_overlay(page, fields, out)
    assert out.exists()


def test_overlay_accepts_field_result_objects(tmp_path: Path):
    from tinydoc.pipeline import FieldResult

    page = _page(tmp_path)
    fields = [
        FieldResult(
            name="total",
            value="9.99",
            present=True,
            evidence={"bbox": [5.0, 5.0, 60.0, 25.0], "quote": "9.99"},
            confidence=0.8,
        )
    ]
    out = tmp_path / "obj.png"
    draw_overlay(page, fields, out)
    assert out.exists()
    drawn = Image.open(out).convert("RGB")
    assert drawn.getpixel((2, 5)) != (255, 255, 255)


def test_overlay_creates_parent_dirs(tmp_path: Path):
    page = _page(tmp_path)
    out = tmp_path / "deep" / "nested" / "out.png"
    draw_overlay(page, [], out)
    assert out.exists()
