"""Unit tests for the grounding probe parser and IoU (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from run_grounding_probe import box_iou, parse_box


def test_parse_box_tag():
    assert parse_box("<box>100 200 300 400</box>", 1000, 1000) == (100.0, 200.0, 300.0, 400.0)


def test_parse_box_json_normalized():
    assert parse_box("[0.1, 0.2, 0.3, 0.4]", 1000, 1000) == (100.0, 200.0, 300.0, 400.0)


def test_parse_box_pixels_are_scaled():
    assert parse_box("<box>1000 200 3000 400</box>", 4000, 2000) == (250.0, 100.0, 750.0, 200.0)


def test_parse_box_sorts_and_clamps():
    assert parse_box("<box>300 400 100 200</box>", 1000, 1000) == (100.0, 200.0, 300.0, 400.0)
    assert parse_box("<box>-10 -10 1200 1200</box>", 1000, 1000) == (0.0, 0.0, 1000.0, 1000.0)


def test_parse_box_invalid():
    assert parse_box("no box here", 1000, 1000) is None
    assert parse_box("<box>1 2 3</box>", 1000, 1000) is None


def test_box_iou():
    a = (0.0, 0.0, 100.0, 100.0)
    assert box_iou(a, a) == 1.0
    assert box_iou(a, (200.0, 200.0, 300.0, 300.0)) == 0.0
    # Half-overlap: intersection 50x100=5000, union 10000+10000-5000=15000.
    assert abs(box_iou(a, (50.0, 0.0, 150.0, 100.0)) - 1 / 3) < 1e-9
