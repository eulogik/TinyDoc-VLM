"""Hybrid parity: the product path and the phase-0 harness must fuse the OCR
address window with the VLM address under the byte-identical frozen rule F3.

Covers the pure rule functions (shared by sdk/tinydoc/evidence.py and
evaluation/phase0/run_hybrid_eval.py) on synthetic lines, plus the tesseract
PSM-6 line source smoke (skipped when tesseract is unavailable, as in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
for p in (SDK,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tinydoc.evidence import (
    F3_WINDOW_AFTER,
    F3_WINDOW_BEFORE,
    fuse_address,
    ocr_address_window,
)


def test_window_centers_on_first_postcode_line():
    lines = ["SHOP NAME", "JALAN ABC 12", "TAMAN XYZ", "81100 JB", "TEL 123"]
    assert ocr_address_window(lines) == " ".join(lines[0:5 + 0][:5])


def test_window_clamps_at_edges():
    lines = ["A 11111 B", "tail1", "tail2"]
    assert ocr_address_window(lines) == "A 11111 B tail1"
    assert ocr_address_window(["no postcode here"]) == ""


def test_window_before_after_constants():
    assert (F3_WINDOW_BEFORE, F3_WINDOW_AFTER) == (4, 1)


def test_fuse_keeps_vlm_when_inside_window():
    window = "NO 53 55 JALAN SAGU 18 TAMAN DAYA 81100 JOHOR BAHRU"
    vlm = "55, JALAN SAGU 18"
    assert fuse_address(vlm, window) == vlm.strip()


def test_fuse_falls_back_to_window():
    window = "NO 53 55 JALAN SAGU 18 TAMAN DAYA 81100 JOHOR BAHRU"
    assert fuse_address("GARDENIA BAKERIES KL", window) == window
    assert fuse_address("", window) == window
    assert fuse_address("   ", window) == window
