"""Prompt parity: the phase-0 harness and the SDK product path must score
receipts under byte-identical instructions.

The 0.870 baseline and every future edge-model comparison are only meaningful if
the prompt is the same string in both places; these drifted once ("no extra
text:"), so this test makes the drift impossible to reintroduce silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
PHASE0 = ROOT / "evaluation" / "phase0"
for p in (SDK, PHASE0):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from engines import EXTRACT_PROMPT as PHASE0_PROMPT
from tinydoc.pipeline import EXTRACT_PROMPT as SDK_PROMPT


def test_prompts_are_identical():
    assert SDK_PROMPT == PHASE0_PROMPT


def test_prompt_is_nonempty_and_requests_four_keys():
    assert SDK_PROMPT.strip()
    for key in ("company", "date", "address", "total"):
        assert f'"{key}"' in SDK_PROMPT
    assert "JSON only" in SDK_PROMPT
