"""Best-of-N oracle probe (val-only, method validation, NOT a bar attempt).

Question: is the right address EVER in the model's top-3 samples when greedy
misses? If oracle@3 ~= greedy, no selection rule can help (misreads, not
variance) and rerank is dead. If oracle@3 >> greedy, a no-gold picker
(OCR-window overlap) is worth building and val-gating.

3 samples/doc at temperature 0.7, base 2.2B, val-40 only. Never test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from metrics import field_match
from tinydoc.pipeline import EXTRACT_PROMPT, _extract_json

RESULTS = ROOT / "evaluation/phase0/results"
MODEL = "/Volumes/KIOXIA 1TB/tinydoc/models/smolvlm2-2.2b-mlx"


def main() -> int:
    from mlx_vlm.generate import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_chat
    from mlx_vlm.utils import load as load_model

    model, processor = load_model(MODEL)
    conv = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": EXTRACT_PROMPT}]}]
    prompt = mlx_chat(processor, getattr(model, "config", {}), conv,
                      add_generation_prompt=True, num_images=1)
    val = json.loads((RESULTS / "sroie_val.json").read_text())
    greedy_hit = oracle_hit = 0
    rows = []
    for n, item in enumerate(val):
        gold = item["gold"].get("address", "")
        outs = []
        for k in range(3):
            out = mlx_generate(model, processor, prompt, image=item["image_path"],
                               max_tokens=256, temperature=0.7)
            raw = getattr(out, "text", None)
            raw = raw if isinstance(raw, str) else str(out)
            obj = _extract_json(raw) or {}
            outs.append(str(obj.get("address", "") or "").strip())
        g = field_match(outs[0], gold, "address")
        o = any(field_match(o, gold, "address") for o in outs)
        greedy_hit += g
        oracle_hit += o
        rows.append({"id": item["id"], "greedy_hit": g, "oracle_hit": o,
                     "samples": outs})
        if (n + 1) % 10 == 0 or n + 1 == len(val):
            print(f"  [{n + 1}/{len(val)}] greedy={greedy_hit} oracle={oracle_hit}",
                  flush=True)
    (RESULTS / "probe_bon_val40.json").write_text(json.dumps(
        {"greedy_hits": greedy_hit, "oracle_hits": oracle_hit,
         "n": len(val), "rows": rows}, indent=2))
    print(f"greedy {greedy_hit}/{len(val)} vs oracle@3 {oracle_hit}/{len(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
