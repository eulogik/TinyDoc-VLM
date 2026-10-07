"""Product-path parity: ReceiptPipeline(hybrid_2p2b) vs the phase-0 harness.

The committed hybrid numbers (F1 0.5985) came from run_hybrid_eval.py with a
stand-alone MLX load. This proves the SHIPPED path (sdk ReceiptPipeline with
the HybridAddressEngine) produces the same per-doc fields: same EXTRACT_PROMPT
constant, same mlx chat-template construction, same frozen F3 rule functions.
Run once per product change; NOT in CI (needs MLX weights + ~15 min).
Usage:
  SCRATCH_VENV/bin/python evaluation/phase0/run_product_parity.py
Writes: results/product_parity_hybrid.json {match_rate, mismatches[]}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))

from tinydoc.pipeline import ReceiptPipeline

RESULTS = ROOT / "evaluation/phase0/results"


def main() -> int:
    ref = {r["id"]: r for r in
           map(json.loads, (RESULTS / "preds_hybrid-2.2b-tess_clean.jsonl")
               .read_text().splitlines()) if True}
    items = json.loads((RESULTS / "sroie_eval_clean.json").read_text())
    pipe = ReceiptPipeline("hybrid_2p2b")
    match = mismatch = 0
    mismatches = []
    lat = []
    for n, item in enumerate(items):
        t0 = time.time()
        doc = pipe.extract(item["image_path"], with_evidence=False)
        lat.append((time.time() - t0) * 1000)
        got = {k: doc.fields.get(k, "") for k in
               ("company", "date", "address", "total")}
        want = ref[item["id"]]["pred"]
        if got == want:
            match += 1
        else:
            mismatch += 1
            if len(mismatches) < 5:
                mismatches.append({"id": item["id"], "product": got,
                                   "harness": want})
        if (n + 1) % 20 == 0 or n + 1 == len(items):
            print(f"  [{n + 1}/{len(items)}] match={match}/{n + 1}", flush=True)
    import statistics
    out = {"n": len(items), "match": match, "mismatch": mismatch,
           "match_rate": match / len(items),
           "latency_ms_mean": round(statistics.mean(lat)),
           "latency_ms_median": round(statistics.median(lat)),
           "latency_ms_p90": round(sorted(lat)[int(len(lat) * 0.9)]),
           "mismatch_examples": mismatches}
    (RESULTS / "product_parity_hybrid.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
