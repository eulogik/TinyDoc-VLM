"""Sensitivity probe: re-attempt 3B docs that failed with exceed_context_size_error
at 8K context. Same model, same EXTRACT_PROMPT, same temperature — only the
window changes. Quantifies how much of the 3B's clean-set deficit is harness
config (4K ctx) vs model capability. NOT the product path (pipeline.py is
untouched); the shipped-path baseline stands as the bar.
Usage:
  python evaluation/phase0/probe_3b_8k.py --ids clean_0018 clean_0021
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from metrics import FIELDS, score_example
from tinydoc.pipeline import EXTRACT_PROMPT, _extract_json

RESULTS = ROOT / "evaluation/phase0/results"


def call_ollama(image_path: str, num_ctx: int) -> tuple[dict, str, float]:
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {"model": "qwen2.5vl:3b",
               "messages": [{"role": "user", "content": EXTRACT_PROMPT, "images": [b64]}],
               "stream": False,
               "options": {"temperature": 0, "num_predict": 512, "num_ctx": num_ctx}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())
    raw = (body.get("message") or {}).get("content", "") or ""
    obj = _extract_json(raw) or {}
    return {f: str(obj.get(f, "") or "").strip() for f in FIELDS}, raw, (time.time() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--num-ctx", type=int, default=8192)
    args = ap.parse_args()
    clean = {c["id"]: c for c in json.loads((RESULTS / "sroie_eval_clean.json").read_text())}
    rows = []
    for i in args.ids:
        c = clean[i]
        try:
            pred, _raw, lat = call_ollama(c["image_path"], args.num_ctx)
            err = ""
        except Exception as e:  # noqa: BLE001
            pred, _raw, lat, err = {f: "" for f in FIELDS}, "", 0.0, f"{type(e).__name__}: {e}"
        sc = score_example(pred, {f: c["gold"].get(f, "") for f in FIELDS})
        rows.append({"id": i, "pred": pred, "latency_ms": lat, "error": err,
                     "f1": sc["f1"], "address_hit": sc["fields"]["address"]["hit"]})
        print(f"{i}: f1={sc['f1']:.2f} addr_hit={sc['fields']['address']['hit']} err={err!r}")
    (RESULTS / "probe_3b_8k.json").write_text(json.dumps(
        {"num_ctx": args.num_ctx, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
