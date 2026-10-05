"""3B teacher-labeling for v2 distillation (Stage-2H).

Labels train images with the 3B pipeline (same EXTRACT_PROMPT, same scorer
family) so the 2.2B student can learn the teacher's exact-transcription style.
Gold is NEVER used here (it's the student's training targets, not eval).
Run in ~15-doc chunks with `ollama stop` between (memory-decay pattern):
  for s in $(seq 0 15 405); do
    ollama stop qwen2.5vl:3b; python evaluation/phase0/teacher_label.py \
      --start $s --end $((s+15)) --out /Volumes/KIOXIA\\ 1TB/tinydoc/distill_train/teacher_$s.jsonl
  done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))

from tinydoc.pipeline import ReceiptPipeline

DROOT = "/Volumes/KIOXIA 1TB/tinydoc/distill_train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = json.loads(Path(f"{DROOT}/manifest.json").read_text())[args.start:args.end]
    pipe = ReceiptPipeline("ollama")
    rows = []
    for n, m in enumerate(man):
        t0 = time.time()
        try:
            doc = pipe.extract(m["image_path"])
            fields = {k: str(doc.fields.get(k, "")) for k in
                      ("company", "date", "address", "total")}
            err = ""
        except Exception as e:  # noqa: BLE001
            fields, err = {k: "" for k in ("company", "date", "address", "total")}, \
                f"{type(e).__name__}: {e}"
        rows.append({"id": m["id"], "image_path": m["image_path"],
                     "teacher": fields, "latency_ms": (time.time() - t0) * 1000,
                     "error": err})
        print(f"  [{n + 1}/{len(man)}] {m['id']} err={bool(err)}", flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} teacher labels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
