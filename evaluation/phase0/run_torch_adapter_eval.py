"""Evaluate Kaggle-trained torch PEFT adapters (vision-LoRA) locally via MPS.

Why torch, not MLX: PEFT adapters (adapter_model.safetensors) match the
transformers architecture exactly; no cross-framework conversion. Slow
(~30s/doc on MPS) but exact. Chunked runs supported via --limit/--offset;
merge by concatenating preds jsonl in doc order and recomputing macro.

Usage (inside venv_torch):
  python evaluation/phase0/run_torch_adapter_eval.py \
      --adapter /Volumes/KIOXIA\\ 1TB/tinydoc/adapters/kaggle_vision/checkpoint-300 \
      --eval-path evaluation/phase0/results/sroie_val.json \
      --out-suffix _kv300val --max-tokens 256
Omit --adapter for the base-model ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from metrics import FIELDS, macro_prf, score_example
from tinydoc.pipeline import EXTRACT_PROMPT, _extract_json

RESULTS = ROOT / "evaluation/phase0/results"
BASE = "/Volumes/KIOXIA 1TB/tinydoc/models/smolvlm2-2.2b-torch"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--adapter", default=None,
                    help="PEFT adapter dir (adapter_model.safetensors); omit for base")
    ap.add_argument("--eval-path", default=str(RESULTS / "sroie_val.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out-suffix", default="_kv300val")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tag-prefix", default="smolvlm2-2.2b")
    ap.add_argument("--doc-timeout", type=int, default=420,
                    help="seconds per doc before recording a timeout error row")
    ap.add_argument("--max-consec-timeouts", type=int, default=3,
                    help="abort the run after this many consecutive timeouts "
                         "(bounds damage from hung generation)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
        model.eval()
    tag = "kv" if args.adapter else "base-torch"

    try:
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": EXTRACT_PROMPT}]}],
            add_generation_prompt=True)
    except Exception:  # noqa: BLE001 — last-resort plain marker
        prompt = f"<image>\n{EXTRACT_PROMPT}"
    print(f"model={args.model} adapter={args.adapter} device={device} "
          f"prompt_len={len(prompt)}", flush=True)

    items = json.loads(Path(args.eval_path).read_text())
    items = items[args.offset:(args.offset + args.limit) if args.limit else None]

    preds, scores = [], []
    import concurrent.futures as _fut

    def _run_one(item):
        t0 = time.time()
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(text=[prompt], images=[[img]],
                               return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                      for k, v in inputs.items() if k != "image_token_id"}
            out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                 do_sample=False, use_cache=True)
            n_in = inputs["input_ids"].shape[-1]
            tok = getattr(processor, "tokenizer", processor)
            raw = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
            obj = _extract_json(raw) or {}
            pred = {f: str(obj.get(f, "") or "").strip() for f in FIELDS}
            error = ""
        except Exception as e:  # noqa: BLE001
            raw, pred, error = "", {f: "" for f in FIELDS}, \
                f"{type(e).__name__}: {e}"
        lat = (time.time() - t0) * 1000
        return raw, pred, error, lat

    pool = _fut.ThreadPoolExecutor(max_workers=1)
    consec_timeouts = 0
    with torch.no_grad():
        for i, item in enumerate(items):
            fut = pool.submit(_run_one, item)
            try:
                raw, pred, error, lat = fut.result(timeout=args.doc_timeout)
                consec_timeouts = 0
            except _fut.TimeoutError:
                consec_timeouts += 1
                raw, pred = "", {f: "" for f in FIELDS}
                error = f"TimeoutError: generate exceeded {args.doc_timeout}s"
                lat = args.doc_timeout * 1000.0
                print(f"  [{i + 1}/{len(items)}] TIMEOUT "
                      f"({consec_timeouts} consecutive)", flush=True)
                if consec_timeouts >= args.max_consec_timeouts:
                    print("  aborting run: too many consecutive timeouts",
                          flush=True)
                    preds.append({"id": item.get("id"),
                                  "image_path": item["image_path"],
                                  "pred": pred,
                                  "gold": {f: item["gold"].get(f, "")
                                           for f in FIELDS},
                                  "latency_ms": lat, "raw": "",
                                  "error": error + " (run aborted)",
                                  "score": score_example(
                                      pred, {f: item["gold"].get(f, "")
                                             for f in FIELDS})})
                    scores.append(preds[-1]["score"])
                    break
            sc = score_example(pred, gold := {f: item["gold"].get(f, "") for f in FIELDS})
            preds.append({"id": item.get("id"), "image_path": item["image_path"],
                          "pred": pred, "gold": gold, "latency_ms": lat,
                          "raw": raw[:2000], "error": error, "score": sc})
            scores.append(sc)
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                agg = macro_prf(scores)
                print(f"  [{i + 1}/{len(items)}] f1={agg['field_f1']:.3f}",
                      flush=True)

    pool.shutdown(wait=False, cancel_futures=True)
    agg = macro_prf(scores)
    out = {
        "engine": f"{args.tag_prefix}_{tag}{args.out_suffix}",
        "adapter": args.adapter,
        "n_examples": len(preds),
        "field_f1": agg["field_f1"],
        "schema_valid_rate": agg["schema_valid_rate"],
        "anls_mean": agg["anls_mean"],
        "avg_latency_ms": sum(p["latency_ms"] for p in preds) / max(len(preds), 1),
        "per_field_hit_rate": {
            f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores)
            for f in FIELDS
        },
        "errors": sum(1 for p in preds if p["error"]),
    }
    (RESULTS / f"scores_{args.tag_prefix}_{tag}{args.out_suffix}.json").write_text(
        json.dumps(out, indent=2))
    (RESULTS / f"preds_{args.tag_prefix}_{tag}{args.out_suffix}.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
