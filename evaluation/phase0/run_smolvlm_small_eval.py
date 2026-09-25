"""Stage-0 probe: zero-shot SmolVLM-500M on the held-out SROIE eval (paired).

Thesis test before spending any training compute: is a sub-1B base strong
enough zero-shot on receipts that SFT on real data could plausibly reach the
3B baseline (field F1 0.870, address 0.72)? Paired with the 3B run: identical
100 documents, identical field-F1 scorer, identical prompt text (EXTRACT_PROMPT
single source of truth), greedy decoding, no evidence pass.

Writes (under evaluation/phase0/results/):
  preds_smolvlm500m_zeroshot{suffix}.jsonl   per-doc {pred, gold, score, latency, raw}
  scores_smolvlm500m_zeroshot{suffix}.json  aggregate + per_field

Usage:
  python evaluation/phase0/run_smolvlm_small_eval.py --limit 2 --out-suffix _smoke
  python evaluation/phase0/run_smolvlm_small_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from metrics import FIELDS, macro_prf, score_example
from tinydoc.pipeline import EXTRACT_PROMPT, _extract_json

MODEL_DIR = (
    "/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/models/smolvlm500m"
)
RESULTS = ROOT / "evaluation/phase0/results"


def load_model(model_dir: str, longest_edge: int = 1024):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # SmolVLM encodes 512x512 patches and defaults to longest_edge=4*512 (2048px).
    # For receipts (typically ~640x480) that upscales 4x for zero information while
    # multiplying activation memory; cap it so the probe measures an edge-real config.
    processor = AutoProcessor.from_pretrained(
        model_dir, size={"longest_edge": longest_edge}
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, processor, device


def predict(model, processor, device: str, image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": EXTRACT_PROMPT}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True)


def mps_alloc_mb() -> float:
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def parse_fields(raw: str) -> dict:
    obj = _extract_json(raw) or {}
    return {k: str(obj.get(k, "") or "").strip() for k in FIELDS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--longest-edge", type=int, default=1024)
    args = ap.parse_args()

    items = json.loads((RESULTS / "sroie_eval.json").read_text())
    if args.limit:
        items = items[: args.limit]

    model, processor, device = load_model(args.model_dir, args.longest_edge)
    print(
        f"device={device} model={args.model_dir} n={len(items)} "
        f"longest_edge={args.longest_edge}",
        flush=True,
    )

    preds, scores = [], []
    peak_alloc = 0.0
    for i, item in enumerate(items):
        # stored image_path is repo-relative; never resolve() the symlink (AGENTS.md)
        img = Path(item["image_path"])
        if not img.is_absolute():
            img = ROOT / img
        gold = {f: item["gold"].get(f, "") for f in FIELDS}
        t0 = time.time()
        try:
            raw = predict(model, processor, device, str(img))
            pred = parse_fields(raw)
            error = ""
        except Exception as e:  # noqa: BLE001
            raw, pred, error = "", {f: "" for f in FIELDS}, f"{type(e).__name__}: {e}"
        lat = (time.time() - t0) * 1000
        # release per-doc activations; the MPS allocator otherwise caches and swap grows
        if device == "mps":
            torch.mps.empty_cache()
        peak_alloc = max(peak_alloc, mps_alloc_mb())
        sc = score_example(pred, gold)
        preds.append(
            {
                "id": item.get("id"),
                "image_path": item["image_path"],
                "pred": pred,
                "gold": gold,
                "latency_ms": lat,
                "raw": raw[:2000],
                "error": error,
                "score": sc,
            }
        )
        scores.append(sc)
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(items):
            agg = macro_prf(scores)
            print(
                f"  [{i+1}/{len(items)}] f1={agg['field_f1']:.3f} "
                f"lat={lat:.0f}ms mps_alloc={peak_alloc:.0f}MB",
                flush=True,
            )

    agg = macro_prf(scores)
    n_scored = sum(1 for p in preds if not p["error"])
    out = {
        "engine": "smolvlm500m_zeroshot",
        "model_dir": args.model_dir,
        "n_examples": len(preds),
        "scored_rows": n_scored,
        "field_f1": agg["field_f1"],
        "schema_valid_rate": agg["schema_valid_rate"],
        "anls_mean": agg["anls_mean"],
        "avg_latency_ms": sum(p["latency_ms"] for p in preds) / max(len(preds), 1),
        "peak_mps_alloc_mb": round(peak_alloc, 1),
        "longest_edge": args.longest_edge,
        "per_field_hit_rate": {
            f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores)
            for f in FIELDS
        },
        "errors": sum(1 for p in preds if p["error"]),
    }
    suffix = args.out_suffix
    (RESULTS / f"preds_smolvlm500m_zeroshot{suffix}.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n"
    )
    (RESULTS / f"scores_smolvlm500m_zeroshot{suffix}.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))

    # Gate G2 token: smoke means every doc produced non-empty output, no error.
    if suffix == "_smoke":
        ok = out["errors"] == 0 and all(
            any(v for v in p["pred"].values()) for p in preds
        )
        print("smoke_ok" if ok else "smoke_FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
