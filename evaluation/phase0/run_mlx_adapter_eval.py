"""Evaluate a trained SmolVLM LoRA adapter on the CLEAN SROIE eval (MLX).

The ship-bar: paired against the 3B on the same 100 clean receipts, same scorer
(same metrics.py as every other engine). This is the gate that decides whether
the first-party model ships.

Usage (inside the MLX venv):
  python evaluation/phase0/run_mlx_adapter_eval.py \
      --adapter $SCRATCH/adapters/full/adapters.safetensors --limit 100
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

SCRATCH = "/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode"
RESULTS = ROOT / "evaluation/phase0/results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{SCRATCH}/models/smolvlm500m")
    ap.add_argument("--adapter", default=None,
                    help="adapter dir (or adapters.safetensors); omit for base-model ablation")
    ap.add_argument("--eval-path", default=str(RESULTS / "sroie_eval_clean.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-suffix", default="_clean")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    from mlx_vlm.utils import load as load_model

    adapter = None
    if args.adapter:
        adapter = Path(args.adapter)
        if adapter.is_file():          # accept .../adapters.safetensors or its dir
            adapter = adapter.parent
        assert (adapter / "adapter_config.json").exists(), f"no adapter_config.json in {adapter}"
        adapter = str(adapter)

    items = json.loads(Path(args.eval_path).read_text())
    if args.limit:
        items = items[: args.limit]

    if adapter:
        model, processor = load_model(args.model, adapter_path=adapter)
    else:
        model, processor = load_model(args.model)
    tag = "lora" if adapter else "base"

    # mirror the trainer's prompt construction exactly (mlx_vlm.prompt_utils;
    # the transformers processor has NO chat template for this model)
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_chat

    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": EXTRACT_PROMPT}]}]
    try:
        prompt = mlx_chat(processor, getattr(model, "config", {}), conv,
                          add_generation_prompt=True, num_images=1)
    except Exception:  # noqa: BLE001 — last-resort plain marker
        prompt = f"<image>\n{EXTRACT_PROMPT}"
    print(f"model={args.model} adapter={adapter} n={len(items)}", flush=True)

    preds, scores = [], []
    for i, item in enumerate(items):
        img = item["image_path"]
        gold = {f: item["gold"].get(f, "") for f in FIELDS}
        t0 = time.time()
        try:
            from mlx_vlm.generate import generate as mlx_generate

            out = mlx_generate(
                model,
                processor,
                prompt,
                image=img,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            raw = getattr(out, "text", None)
            raw = raw if isinstance(raw, str) else str(out)
            obj = _extract_json(raw) or {}
            pred = {f: str(obj.get(f, "") or "").strip() for f in FIELDS}
            error = ""
        except Exception as e:  # noqa: BLE001
            raw, pred, error = "", {f: "" for f in FIELDS}, f"{type(e).__name__}: {e}"
        lat = (time.time() - t0) * 1000
        sc = score_example(pred, gold)
        preds.append(
            {
                "id": item.get("id"),
                "image_path": img,
                "pred": pred,
                "gold": gold,
                "latency_ms": lat,
                "raw": raw[:2000],
                "error": error,
                "score": sc,
            }
        )
        scores.append(sc)
        if (i + 1) % 20 == 0 or i == 0 or i + 1 == len(items):
            agg = macro_prf(scores)
            print(f"  [{i+1}/{len(items)}] f1={agg['field_f1']:.3f}", flush=True)

    agg = macro_prf(scores)
    out = {
        "engine": f"smolvlm500m_{tag}{args.out_suffix}",
        "adapter": adapter,
        "n_examples": len(preds),
        "field_f1": agg["field_f1"],
        "schema_valid_rate": agg["schema_valid_rate"],
        "anls_mean": agg["anls_mean"],
        "avg_latency_ms": sum(p["latency_ms"] for p in preds) / max(len(preds), 1),
        "per_field_hit_rate": {
            f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores) for f in FIELDS
        },
        "errors": sum(1 for p in preds if p["error"]),
    }
    (RESULTS / f"scores_smolvlm500m_{tag}{args.out_suffix}.json").write_text(json.dumps(out, indent=2))
    (RESULTS / f"preds_smolvlm500m_{tag}{args.out_suffix}.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n"
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
