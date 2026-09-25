"""Real OCRBench evaluation for TinyDoc-VLM checkpoints.

Replaces the placeholder scorer in evaluation/evaluate.py, which returned hardcoded
zeros. Implements per-type metrics over the public 1000-sample OCRBench
reconstruction, persists per-sample predictions, and reports both a strict
exact-match and a lenient edit-distance variant so the score cannot be inflated
by metric choice.

Also reconciles the image_token_id defect: modeling.py selects visual features
with `input_ids == self.image_token_id` and silently skips insertion when the
count is zero, so a processor/model id mismatch yields a blind model with no
error. This script reports both ids and can force alignment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).absolute().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

import tinydoc_vlm  # noqa: F401  registers the tinydoc_vlm config/model
from tinydoc_vlm.image_processing import TinyDocImageProcessor
from tinydoc_vlm.processing import TinyDocVLMProcessor

RECOGNITION_TYPES = {
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
}
VQA_TYPES = {
    "Scene Text-centric VQA",
    "Doc-oriented VQA",
    "Key Information Extraction",
}
MATH_TYPE = "Handwritten Mathematical Expression Recognition"


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_math(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def score_sample(prediction: str, answers: list[str], item_type: str) -> dict:
    pred = prediction.strip()
    if not pred:
        return {"strict": 0.0, "lenient": 0.0, "best_edit_distance": None}

    if item_type in VQA_TYPES:
        target_norm = [normalize(a) for a in answers]
        pred_norm = normalize(pred)
        strict = 1.0 if pred_norm in target_norm else 0.0
        best = min(edit_distance(pred_norm, t) for t in target_norm)
        denom = max(len(pred_norm), 1)
        lenient = 1.0 if best / denom <= 0.5 else 0.0
        return {"strict": strict, "lenient": lenient, "best_edit_distance": best}

    if item_type == MATH_TYPE:
        target_norm = [normalize_math(a) for a in answers]
        pred_norm = normalize_math(pred)
        best = min(edit_distance(pred_norm, t) for t in target_norm)
        strict = 1.0 if pred_norm in target_norm else 0.0
        lenient = 1.0 if best / max(len(pred_norm), 1) <= 0.5 else 0.0
        return {"strict": strict, "lenient": lenient, "best_edit_distance": best}

    target_norm = [normalize(a) for a in answers]
    pred_norm = normalize(pred)
    strict = 1.0 if pred_norm in target_norm else 0.0
    best = min(edit_distance(pred_norm, t) for t in target_norm)
    lenient = 1.0 if best / max(len(pred_norm), 1) <= 0.5 else 0.0
    return {"strict": strict, "lenient": lenient, "best_edit_distance": best}


def load_samples(data_path: Path, limit: int | None) -> list[dict]:
    items = json.loads((data_path / "ocrbench.json").read_text())
    if limit:
        items = items[:limit]
    return items


def resolve_image(data_path: Path, relative: str) -> Path | None:
    stem = Path(relative).name
    for candidate in (data_path / "images" / stem, data_path / relative, ROOT / relative):
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="eulogik/TinyDoc-VLM-256M")
    parser.add_argument("--data-dir", default="evaluation/data/ocrbench")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-token-id", default="processor",
                        choices=["processor", "config", "49152", "49153"])
    parser.add_argument("--out-prefix", default="evaluation/phase0/results/ocrbench_256m")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    samples = load_samples(data_path, args.limit)
    if not samples:
        print("no samples", file=sys.stderr)
        return 1

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps unavailable, falling back to cpu", file=sys.stderr)
        device = "cpu"

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval().to(device)

    # The published processor_config declares TinyDocVLMProcessor, but that class is
    # not a HF ProcessorMixin so AutoProcessor silently degrades to a bare tokenizer.
    # Build it explicitly at the checkpoint's native resolution instead.
    if not hasattr(processor, "image_processor"):
        image_size = int(getattr(model.config, "image_size", 384))
        processor = TinyDocVLMProcessor(
            image_processor=TinyDocImageProcessor(image_size=image_size, tiling_mode="none"),
            tokenizer=AutoTokenizer.from_pretrained(args.model),
            config=model.config,
        )
        print(json.dumps({"processor": "rebuilt_explicit", "image_size": image_size}), flush=True)

    processor_id = getattr(processor, "image_token_id", None)
    model_id = getattr(model, "image_token_id", None)
    if args.image_token_id == "processor":
        forced = processor_id
    elif args.image_token_id == "config":
        forced = model_id
    else:
        forced = int(args.image_token_id)
    model.image_token_id = forced
    id_info = {
        "processor_image_token_id": processor_id,
        "model_image_token_id_before": model_id,
        "forced_image_token_id": forced,
        "aligned": processor_id == forced,
    }
    print(json.dumps({"image_token_id_reconciliation": id_info}), flush=True)

    records: list[dict] = []
    per_type_strict: dict[str, list[float]] = defaultdict(list)
    per_type_lenient: dict[str, list[float]] = defaultdict(list)
    started = time.time()

    for index, item in enumerate(samples):
        image_path = resolve_image(data_path, item["image"])
        record = {
            "index": index,
            "image": item["image"],
            "question": item["question"],
            "answers": item["answers"],
            "type": item["type"],
            "prediction": "",
            "error": None,
        }
        if image_path is None:
            record["error"] = "image_missing"
            records.append(record)
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            # The <image> tag is mandatory: the processor expands it into the 64
            # visual placeholder tokens that modeling.py uses to splice in SigLIP
            # features. Omitting it yields num_places == 0, the insertion branch is
            # skipped, and the model runs blind while still returning plausible text.
            inputs = processor(
                text=item["question"] + " <image>", images=image, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                        do_sample=False)
            prompt_len = inputs["input_ids"].shape[1]
            record["prediction"] = processor.tokenizer.decode(
                output[0][prompt_len:], skip_special_tokens=True
            )
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        if record["error"] is None:
            scored = score_sample(record["prediction"], item["answers"], item["type"])
            record.update(scored)
            per_type_strict[item["type"]].append(scored["strict"])
            per_type_lenient[item["type"]].append(scored["lenient"])
        else:
            record.update({"strict": 0.0, "lenient": 0.0, "best_edit_distance": None})
            per_type_strict[item["type"]].append(0.0)
            per_type_lenient[item["type"]].append(0.0)
        records.append(record)
        if (index + 1) % 25 == 0 or index + 1 == len(samples):
            elapsed = time.time() - started
            rate = (index + 1) / elapsed if elapsed else 0.0
            print(
                f"[{index + 1}/{len(samples)}] {elapsed:.1f}s {rate:.2f}/s",
                flush=True,
            )

    scored_records = [r for r in records if r["error"] is None]
    n = len(scored_records)
    strict_mean = sum(r["strict"] for r in scored_records) / n if n else 0.0
    lenient_mean = sum(r["lenient"] for r in scored_records) / n if n else 0.0
    errors = [r for r in records if r["error"]]

    summary = {
        "model": args.model,
        "tag": args.tag,
        "image_token_id_reconciliation": id_info,
        "device": device,
        "total_samples": len(records),
        "scored_samples": n,
        "error_count": len(errors),
        "ocrbench_score_strict": round(strict_mean * 100, 4),
        "ocrbench_score_lenient": round(lenient_mean * 100, 4),
        "per_type": {
            t: {
                "n": len(v),
                "strict": round(sum(v) / len(v) * 100, 4),
                "lenient": round(
                    sum(per_type_lenient[t]) / len(per_type_lenient[t]) * 100, 4
                ),
            }
            for t, v in sorted(per_type_strict.items())
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "scorer": "reimplementation: strict=casefold exact match vs any answer; "
                  "lenient=normalized edit distance <=0.5 of prediction length",
    }

    out_base = ROOT / args.out_prefix if not args.out_prefix.startswith("/") else Path(args.out_prefix)
    if args.tag:
        out_base = Path(f"{out_base}_{args.tag}")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    with out_base.with_suffix(".jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"VISION_TOWER_ALIVE_PROBE see summary; wrote {out_base}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
