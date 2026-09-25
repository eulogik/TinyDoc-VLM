"""Build the Stage-1 SFT manifest: real SROIE receipts -> image/question/answer.

Purpose: give the edge fine-tune exactly what the eval asks for, and nothing else.

  question = sdk.tinydoc.pipeline.EXTRACT_PROMPT   (single source of truth; the
              exact text the 0.870 baseline was scored under)
  answer   = compact JSON with the four fields, fixed key order

Leakage protection: our 100-doc held-out eval was drawn from the SROIE image pool
WITHOUT respecting the official train/test boundary, so some eval images are very
likely inside the 626-row public train split. Filenames differ across mirrors AND
the same receipt is re-encoded across mirrors, so an exact file/pixel hash is NOT
sufficient — a perceptual match is required:

  MEASURED 2026-09-25: sha256 pixel hashing found 0 overlaps, but 64x64
  normalized cross-correlation found 55/100 holdout receipts visually identical
  to train rows (corr >= 0.99). Exact hashing is blind to re-encoding; this
  builder therefore excludes on perceptual correlation >= --leak-corr (0.90).

Writes:
  evaluation/phase0/results/eval_leakage_audit.json    the overlap measurement
  evaluation/phase0/results/stage1_manifest_summary.json  counts after filtering
  <out>/train  and  <out>/val   HF datasets (image, question, answer)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))

from tinydoc.pipeline import EXTRACT_PROMPT  # noqa: E402

FIELDS = ("company", "date", "address", "total")
RESULTS = ROOT / "evaluation/phase0/results"


def thumb(img) -> np.ndarray:
    """64x64 grayscale thumbnail for perceptual comparison."""
    return np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32).ravel()


def correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    return a @ b.T


def image_hash(img) -> str:
    """Exact content hash (kept for the audit; insufficient alone)."""
    if img.mode != "L":
        img = img.convert("L")
    return hashlib.sha256(img.tobytes()).hexdigest()


def parse_gold(text: str) -> dict:
    """SROIE labels arrive as <s_field>value</s_field> tags; tolerate JSON and
    key: value line forms too. Never guess a field that is absent."""
    text = (text or "").strip()
    gold: dict = {f: "" for f in FIELDS}

    # 1) native SROIE tag format: <s_total>9.00</s_total>...
    tags = dict(re.findall(r"<s_(\w+)>(.*?)</s_\1>", text, flags=re.S))
    if tags:
        for f in FIELDS:
            if f in tags:
                gold[f] = re.sub(r"\s+", " ", tags[f]).strip()
        if any(gold.values()):
            return gold

    # 2) JSON object
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {f: str(obj.get(f, "") or "").strip() for f in FIELDS}
    except Exception:  # noqa: BLE001
        pass

    # 3) key: value lines
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower()
            if k in gold:
                gold[k] = v.strip()
    return gold


def answer_json(gold: dict) -> str:
    return json.dumps({f: gold[f] for f in FIELDS}, ensure_ascii=False, separators=(", ", ": "))


def messages_row(question: str, answer: str) -> list:
    """mlx-vlm VisionDataset expects a LIST of role/content turns (the
    --custom-prompt-format output is a dict and crashes the chat template)."""
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/stage1_data")
    ap.add_argument("--source", default="/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/data/sroie-labels",
                    help="local dir with data/train-*.parquet (rajistics/sroie)")
    ap.add_argument("--val-size", type=int, default=40)
    ap.add_argument("--eval-size", type=int, default=100,
                    help="clean held-out eval docs carved from the leakage-filtered pool")
    ap.add_argument("--min-fields", type=int, default=3,
                    help="require at least this many non-empty gold fields")
    ap.add_argument("--leak-corr", type=float, default=0.90,
                    help="exclude train rows whose image correlates >= this with any holdout image")
    args = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image

    # 1) leakage reference: our 100 held-out eval images (thumbnails + exact hashes)
    holdout = json.loads((RESULTS / "sroie_eval.json").read_text())
    holdout_thumbs, holdout_hashes, holdout_ids = [], set(), []
    for item in holdout:
        p = ROOT / item["image_path"]
        if p.exists():
            with Image.open(p) as im:
                holdout_thumbs.append(thumb(im))
                holdout_hashes.add(image_hash(im))
                holdout_ids.append(item["id"])
    holdout_arr = np.stack(holdout_thumbs) if holdout_thumbs else np.zeros((0, 4096), np.float32)
    print(f"holdout images: {len(holdout_thumbs)} (exact hashes: {len(holdout_hashes)})")

    # 2) public SROIE train split (626, has labels)
    ds = load_dataset(
        args.source, data_files={"train": "data/train-*.parquet"}, split="train"
    )
    print(f"public train rows: {len(ds)}")

    # 3) perceptual leakage audit: which train rows duplicate a holdout image
    train_thumbs, exact_leaks = [], 0
    for ex in ds:
        train_thumbs.append(thumb(ex["image"]))
        if image_hash(ex["image"]) in holdout_hashes:
            exact_leaks += 1
    train_arr = np.stack(train_thumbs)
    corr = correlations(train_arr, holdout_arr)  # (n_train, n_holdout)
    max_corr = corr.max(axis=1) if corr.size else np.zeros(len(train_arr), np.float32)
    best_holdout = corr.argmax(axis=1) if corr.size else np.zeros(len(train_arr), int)
    leak_mask = max_corr >= args.leak_corr
    n_leak = int(leak_mask.sum())

    audit = {
        "question": "does the public SROIE train split contain the same receipt images as our 100-doc eval?",
        "n_holdout": len(holdout_thumbs),
        "n_public_train": len(train_thumbs),
        "exact_pixel_hash_leaks": exact_leaks,
        "perceptual_leaks_at_corr_0.90": n_leak,
        "holdout_fraction_leaked": round(
            len({holdout_ids[int(best_holdout[i])] for i in np.where(leak_mask)[0]})
            / max(len(holdout_ids), 1),
            4,
        ),
        "method": "64x64 grayscale normalized cross-correlation (Pearson); sha256 exact hash reported for contrast",
        "implication": (
            "Our 100-doc SROIE eval is NOT a clean held-out set: it overlaps the public train "
            "split. Absolute scores on it are optimistically biased for any model trained on "
            "public SROIE (including possibly the evaluated 3B). Paired same-document "
            "comparisons remain internally valid; absolute held-out claims do not."
        ),
        "examples": [
            {"holdout_id": holdout_ids[int(best_holdout[i])], "corr": round(float(max_corr[i]), 4)}            for i in np.where(leak_mask)[0][:10]
        ],
    }
    (RESULTS / "eval_leakage_audit.json").write_text(json.dumps(audit, indent=2))
    print(
        f"LEAKAGE AUDIT: exact-hash {exact_leaks}, perceptual(>={args.leak_corr}) {n_leak} "
        f"of {len(train_thumbs)} train rows duplicate a holdout image"
    )

    rows, dropped_empty = [], 0
    for i, ex in enumerate(ds):
        if bool(leak_mask[i]):
            continue
        gold = parse_gold(ex["text"])
        nonempty = sum(1 for f in FIELDS if gold[f])
        if nonempty < args.min_fields:
            dropped_empty += 1
            continue
        ans = answer_json(gold)
        rows.append({
            "image": ex["image"],
            "question": EXTRACT_PROMPT,
            "answer": ans,
            "messages": messages_row(EXTRACT_PROMPT, ans),
            "gold": gold,
        })

    print(f"leaked (excluded by perceptual match): {n_leak}")
    print(f"dropped (fewer than {args.min_fields} gold fields): {dropped_empty}")
    print(f"usable training rows: {len(rows)}")

    # 4) deterministic splits: clean EVAL first (the ship-bar), then val, then train
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = out / "eval_images"
    eval_dir.mkdir(exist_ok=True)
    from datasets import Dataset

    cols = ["image", "question", "answer", "messages"]
    eval_rows = rows[: args.eval_size]
    val_rows = rows[args.eval_size : args.eval_size + args.val_size]
    train_rows = rows[args.eval_size + args.val_size :]

    # export clean eval images to disk so any runner can use file paths
    clean_eval = []
    for i, r in enumerate(eval_rows):
        img = r["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        name = f"clean_{i:04d}.jpg"
        img.convert("RGB").save(eval_dir / name, quality=95)
        clean_eval.append(
            {
                "id": f"clean_{i:04d}",
                "image_path": str(eval_dir / name),
                "source": "sroie_public_train_clean",
                "gold": r["gold"],
            }
        )
    (RESULTS / "sroie_eval_clean.json").write_text(json.dumps(clean_eval, indent=2))

    Dataset.from_list([{k: r[k] for k in cols} for r in train_rows]).save_to_disk(out / "train")
    Dataset.from_list([{k: r[k] for k in cols} for r in val_rows]).save_to_disk(out / "val")

    summary = {
        "source": "rajistics/sroie (official 626-row train split, images+labels)",
        "n_total_public": len(ds),
        "n_holdout_hashes": len(holdout_hashes),
        "n_leaked_excluded": n_leak,
        "leakage_filter": f"perceptual 64x64 correlation >= {args.leak_corr} against all holdout images (exact sha256 hashing misses re-encoded duplicates)",
        "n_dropped_insufficient_fields": dropped_empty,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_clean_eval": len(eval_rows),
        "clean_eval_artifact": "evaluation/phase0/results/sroie_eval_clean.json",
        "min_fields_required": args.min_fields,
        "field_nonempty_counts": {f: sum(1 for r in rows if r["gold"][f]) for f in FIELDS},
        "question": "sdk.tinydoc.pipeline.EXTRACT_PROMPT (identical to eval prompt)",
    }
    (RESULTS / "stage1_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\ntrain/val datasets written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
