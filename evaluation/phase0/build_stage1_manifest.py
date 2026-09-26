"""Build the Stage-1 SFT manifest: real SROIE receipts -> image/question/answer.

Purpose: give the edge fine-tune exactly what the eval asks for, and nothing else.

  question = sdk.tinydoc.pipeline.EXTRACT_PROMPT   (single source of truth; the
              exact text the 0.870 baseline was scored under)
  answer   = compact JSON with the four fields, fixed key order

Leakage protection (three independent threats, measured 2026-09-25):

  T1  our 100-doc held-out eval (sroie_eval.json) overlaps the public 626-row
      SROIE train split (58-61% of receipts, verified two ways: perceptual
      correlation >= 0.90 with label agreement 61/61 on company name).
      Exact sha256 hashing finds 0 and must NEVER be trusted alone — mirrors
      re-encode images. Filtered by perceptual correlation (union of two
      resampler implementations at >= --leak-corr, so the result does not
      depend on PIL's resize default).
  T2  the public pool itself contains near-duplicate receipts (58 components:
      558 singletons, 17x2, 7x3, 1x4, 1x9). Splitting rows independently lets
      a duplicate pair straddle train/eval. Fixed by splitting COMPONENTS
      wholesale (union-find over the same correlation graph).
  T3  2 of 626 rows lack an empty gold field (row 33 total, row 104 address).
      Fixed by requiring all 4 fields everywhere (min-fields 4).

After writing, the builder re-loads its own splits and asserts:
  * no >= leak-corr pair crosses train/val/eval (either resampler)
  * no holdout-linked row in any split
  * every eval doc has all 4 gold fields and exists on disk
  * exact requested split sizes

Writes:
  evaluation/phase0/results/eval_leakage_audit.json       the T1 overlap measurement
  evaluation/phase0/results/stage1_manifest_summary.json  counts + integrity checks
  evaluation/phase0/results/sroie_eval_clean.json         the 100-doc ship-bar eval
  <out>/train  and  <out>/val   HF datasets (image, question, answer, messages)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))

from tinydoc.pipeline import EXTRACT_PROMPT

FIELDS = ("company", "date", "address", "total")
LABEL_FIELDS = ("company", "total", "date", "address")
RESULTS = ROOT / "evaluation/phase0/results"


def thumb(img, resample) -> np.ndarray:
    """64x64 grayscale thumbnail, explicitly resampled (PIL's default varies)."""
    return np.asarray(
        img.convert("L").resize((64, 64), resample), dtype=np.float32
    ).ravel()


def correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
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

    tags = dict(re.findall(r"<s_(\w+)>(.*?)</s_\1>", text, flags=re.DOTALL))
    if tags:
        for f in FIELDS:
            if f in tags:
                gold[f] = re.sub(r"\s+", " ", tags[f]).strip()
        if any(gold.values()):
            return gold

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {f: str(obj.get(f, "") or "").strip() for f in FIELDS}
    except Exception:  # noqa: BLE001, S110
        pass

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


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/stage1_data")
    ap.add_argument("--source", default="/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/data/sroie-labels",
                    help="local dir with data/train-*.parquet (rajistics/sroie)")
    ap.add_argument("--val-size", type=int, default=40)
    ap.add_argument("--eval-size", type=int, default=100,
                    help="clean held-out eval docs carved from the leakage-filtered pool")
    ap.add_argument("--min-fields", type=int, default=4,
                    help="require all gold fields (empty gold poisons the ship-bar eval)")
    ap.add_argument("--leak-corr", type=float, default=0.90,
                    help="exclude rows correlating >= this with any holdout image or split sibling")
    args = ap.parse_args()

    from datasets import Dataset, load_dataset
    from PIL import Image

    RES_BILINEAR = Image.Resampling.BILINEAR
    RES_NEAREST = Image.Resampling.NEAREST

    def both_thumbs(img) -> tuple[np.ndarray, np.ndarray]:
        return thumb(img, RES_BILINEAR), thumb(img, RES_NEAREST)

    # 1) leakage reference: our 100 held-out eval images
    holdout = json.loads((RESULTS / "sroie_eval.json").read_text())
    hb, hn = [], []
    holdout_hashes, holdout_ids = set(), []
    for item in holdout:
        p = ROOT / item["image_path"]
        if p.exists():
            with Image.open(p) as im:
                b, n = both_thumbs(im)
                hb.append(b)
                hn.append(n)
                holdout_hashes.add(image_hash(im))
                holdout_ids.append(item["id"])
    HB, HN = np.stack(hb), np.stack(hn)
    print(f"holdout images: {len(hb)} (exact hashes: {len(holdout_hashes)})")

    # 2) public SROIE train split (626, has labels)
    ds = load_dataset(
        args.source, data_files={"train": "data/train-*.parquet"}, split="train"
    )
    print(f"public train rows: {len(ds)}")

    # 3) T1 audit: which pool rows duplicate a holdout image (union of resamplers)
    pool_b, pool_n = [], []
    exact_leaks = 0
    golds, hashes, raw_pil = [], [], []
    for ex in ds:
        img = ex["image"]
        b, n = both_thumbs(img)
        pool_b.append(b)
        pool_n.append(n)
        if image_hash(img) in holdout_hashes:
            exact_leaks += 1
        golds.append(parse_gold(ex["text"]))
        hashes.append(image_hash(img))
        raw_pil.append(img)

    PB, PN = np.stack(pool_b), np.stack(pool_n)
    CB, CN = correlations(PB, HB), correlations(PN, HN)
    leak = (CB >= args.leak_corr) | (CN >= args.leak_corr)   # (626, 100)
    leak_row = leak.any(axis=1)
    score = np.maximum(CB, CN)
    best = np.where(leak_row, score.argmax(axis=1), -1)
    n_leak = int(leak_row.sum())
    n_holdout_dup = len({holdout_ids[int(best[i])] for i in np.where(leak_row)[0]})

    audit = {
        "question": "does the public SROIE train split contain the same receipt images as our 100-doc eval?",
        "n_holdout": len(hb),
        "n_public_train": len(ds),
        "exact_pixel_hash_leaks": exact_leaks,
        "perceptual_leaks_at_corr_0.90": n_leak,
        "holdout_fraction_leaked": round(n_holdout_dup / max(len(hb), 1), 4),
        "method": (
            "64x64 grayscale Pearson correlation, UNION of bilinear and nearest "
            "resampling (threshold does not depend on PIL resize default); "
            "sha256 exact hash reported for contrast. Label cross-check: "
            "company names agree on 61/61 matched pairs (independent verification 2026-09-26)."
        ),
        "implication": (
            "Our 100-doc SROIE eval is NOT a clean held-out set: it overlaps the public train "
            "split. Absolute scores on it are optimistically biased for any model trained on "
            "public SROIE (including possibly the evaluated 3B). Paired same-document "
            "comparisons remain internally valid; absolute held-out claims do not."
        ),
        "examples": [
            {"holdout_id": holdout_ids[int(best[i])], "corr_best": round(float(score[i, int(best[i])]), 4)}
            for i in np.where(leak_row)[0][:10]
        ],
    }
    (RESULTS / "eval_leakage_audit.json").write_text(json.dumps(audit, indent=2))
    print(
        f"LEAKAGE AUDIT (T1): exact-hash {exact_leaks}, perceptual {n_leak} rows "
        f"-> {n_holdout_dup}/{len(hb)} holdout receipts duplicated ({audit['holdout_fraction_leaked']:.0%})"
    )

    # 4) T2: components — never let a duplicate pair straddle splits
    uf = UnionFind(len(ds))
    edge = (correlations(PB, PB) >= args.leak_corr) | (correlations(PN, PN) >= args.leak_corr)
    np.fill_diagonal(edge, False)
    for i, j in zip(*np.where(np.triu(edge, 1))):
        uf.union(int(i), int(j))
    comp_of: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        comp_of[uf.find(i)].append(i)
    sizes = defaultdict(int)
    for v in comp_of.values():
        sizes[len(v)] += 1
    print(f"T2 components: {len(comp_of)} (sizes {dict(sorted(sizes.items()))})")

    # 5) keep rules: not holdout-linked, complete component, all 4 fields
    def complete(i: int) -> bool:
        return all(golds[i][f].strip() for f in FIELDS)

    dropped_linked = dropped_fields = 0
    rows_by_comp = []
    for members in sorted(comp_of.values(), key=lambda v: (len(v), v[0])):
        if any(leak_row[i] for i in members):
            dropped_linked += len(members)
            continue
        if not all(complete(i) for i in members):
            dropped_fields += len(members)
            continue
        rows_by_comp.append(members)
    usable = [i for m in rows_by_comp for i in m]
    print(f"usable rows: {len(usable)} (dropped linked {dropped_linked}, incomplete-field {dropped_fields})")

    # 6) component-aware deterministic splits: eval -> val -> train
    def take_components(pool: list[list[int]], target: int) -> tuple[list[int], list[list[int]]]:
        picked_rows: list[int] = []
        picked_comps: list[list[int]] = []
        for m in pool:
            if len(picked_rows) + len(m) > target:
                continue
            picked_rows.extend(m)
            picked_comps.append(m)
            if len(picked_rows) == target:
                break
        return picked_rows, picked_comps

    eval_rows_idx, eval_comps = take_components(rows_by_comp, args.eval_size)
    eval_comp_ids = {id(c) for c in eval_comps}
    val_pool = [m for m in rows_by_comp if id(m) not in eval_comp_ids]
    del eval_comps
    val_rows_idx, _val_comps = take_components(val_pool, args.val_size)
    assert len(eval_rows_idx) == args.eval_size, f"eval short: {len(eval_rows_idx)}"
    assert len(val_rows_idx) == args.val_size, f"val short: {len(val_rows_idx)}"
    taken = set(eval_rows_idx) | set(val_rows_idx)
    train_rows_idx = [i for i in usable if i not in taken]
    print(f"splits: eval={len(eval_rows_idx)} val={len(val_rows_idx)} train={len(train_rows_idx)}")

    # 7) write artifacts
    out = Path(args.out)
    eval_dir = out / "eval_images"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image", "question", "answer", "messages"]

    def build_row(i: int) -> dict:
        gold = golds[i]
        ans = answer_json(gold)
        return {
            "image": raw_pil[i],
            "question": EXTRACT_PROMPT,
            "answer": ans,
            "messages": messages_row(EXTRACT_PROMPT, ans),
            "gold": gold,
        }

    clean_eval = []
    for k, i in enumerate(eval_rows_idx):
        name = f"clean_{k:04d}.jpg"
        raw_pil[i].convert("RGB").save(eval_dir / name, quality=95)
        clean_eval.append({
            "id": f"clean_{k:04d}",
            "pool_row": int(i),
            "image_path": str(eval_dir / name),
            "source": "sroie_public_train_clean",
            "gold": golds[i],
        })
    (RESULTS / "sroie_eval_clean.json").write_text(json.dumps(clean_eval, indent=2))

    # export val the same way (checkpoint selection must score val, never test)
    val_dir = out / "val_images"
    val_dir.mkdir(exist_ok=True)
    val_set = []
    for k, i in enumerate(val_rows_idx):
        name = f"val_{k:04d}.jpg"
        raw_pil[i].convert("RGB").save(val_dir / name, quality=95)
        val_set.append({
            "id": f"val_{k:04d}",
            "pool_row": int(i),
            "image_path": str(val_dir / name),
            "source": "sroie_public_train_val",
            "gold": golds[i],
        })
    (RESULTS / "sroie_val.json").write_text(json.dumps(val_set, indent=2))

    Dataset.from_list([{k: r[k] for k in cols} for r in (build_row(i) for i in train_rows_idx)]).save_to_disk(out / "train")
    Dataset.from_list([{k: r[k] for k in cols} for r in (build_row(i) for i in val_rows_idx)]).save_to_disk(out / "val")

    # 8) self-assertions: re-verify from what was just written
    from datasets import load_from_disk

    tr = load_from_disk(str(out / "train"))
    va = load_from_disk(str(out / "val"))
    assert len(tr) == len(train_rows_idx) and len(va) == len(val_rows_idx)

    def as_pil(v):
        return v if isinstance(v, Image.Image) else Image.open(v) if isinstance(v, (str, Path)) else Image.open(__import__("io").BytesIO(v["bytes"]))

    checks = {"cross_split_pairs": 0, "holdout_linked_in_splits": 0,
              "eval_gold_incomplete": 0, "eval_files_missing": 0}
    fb = [thumb(as_pil(r["image"]), RES_BILINEAR) for r in tr]
    fn = [thumb(as_pil(r["image"]), RES_NEAREST) for r in tr]
    vb = [thumb(as_pil(r["image"]), RES_BILINEAR) for r in va]
    vn = [thumb(as_pil(r["image"]), RES_NEAREST) for r in va]
    eb = [thumb(Image.open(c["image_path"]), RES_BILINEAR) for c in clean_eval]
    en = [thumb(Image.open(c["image_path"]), RES_NEAREST) for c in clean_eval]
    for A, B in ((fb, eb), (fn, en), (vb, eb), (vn, en), (fb, vb), (fn, vn)):
        checks["cross_split_pairs"] += int((correlations(np.stack(A), np.stack(B)) >= args.leak_corr).sum())
    # eval must also be perceptually disjoint from the ORIGINAL (contaminated) holdout
    for E in (eb, en):
        checks["holdout_linked_in_splits"] += int(
            (correlations(np.stack(E), HB if E is eb else HN) >= args.leak_corr).any(axis=1).sum()
        )
    checks["eval_gold_incomplete"] = sum(
        1 for c in clean_eval if not all((c["gold"].get(f) or "").strip() for f in FIELDS)
    )
    checks["eval_files_missing"] = sum(1 for c in clean_eval if not Path(c["image_path"]).exists())
    assert checks == {"cross_split_pairs": 0, "holdout_linked_in_splits": 0,
                      "eval_gold_incomplete": 0, "eval_files_missing": 0}, checks
    print(f"integrity assertions passed: {checks}")

    summary = {
        "source": "rajistics/sroie (626-row public train split, images+labels)",
        "n_total_public": len(ds),
        "n_holdout_hashes": len(holdout_hashes),
        "n_leaked_excluded": n_leak,
        "leakage_filter": (
            f"perceptual 64x64 correlation >= {args.leak_corr} (union of bilinear and nearest "
            "resampling) against all holdout images; exact sha256 hashing misses re-encoded duplicates"
        ),
        "n_dropped_insufficient_fields": dropped_fields,
        "n_dropped_holdout_linked": dropped_linked,
        "n_components": len(comp_of),
        "duplicate_component_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "split_method": "whole duplicate components assigned deterministically: eval, then val, then train; never split across",
        "n_train": len(train_rows_idx),
        "n_val": len(val_rows_idx),
        "n_clean_eval": len(eval_rows_idx),
        "clean_eval_artifact": "evaluation/phase0/results/sroie_eval_clean.json",
        "min_fields_required": args.min_fields,
        "field_nonempty_counts": {f: sum(1 for i in usable if golds[i][f]) for f in FIELDS},
        "integrity_checks": checks,
        "question": "sdk.tinydoc.pipeline.EXTRACT_PROMPT (identical to eval prompt)",
    }
    (RESULTS / "stage1_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\ntrain/val datasets written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
