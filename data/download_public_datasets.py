#!/usr/bin/env python3
"""
Download and prepare public document datasets for TinyDoc-VLM training.

Writes, per dataset, into data/datasets/<name>/:
  <name>.json        - [{image_path (repo-relative), prompt, target, source}]
  images/img_<n>.jpg - resized to max edge 768 (JPEG q85)

Datasets (permissive licenses):
  DocVQA   lmms-lab/DocVQA          apache-2.0  real-document VQA (train split)
  SROIE    jsdnrs/ICDAR2019-SROIE   cc-by-4.0   scanned receipts (KIE + OCR)
  Docmatix HuggingFaceM4/Docmatix   MIT         real-PDF QA (sampled, streaming)

Resume-safe: existing pairs/images are reused and image hashes de-duplicated,
so an interrupted run can be re-launched safely.

Usage:
    python3 data/download_public_datasets.py [--docmatix-limit 20000]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "datasets"
MAX_EDGE = 768
JPEG_Q = 85

LICENSES = {
    "docvqa": "apache-2.0 (lmms-lab/DocVQA card)",
    "sroie": "cc-by-4.0 (ICDAR2019-SROIE)",
    "docmatix": "mit (HuggingFaceM4/Docmatix)",
}
SOURCES = {
    "docvqa": "docvqa",
    "sroie": "sroie",
    "docmatix": "docmatix",
}


def log(msg):
    logging.getLogger("download").info(msg)
    print(msg, flush=True)


def hf_token():
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        tf = REPO_ROOT / "training" / ".hf_token"
        if tf.exists():
            tok = tf.read_text().strip()
    return tok or None


def prep_image(pil, img_path: Path) -> tuple:
    """Resize to max edge 768, save JPEG q85, return (rel_path, sha1)."""
    w, h = pil.size
    scale = min(1.0, MAX_EDGE / max(w, h))
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), 1)  # LANCZOS
    buf = BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=JPEG_Q)
    data = buf.getvalue()
    img_path.write_bytes(data)
    return data, img_path


def save_pairs(name: str, pairs: list):
    path = OUT_DIR / name / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    tmp.replace(path)


def load_existing(name: str):
    """Return (pairs, img_hashes, img_index) from a previous run."""
    path = OUT_DIR / name / f"{name}.json"
    pairs = []
    if path.exists():
        pairs = [json.loads(l) for l in path.open() if l.strip()]
    hashes = {}
    img_idx = 0
    imgs_dir = OUT_DIR / name / "images"
    if imgs_dir.exists():
        for f in imgs_dir.glob("img_*.jpg"):
            idx = int(f.stem.split("_")[1])
            img_idx = max(img_idx, idx + 1)
            hashes[hashlib.sha1(f.read_bytes()).hexdigest()] = str(f)
    return pairs, hashes, img_idx


def iter_docvqa(limit):
    from datasets import load_dataset
    ds = None
    for split in ("train", "validation", "val", "test"):
        try:
            ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split,
                              streaming=True, token=hf_token())
            log(f"[docvqa] using split '{split}'")
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("no usable DocVQA split")
    n = 0
    for row in ds:
        if n >= limit:
            break
        image = None
        if "image" in row and hasattr(row["image"], "size"):
            image = row["image"]
        elif "images" in row:
            imgs = row["images"]
            if isinstance(imgs, (list, tuple)) and imgs and hasattr(imgs[0], "size"):
                image = imgs[0]
        q = row.get("question", "")
        a = row.get("answer", row.get("answers", ""))
        if isinstance(a, (list, tuple)):
            a = next((x for x in a if isinstance(x, str) and x.strip()), "")
        if image is None or not isinstance(q, str) or not isinstance(a, str):
            continue
        n += 1
        yield image, f"Answer the question: {q}", a


def iter_sroie():
    from datasets import load_dataset
    for split in ("train", "test"):
        try:
            ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split=split,
                              streaming=True, token=hf_token())
        except Exception:
            continue
        for row in ds:
            image = row.get("image")
            if image is None:
                continue
            ents = row.get("entities", {})
            ent = {k: v for k, v in ents.items() if v}
            words = row.get("words", [])
            yield image, "Extract the document as JSON:", json.dumps(ent, ensure_ascii=False)
            if words:
                yield image, "Extract all text:", " ".join(words)


def iter_docmatix(limit):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/Docmatix", "images", split="train",
                      streaming=True, token=hf_token())
    images_seen = 0
    for row in ds:
        if images_seen >= limit:
            break
        imgs = row.get("images", [])
        if not imgs:
            continue
        image = imgs[0]
        n_pairs = 0
        for t in row.get("texts", []):
            q = t.get("user", "")
            a = t.get("assistant", "")
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            if not q or not (2 <= len(a) <= 250):
                continue
            if "unanswerable" in a.lower() or a.lower().startswith("i cannot"):
                continue
            if a.count("\n") > 2:
                continue
            if "\x00" in a or "\ufffd" in a:
                continue
            yield image, f"Answer the question: {q}", a
            n_pairs += 1
            if n_pairs >= 2:
                break
        images_seen += 1


def run(name, iterator, limit=None):
    log(f"[{name}] starting (limit={limit or 'none'})")
    pairs, hashes, img_idx = load_existing(name)
    log(f"[{name}] resuming with {len(pairs)} pairs, {len(hashes)} images")
    out_dir = OUT_DIR / name
    imgs_dir = out_dir / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    known = set()
    for p in pairs:
        known.add((p.get("image_path"), p.get("prompt"), p.get("target")))
    new_pairs = 0
    for image, prompt, target in iterator:
        data, rel = prep_image(image, imgs_dir / f"img_{img_idx:06d}.jpg")
        h = hashlib.sha1(data).hexdigest()
        if h in hashes:
            rel = hashes[h]
            rel = Path(rel)
        else:
            hashes[h] = str(rel)
            img_idx += 1
        rec = (str(rel), prompt, target)
        if rec in known:
            continue
        known.add(rec)
        pairs.append({
            "image_path": str(rel),
            "prompt": prompt,
            "target": target,
            "source": SOURCES[name],
        })
        new_pairs += 1
        if new_pairs % 500 == 0:
            save_pairs(name, pairs)
            log(f"[{name}] {len(pairs)} pairs, {len(hashes)} images")
    save_pairs(name, pairs)
    meta = {
        "name": name,
        "license": LICENSES[name],
        "n_images": len(hashes),
        "n_pairs": len(pairs),
        "max_edge": MAX_EDGE,
    }
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[{name}] DONE: {len(pairs)} pairs, {len(hashes)} images")
    return len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docmatix-limit", type=int, default=20000)
    ap.add_argument("--docvqa-limit", type=int, default=60000)
    ap.add_argument("--only", type=str, default=None,
                    help="only download one dataset: docvqa|sroie|docmatix")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"output: {OUT_DIR} | free disk ok")

    targets = [args.only] if args.only else ["docvqa", "sroie", "docmatix"]
    total = 0
    for name in targets:
        try:
            if name == "docvqa":
                total += run(name, iter_docvqa(args.docvqa_limit), args.docvqa_limit)
            elif name == "sroie":
                total += run(name, iter_sroie())
            elif name == "docmatix":
                total += run(name, iter_docmatix(args.docmatix_limit), args.docmatix_limit)
        except Exception as e:
            log(f"[{name}] FAILED: {e!r}")
            sys.exit(1)
    log(f"ALL DONE: {total} total pairs")
    # datasets' streaming spawns a multiprocessing resource_tracker that can
    # hang interpreter shutdown on macOS; all data is flushed already.
    os._exit(0)


if __name__ == "__main__":
    main()
