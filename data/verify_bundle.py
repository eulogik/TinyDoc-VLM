#!/usr/bin/env python3
"""Simulate Kaggle consumption of training.tar.gz: extract and verify that
every manifest image_path resolves from the repo root (as full_train.py does
with cwd=REPO). Exit non-zero on any missing image.

Usage:
    python3 data/verify_bundle.py [path-to-training.tar.gz]
"""

import json
import sys
import tarfile
import tempfile
from pathlib import Path

bundle = Path(sys.argv[1] if len(sys.argv) > 1 else "data/training/training.tar.gz")
tmp = Path(tempfile.mkdtemp(prefix="bundle_check_"))
data_dir = tmp / "data" / "training"
print(f"extracting {bundle} -> {data_dir} ...")
with tarfile.open(bundle) as tf:
    tf.extractall(data_dir)

manifest = data_dir / "manifest.jsonl"
if not manifest.exists():
    sys.exit("FAIL: manifest.jsonl missing after extraction")

entries = [json.loads(l) for l in manifest.open() if l.strip()]
missing = []
seen = set()
for e in entries:
    p = tmp / e["image_path"]
    if not p.exists():
        missing.append(e["image_path"])
    else:
        seen.add(str(p))

print(f"pairs: {len(entries)} | unique images found: {len(seen)}")
if missing:
    print(f"FAIL: {len(missing)} missing image paths, e.g.:")
    for m in missing[:10]:
        print("  ", m)
    sys.exit(1)
from collections import Counter
by_src = Counter(e.get("source", "?") for e in entries)
print("PASS: all image paths resolve | by_source:", dict(by_src))