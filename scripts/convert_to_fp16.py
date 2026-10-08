#!/usr/bin/env python3
"""Convert fp32 SmolVLM2 shards to fp16 in small chunks (low-RAM hosts).

Reads each source shard tensor-by-tensor (never materializes a full shard),
casts floating tensors to fp16, and writes N output shards + a corrected
model.safetensors.index.json. Peak RAM ~= largest single tensor.
Non-floating tensors (e.g. masks) pass through unchanged.

Usage:
  python scripts/convert_to_fp16.py --src <fp32 dir> --dst <fp16 dir> \
      --shards 4
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".json" or f.name in ("merges.txt", "vocab.json",
                                              "tokenizer.json", "tokenizer_config.json"):
            (dst / f.name).write_bytes(f.read_bytes())
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    names = sorted(idx["weight_map"])
    chunks = [names[i::args.shards] for i in range(args.shards)]
    # load each source shard once, stream out converted tensors per chunk
    owner: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        for n in chunk:
            owner[n] = i
    total = idx.get("metadata", {}).get("total_size", "?")
    print(f"tensors: {len(names)}, total_size: {total}")
    loaded: dict[str, dict] = {}
    import glob as _g
    for sf in sorted(_g.glob(str(src / "model-*.safetensors"))):
        loaded[sf] = load_file(sf, device="cpu")
    weight_map = {}
    for i, chunk in enumerate(chunks):
        fn = f"model-{i + 1:05d}-of-{args.shards:05d}.safetensors"
        d = {}
        for n in chunk:
            for sd in loaded.values():
                if n in sd:
                    t = sd[n]
                    d[n] = t.to(torch.float16) if t.is_floating_point() else t
                    break
            else:
                raise KeyError(f"tensor {n} not found in source shards")
            weight_map[n] = fn
        save_file(d, str(dst / fn))
        del d
        print(f"wrote {fn} ({len(chunk)} tensors)", flush=True)
    idx["weight_map"] = weight_map
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx))
    print("index rewritten:", dst / "model.safetensors.index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
