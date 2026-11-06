#!/usr/bin/env python3
"""Grounding probe: can a base VLM place document fields in the right box?

This is the measurement the product needs before it can claim VLM-predicted
boxes. It reads a synthetic grounding manifest (perfect [0,1000] bboxes from
the renderer), asks the model for one box per field, and reports IoU.

It does NOT run by default. Generating the manifest and running this over
multiple documents is model inference and needs explicit approval.

Prepare data (CPU renderer, no model):
    python data/synthetic/gen_grounding_sample.py --num-docs 30 \
        --output-dir /Volumes/KIOXIA\\ 1TB/tinydoc/grounding_probe

Run (needs Ollama):
    python evaluation/phase0/run_grounding_probe.py \
        --manifest /Volumes/KIOXIA\\ 1TB/tinydoc/grounding_probe/manifest.jsonl \
        --model qwen2.5vl:3b --limit 100 \
        --out evaluation/phase0/results/grounding_probe_qwen2.5vl_3b.json

The manifest rows are `{image_path, prompt, target}`; `target` is
`<box>x1 y1 x2 y2</box>` in [0,1000] coordinates. JSON `[x1,y1,x2,y2]` is also
accepted. Pixel outputs are converted using the image size.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
BOX_RE = re.compile(r"<box>\s*([^<]+?)\s*</box>", re.IGNORECASE)


def parse_box(text: str, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Parse a box from `<box>...</box>`, a JSON list, or a bare 4-number list.

    Returns [0,1000] coordinates with x1<=x2, y1<=y2, or None.
    """
    raw: str | None = None
    match = BOX_RE.search(text or "")
    if match:
        raw = match.group(1)
    else:
        # JSON list or object with a bbox field.
        try:
            obj = json.loads((text or "").strip())
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            obj = obj.get("bbox") or obj.get("box")
        if isinstance(obj, (list, tuple)) and len(obj) >= 4:
            raw = " ".join(str(v) for v in obj[:4])
        elif obj is None:
            # Last resort: first four numbers in the text.
            raw = " ".join(re.findall(r"-?\d+(?:\.\d+)?", text or "")[:4])

    if not raw:
        return None
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]

    # Normalize common coordinate conventions to [0,1000].
    finite = [abs(v) for v in (x1, y1, x2, y2)]
    if max(finite) <= 1.5:
        x1, y1, x2, y2 = (v * 1000.0 for v in (x1, y1, x2, y2))
    elif max(finite) > 1000.0:
        x1, x2 = x1 * 1000.0 / max(width, 1), x2 * 1000.0 / max(width, 1)
        y1, y2 = y1 * 1000.0 / max(height, 1), y2 * 1000.0 / max(height, 1)

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (
        min(max(x1, 0.0), 1000.0),
        min(max(y1, 0.0), 1000.0),
        min(max(x2, 0.0), 1000.0),
        min(max(y2, 0.0), 1000.0),
    )


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def resolve_image(manifest: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.exists():
        return candidate
    for path in (
        manifest.parent / image_path,
        manifest.parent / "images" / Path(image_path).name,
    ):
        if path.exists():
            return path
    raise FileNotFoundError(image_path)


def ask_ollama(
    host: str, model: str, prompt: str, b64: str, num_ctx: int, timeout: float
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 96,
            "num_ctx": num_ctx,
        },
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return (body.get("message") or {}).get("content", "") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--num-ctx", type=int, default=int(os.environ.get("TINYDOC_OLLAMA_NUM_CTX", "8192")))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("TINYDOC_OLLAMA_TIMEOUT", "600")))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    rows = [r for r in rows if "<box>" in str(r.get("target", "")).lower()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"no <box> rows in {manifest}", file=sys.stderr)
        return 2

    host = args.host.rstrip("/")
    out_rows: list[dict] = []
    latencies: list[float] = []
    for i, row in enumerate(rows, 1):
        image_path = resolve_image(manifest, str(row["image_path"]))
        img = Image.open(image_path).convert("RGB")
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        prompt = (
            str(row["prompt"]).strip()
            + " Coordinates are integers in [0,1000] relative to image width and height."
            + " Reply with only <box>x1 y1 x2 y2</box>."
        )
        target = parse_box(str(row["target"]), img.width, img.height)
        t0 = time.time()
        try:
            raw = ask_ollama(host, args.model, prompt, b64, args.num_ctx, args.timeout)
            err = ""
        except Exception as exc:  # noqa: BLE001
            raw, err = "", f"{type(exc).__name__}: {exc}"
        latency = (time.time() - t0) * 1000
        latencies.append(latency)
        pred = parse_box(raw, img.width, img.height) if raw else None
        iou = box_iou(pred, target) if pred and target else 0.0
        out_rows.append(
            {
                "id": f"g{i:04d}",
                "image_path": str(image_path),
                "prompt": prompt,
                "raw": raw,
                "pred": pred,
                "target": target,
                "iou": iou,
                "hit_at_0.5": bool(iou >= 0.5),
                "latency_ms": latency,
                "error": err,
            }
        )
        if i % 10 == 0:
            print(f"[{i}/{len(rows)}] mean_iou={statistics.mean(r['iou'] for r in out_rows):.3f}", flush=True)

    parsed = [r for r in out_rows if r["pred"] is not None]
    agg = {
        "model": args.model,
        "manifest": str(manifest),
        "n": len(out_rows),
        "parsed": len(parsed),
        "parse_rate": len(parsed) / len(out_rows),
        "mean_iou": statistics.mean(r["iou"] for r in out_rows),
        "mean_iou_parsed": statistics.mean(r["iou"] for r in parsed) if parsed else 0.0,
        "hit_at_0.5": sum(r["hit_at_0.5"] for r in out_rows) / len(out_rows),
        "hit_at_0.5_parsed": (
            sum(r["hit_at_0.5"] for r in parsed) / len(parsed) if parsed else 0.0
        ),
        "errors": sum(1 for r in out_rows if r["error"]),
        "p50_latency_ms": statistics.median(latencies),
        "rows": out_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps({k: v for k, v in agg.items() if k != "rows"}, indent=2))
    print(f"rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
