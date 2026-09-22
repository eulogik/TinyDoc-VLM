#!/usr/bin/env python3
"""TinyDoc Phase-1 CLI: folder → schema-valid JSON + evidence + confidence.

Usage:
  python -m tinydoc.cli extract <folder_or_image> [--engine auto|ollama|ocr_regex|smolvlm2]
      [--out results.jsonl] [--limit N] [--no-evidence]

Writes JSONL (one doc per line) and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    # allow running from repo without install
    sdk_root = Path(__file__).absolute().parent.parent
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))

    from tinydoc.pipeline import ReceiptPipeline  # noqa: E402

    ap = argparse.ArgumentParser(prog="tinydoc", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="Extract fields from an image or folder")
    ex.add_argument("path", help="Image file or directory")
    ex.add_argument("--engine", default="auto")
    ex.add_argument("--out", default=None, help="Write JSONL here")
    ex.add_argument("--limit", type=int, default=None)
    ex.add_argument("--no-evidence", action="store_true")
    ex.add_argument("--model", default=None, help="Engine-specific model id")
    ex.add_argument(
        "--overlay",
        default=None,
        metavar="DIR",
        help="Write evidence-overlay PNGs into DIR (red/blue boxes + labels)",
    )

    args = ap.parse_args(argv)

    p = Path(args.path)
    if not p.exists():
        print(f"path not found: {p}", file=sys.stderr)
        return 2
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    pipe = ReceiptPipeline(args.engine, **kwargs)
    with_ev = not args.no_evidence

    if p.is_dir():
        results = pipe.extract_folder(p, limit=args.limit, with_evidence=with_ev)
    else:
        results = [pipe.extract(p, with_evidence=with_ev)]

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")

    if args.overlay:
        from tinydoc.overlay import draw_overlay

        odir = Path(args.overlay)
        odir.mkdir(parents=True, exist_ok=True)
        for r in results:
            name = Path(r.image_path).stem + "_overlay.png"
            try:
                draw_overlay(r.image_path, r.field_results, odir / name)
            except Exception as e:
                print(f"  overlay failed for {r.image_path}: {e}")
        print(f"wrote overlays → {odir}")

    n = len(results)
    if n == 0:
        print("no images found")
        return 1
    schema_rate = sum(r.schema_valid for r in results) / n
    mean_conf = sum(r.confidence for r in results) / n
    ev_rate = sum(r.evidence_coverage for r in results) / n
    mean_lat = sum(r.latency_ms for r in results) / n
    print(
        f"engine={pipe.engine.name} n={n} "
        f"schema={schema_rate:.3f} conf={mean_conf:.3f} "
        f"evidence={ev_rate:.3f} latency_ms={mean_lat:.0f}"
    )
    for r in results[:5]:
        print(
            f"  {Path(r.image_path).name}: schema={r.schema_valid} "
            f"conf={r.confidence:.2f} fields={r.fields}"
        )
    if out_path:
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
