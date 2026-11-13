#!/usr/bin/env python3
"""TinyDoc CLI: receipts, multipage invoices, and the local UI.

Usage:
  tinydoc extract <folder_or_image> [--engine ...] [--out results.jsonl]
  tinydoc invoice <pdf_or_image> [...] [--engine ...] [--out results.jsonl]
  tinydoc ui [--host 127.0.0.1] [--port 7860]

``extract`` is the single-image receipt path. ``invoice`` is the page-wise
multipage path with deterministic merge and optional OCR totals. ``ui`` starts
the local Gradio app; it is not a hosted service.

Writes JSONL (one document per line) and prints a summary table.
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

    from tinydoc.pipeline import ReceiptPipeline

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

    inv = sub.add_parser(
        "invoice", help="Extract a multipage invoice from a PDF or page image"
    )
    inv.add_argument(
        "path",
        nargs="+",
        help="PDF or image paths; one invoice per path (use one PDF for multipage)",
    )
    inv.add_argument("--engine", default="ollama:qwen3-vl:2b-instruct")
    inv.add_argument("--model", default=None, help="Ollama model id override")
    inv.add_argument("--out", default=None, help="Write JSONL here")
    inv.add_argument("--max-px", type=int, default=1400)
    inv.add_argument(
        "--retry-tax-px",
        type=int,
        default=0,
        metavar="PX",
        help=(
            "Re-read the final page at this higher resolution when the printed "
            "tax is structurally implausible (0 disables; try 2400)"
        ),
    )
    inv.add_argument("--no-ocr-totals", action="store_true")
    inv.add_argument("--no-evidence", action="store_true")
    inv.add_argument("--prompt-version", type=int, choices=(1, 2), default=1)

    ui = sub.add_parser("ui", help="Launch the local Gradio UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--share", action="store_true", help="Expose a public Gradio link")

    args = ap.parse_args(argv)

    if args.cmd == "ui":
        from tinydoc.ui import launch

        launch(host=args.host, port=args.port, share=args.share)
        return 0

    if args.cmd == "invoice":
        return _run_invoice(args)

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


def _run_invoice(args) -> int:
    """Run the multipage invoice path over one document per input path."""
    from tinydoc.invoice import InvoicePipeline

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    pipe = InvoicePipeline(args.engine, **kwargs)
    paths = []
    for raw in args.path:
        entry = Path(raw)
        if not entry.exists():
            print(f"path not found: {entry}", file=sys.stderr)
            return 2
        if entry.is_dir():
            paths.extend(
                sorted(
                    p
                    for p in entry.iterdir()
                    if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
                )
            )
        else:
            paths.append(entry)
    if not paths:
        print("no documents found", file=sys.stderr)
        return 1

    results = []
    for path in paths:
        try:
            result = pipe.extract(
                path,
                max_px=args.max_px,
                retry_tax_px=args.retry_tax_px,
                with_ocr_totals=not args.no_ocr_totals,
                with_evidence=not args.no_evidence,
                prompt_version=args.prompt_version,
            )
            results.append(result)
            print(
                f"{path.name}: pages={result.page_count} "
                f"flags={','.join(result.review_flags) or '-'} "
                f"total={result.fields.get('total_amount')} "
                f"{result.latency_ms / 1000:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"{path.name}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append(None)

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            for result in results:
                if result is not None:
                    handle.write(json.dumps(result.to_dict()) + "\n")
        print(f"wrote {out_path}")
    ok = sum(1 for result in results if result is not None)
    print(f"engine={pipe.engine.name} documents={ok}/{len(paths)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
