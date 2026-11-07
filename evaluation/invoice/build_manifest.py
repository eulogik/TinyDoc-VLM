#!/usr/bin/env python3
"""Build an invoice-eval manifest from the Invoice Extraction Benchmark v1.

Corpus (CC BY 4.0, synthetic): 181 invoices with answer keys, categories from
clean digital PDFs to phone photos, scans, handwritten and multi-page files.
This builder renders page 1 of PDFs to PNG (pypdfium2) and leaves image files
in place. It does not call a model.

The benchmark is synthetic and vendor-made; it is a transfer probe, not an
external real-invoice benchmark. Name it that way in any report.

Usage:
    python evaluation/invoice/build_manifest.py \
        --corpus "/Volumes/KIOXIA 1TB/tinydoc/invoice_benchmark_v1/corpus/v1" \
        --render-dir "/Volumes/KIOXIA 1TB/tinydoc/invoice_benchmark_v1/rendered" \
        --out evaluation/invoice/manifest_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MAX_DIM = 1800


def render_pdf_first_page(src: Path, out: Path, max_dim: int = MAX_DIM) -> dict:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(src))
    pages = len(pdf)
    page = pdf[0]
    width, height = page.get_size()
    scale = max_dim / max(width, height)
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil().convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, format="PNG")
    return {"rendered_pages": 1, "source_pages": pages, "width": image.width, "height": image.height}


def render_pdf_pages(
    src: Path,
    out_dir: Path,
    stem: str,
    max_dim: int = MAX_DIM,
    max_pages: int = 0,
) -> dict:
    """Render every PDF page (or the first `max_pages`) to `<stem>_pN.png`."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(src))
    source_pages = len(pdf)
    count = source_pages if max_pages <= 0 else min(source_pages, max_pages)
    paths: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        page = pdf[index]
        width, height = page.get_size()
        scale = max_dim / max(width, height)
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        out = out_dir / f"{stem}_p{index + 1}.png"
        image.save(out, format="PNG")
        paths.append(str(out))
        bitmap.close()
        page.close()
    pdf.close()
    return {
        "rendered_pages": len(paths),
        "source_pages": source_pages,
        "image_paths": paths,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--max-dim", type=int, default=MAX_DIM)
    ap.add_argument(
        "--all-pages",
        action="store_true",
        help="render every PDF page (for a real multipage run), not just page 1",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="with --all-pages, stop after N pages (0 = all)",
    )
    args = ap.parse_args()

    corpus = Path(args.corpus)
    meta = json.loads((corpus / "manifest.json").read_text())
    docs = meta["documents"]
    if args.categories:
        wanted = set(args.categories)
        docs = [d for d in docs if d["category"] in wanted]
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        print("no documents selected")
        return 2

    render_dir = Path(args.render_dir)
    rows = []
    for i, doc in enumerate(docs, 1):
        src = (corpus / doc["file"]).resolve()
        if not src.exists():
            print(f"missing {src}")
            continue
        if doc["render"] == "pdf":
            if args.all_pages:
                info = render_pdf_pages(src, render_dir, doc["doc_id"], args.max_dim, args.max_pages)
                image_paths = info.pop("image_paths")
                image_path = image_paths[0]
            else:
                image_path = render_dir / f"{doc['doc_id']}.png"
                info = render_pdf_first_page(src, image_path, args.max_dim)
                image_paths = [str(image_path)]
        else:
            image_path = src
            image_paths = [str(src)]
            info = {"rendered_pages": 1, "source_pages": doc.get("pages", 1)}
        rows.append(
            {
                "id": doc["doc_id"],
                "image_path": str(image_path),
                "image_paths": image_paths,
                "source_path": str(src),
                "category": doc["category"],
                "difficulty": doc["difficulty"],
                "kind": doc["kind"],
                "currency": doc.get("currency"),
                "pages": doc.get("pages", 1),
                "render": doc["render"],
                "expected": doc["expected"],
                **info,
            }
        )
        if i % 20 == 0:
            print(f"[{i}/{len(docs)}] {doc['doc_id']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    categories = Counter(r["category"] for r in rows)
    print(
        json.dumps(
            {
                "corpus": str(corpus),
                "out": str(out),
                "n": len(rows),
                "categories": dict(sorted(categories.items())),
                "rendered_pdfs": sum(1 for r in rows if r["render"] == "pdf"),
                "rendered_page_images": sum(len(r["image_paths"]) for r in rows),
                "all_pages": args.all_pages,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
