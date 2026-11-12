#!/usr/bin/env python3
"""Build a manifest for the real Portuguese invoice/receipt probe.

Source data (CC BY 4.0, human transcriptions):
- Zenodo 6371710: 813 photos of invoices/receipts from a private Portuguese
  company. Fields: company, date, address, total, invoice_number, NIFs, IVA.
- Zenodo 7213544: 190 photos of one author's personal invoices/receipts, same
  fields.

This is a *field-subset* real-world probe, not a full invoice benchmark. The
transcriptions do not contain due dates, line items, or a document type, and
the IVA field is occasionally impossible (for example larger than the total);
those values are left out of `expected.scored` rather than corrected. Duplicate
photos of the same invoice are dropped by business key before sampling.

Usage:
    python evaluation/invoice/build_pt_manifest.py \
        --root "$TINYDOC_PT_ROOT" \
        --out evaluation/invoice/manifest_pt_real.jsonl \
        --probe evaluation/invoice/manifest_pt_real_probe100.jsonl \
        --stats evaluation/invoice/results/pt_real_dataset_stats.json

The manifest stores absolute `image_path` values for the current machine plus a
portable `image_rel` path. Images are not committed; re-run the script after
re-downloading the Zenodo archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

SOURCES = {
    "company": {
        "record": "6371710",
        "doi": "10.5281/zenodo.6371710",
        "title": "Dataset of invoices and receipts including annotation of relevant fields",
        "annotation_dir": "ann",
        "image_dir": "images_6371710",
        "category": "real_pt_company",
        "expected_docs": 813,
    },
    "personal": {
        "record": "7213544",
        "doi": "10.5281/zenodo.7213544",
        "title": "Dataset of personal invoices and receipts including annotation of relevant fields",
        "annotation_dir": "ann2",
        "image_dir": "images_7213544",
        "category": "real_pt_personal",
        "expected_docs": 190,
    },
}

# Official Zenodo checksums (MD5) and sizes. Verified before extraction.
ARCHIVES = {
    ("company", "images"): ("5109286806b96db32da7d6206051ee61", 410_222_137),
    ("company", "annotations"): ("941deec66609f158cfd7ce5add75a17b", 290_214),
    ("personal", "images"): ("1b7c2c1ebb3e88f5294c5ec785135069", 114_472_170),
    ("personal", "annotations"): ("914fd4127ccc16233e6503e786d8574b", 69_448),
}

CENT = 0.005
MAX_TAX_RATIO = 0.35


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_key(text: str) -> str:
    """Case/punctuation-insensitive key used only for duplicate detection."""
    text = _ascii(text or "").casefold().replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def parse_amount(value: Any) -> float | None:
    """Parse Portuguese (1.234,56) and dot-decimal (1234.56) amounts."""
    if value is None:
        return None
    text = str(value).strip().replace("\u00a0", " ").replace(" ", "")
    text = text.replace("€", "").replace("EUR", "").replace("$", "")
    if not text:
        return None
    negative = text.startswith("-")
    text = text.lstrip("+-")
    # The transcriptions use both 13,00 and 6.950 for six euros ninety-five.
    # A single separator is decimal; with both, the last one is decimal.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def parse_date(value: Any) -> str | None:
    """Normalize Portuguese dates to ISO, day-first unless the year is first."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            return None
    parts = re.findall(r"\d+", text)
    if len(parts) < 3:
        return None
    a, b, c = parts[0], parts[1], parts[2]
    if len(a) == 4:
        year, month, day = int(a), int(b), int(c)
    else:
        day, month, year = int(a), int(b), int(c)
        if year < 100:
            year += 2000
        if day > 31 or month > 12:
            day, month = month, day
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def load_annotations(path: Path) -> list[dict[str, Any]]:
    rows = []
    for file in sorted(path.rglob("*.txt")):
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip unreadable annotation {file}: {exc}", file=sys.stderr)
            continue
        raw["_id"] = file.stem
        rows.append(raw)
    return rows


def download_archives(root: Path) -> None:
    """Download and verify the four Zenodo archives if they are missing."""
    root.mkdir(parents=True, exist_ok=True)
    for source, spec in SOURCES.items():
        for kind, filename in (
            ("images", "1_Images.zip"),
            ("annotations", "2_Annotations_Json.zip"),
        ):
            expected_md5, expected_size = ARCHIVES[(source, kind)]
            destination = root / f"{spec['record']}_{filename}"
            if destination.exists() and destination.stat().st_size == expected_size:
                digest = hashlib.md5(destination.read_bytes()).hexdigest()
                if digest == expected_md5:
                    continue
            url = f"https://zenodo.org/api/records/{spec['record']}/files/{filename}/content"
            print(f"download {url}", flush=True)
            urllib.request.urlretrieve(url, destination)
            digest = hashlib.md5(destination.read_bytes()).hexdigest()
            if digest != expected_md5 or destination.stat().st_size != expected_size:
                raise SystemExit(
                    f"checksum mismatch for {destination}: {digest} != {expected_md5}"
                )


def extract_archives(root: Path) -> None:
    for spec in SOURCES.values():
        images = root / f"{spec['record']}_1_Images.zip"
        annotations = root / f"{spec['record']}_2_Annotations_Json.zip"
        for archive, target in (
            (images, root / spec["image_dir"]),
            (annotations, root / spec["annotation_dir"]),
        ):
            if not archive.exists():
                raise SystemExit(f"missing {archive}; run with --download")
            if target.exists() and any(target.rglob("*")):
                continue
            target.mkdir(parents=True, exist_ok=True)
            print(f"extract {archive.name} -> {target}", flush=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)


def build_row(
    raw: dict[str, Any],
    image: Path,
    source: str,
    root: Path,
) -> dict[str, Any]:
    spec = SOURCES[source]
    vendor = str(raw.get("company") or "").strip()
    number = str(raw.get("invoice_number") or "").strip()
    invoice_date = parse_date(raw.get("date"))
    total = parse_amount(raw.get("total"))
    tax = parse_amount(raw.get("iva_amount"))
    issues: list[str] = []
    if not vendor:
        issues.append("missing_vendor")
    if not number:
        issues.append("missing_invoice_number")
    if not invoice_date:
        issues.append("unparseable_date")
    if total is None:
        issues.append("unparseable_total")
    elif total < 0:
        issues.append("negative_total")
        total = None

    if tax is not None and (
        total is None or tax < 0 or tax >= total or (total > 0 and tax / total > MAX_TAX_RATIO)
    ):
        issues.append("implausible_tax")
        tax = None

    scored: list[str] = []
    if vendor:
        scored.append("vendor_name")
    if number:
        scored.append("invoice_number")
    if invoice_date:
        scored.append("invoice_date")
    scored.append("currency")
    if tax is not None:
        scored.append("tax")
    if total is not None:
        scored.append("total")

    expected: dict[str, Any] = {
        "vendor_name": vendor,
        "invoice_number": number,
        "invoice_date": invoice_date,
        "currency": "EUR",
        "document_type": "invoice",
        "scored": scored,
        "lines": [],
        "should_flag": False,
        "references": [],
    }
    if tax is not None:
        expected["tax"] = tax
    if total is not None:
        expected["total"] = total

    try:
        from PIL import Image

        with Image.open(image) as handle:
            width, height = handle.size
    except Exception:  # noqa: BLE001 - dimensions are descriptive, not required
        width = height = None

    return {
        "id": f"ptreal-{source}-{raw['_id']}",
        "image_path": str(image),
        "image_rel": str(image.relative_to(root)),
        "source_path": f"https://doi.org/{spec['doi']}",
        "source": f"zenodo:{spec['record']}",
        "category": spec["category"],
        "difficulty": "real",
        "kind": "invoice",
        "currency": "EUR",
        "pages": 1,
        "render": "photo",
        "width": width,
        "height": height,
        "license": "CC-BY-4.0",
        "expected": expected,
        # Only the fields used for scoring. The upstream files also carry
        # addresses and buyer/seller tax IDs; they are deliberately not copied
        # into this public manifest.
        "annotation_raw": {
            key: raw.get(key)
            for key in ("company", "date", "total", "invoice_number", "iva_amount")
        },
        "annotation_issues": issues,
        "_bytes": image.stat().st_size,
    }


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop repeat photos of the same invoice; keep the richest copy."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -len(row["expected"]["scored"]),
            -int(row.get("_bytes") or 0),
            row["id"],
        ),
    )
    seen: dict[tuple[Any, ...], str] = {}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in ordered:
        expected = row["expected"]
        key = None
        if (
            expected.get("vendor_name")
            and expected.get("invoice_number")
            and expected.get("invoice_date")
            and expected.get("total") is not None
        ):
            key = (
                normalize_key(expected["vendor_name"]),
                normalize_key(expected["invoice_number"]),
                expected["invoice_date"],
                round(float(expected["total"]), 2),
            )
        if key is not None and key in seen:
            dropped.append(
                {
                    "id": row["id"],
                    "duplicate_of": seen[key],
                    "business_key": list(key),
                }
            )
            continue
        if key is not None:
            seen[key] = row["id"]
        kept.append(row)
    return sorted(kept, key=lambda row: row["id"]), sorted(dropped, key=lambda row: row["id"])


def _range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def pct(q: float) -> float:
        index = min(max(round(q * (len(ordered) - 1)), 0), len(ordered) - 1)
        return ordered[index]

    return {
        "min": ordered[0],
        "p25": pct(0.25),
        "median": pct(0.5),
        "p75": pct(0.75),
        "max": ordered[-1],
    }


def summarize(rows: list[dict[str, Any]], dropped: list[dict[str, Any]], raw_count: int) -> dict[str, Any]:
    fields = Counter()
    for row in rows:
        fields.update(row["expected"]["scored"])
    totals = [float(row["expected"]["total"]) for row in rows if row["expected"].get("total") is not None]
    ratios = [
        float(row["expected"]["tax"]) / float(row["expected"]["total"])
        for row in rows
        if row["expected"].get("tax") is not None and row["expected"].get("total")
    ]
    sources = Counter(row["category"] for row in rows)
    issues = Counter(issue for row in rows for issue in row["annotation_issues"])
    dates = [row["expected"]["invoice_date"] for row in rows if row["expected"].get("invoice_date")]
    return {
        "raw_documents": raw_count,
        "documents": len(rows),
        "duplicates_dropped": len(dropped),
        "sources": dict(sorted(sources.items())),
        "field_scored": dict(sorted(fields.items())),
        "totals_eur": _range(totals),
        "tax_ratio": _range(ratios),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "annotation_issues": dict(sorted(issues.items())),
        "duplicates": dropped,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/datasets/pt_invoices"),
        help="dataset root containing the extracted archives",
    )
    ap.add_argument("--out", type=Path, default=Path("evaluation/invoice/manifest_pt_real.jsonl"))
    ap.add_argument(
        "--probe",
        type=Path,
        default=Path("evaluation/invoice/manifest_pt_real_probe100.jsonl"),
    )
    ap.add_argument("--probe-size", type=int, default=100)
    ap.add_argument("--probe-seed", type=int, default=42)
    ap.add_argument(
        "--stats",
        type=Path,
        default=Path("evaluation/invoice/results/pt_real_dataset_stats.json"),
    )
    ap.add_argument("--download", action="store_true", help="download/verify/extract the Zenodo archives")
    ap.add_argument("--no-probe", action="store_true", help="skip the fixed probe sample")
    args = ap.parse_args()

    root: Path = args.root
    if args.download:
        download_archives(root)
        extract_archives(root)

    rows: list[dict[str, Any]] = []
    for source, spec in SOURCES.items():
        annotation_dir = root / spec["annotation_dir"]
        image_dir = root / spec["image_dir"]
        if not annotation_dir.exists() or not image_dir.exists():
            print(f"skip {source}: missing {annotation_dir} or {image_dir}", file=sys.stderr)
            continue
        image_index: dict[str, Path] = {}
        for path in image_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_index[path.stem] = path
        annotations = load_annotations(annotation_dir)
        print(f"{source}: {len(annotations)} annotations, {len(image_index)} images")
        for raw in annotations:
            image = image_index.get(raw["_id"])
            if image is None:
                print(f"skip {source}/{raw['_id']}: image missing", file=sys.stderr)
                continue
            rows.append(build_row(raw, image, source, root))

    kept, dropped = deduplicate(rows)
    write_jsonl(args.out, kept)
    stats = summarize(kept, dropped, len(rows))
    stats["probe"] = None
    print(json.dumps({k: v for k, v in stats.items() if k != "duplicates"}, indent=2))

    if not args.no_probe and args.probe_size > 0:
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in kept:
            by_source[row["category"]].append(row)
        total = sum(len(items) for items in by_source.values())
        if total == 0:
            raise SystemExit("no rows to sample")
        size = min(args.probe_size, total)
        company_share = len(by_source.get("real_pt_company", [])) / total
        n_company = min(len(by_source.get("real_pt_company", [])), round(size * company_share))
        n_personal = min(len(by_source.get("real_pt_personal", [])), size - n_company)
        # Rebalance if one source is too small.
        n_company = min(len(by_source.get("real_pt_company", [])), size - n_personal)
        rng = random.Random(args.probe_seed)
        sample: list[dict[str, Any]] = []
        for category, n in (
            ("real_pt_company", n_company),
            ("real_pt_personal", n_personal),
        ):
            pool = sorted(by_source.get(category, []), key=lambda row: row["id"])
            sample.extend(rng.sample(pool, n))
        sample.sort(key=lambda row: row["id"])
        write_jsonl(args.probe, sample)
        stats["probe"] = {
            "path": str(args.probe),
            "size": len(sample),
            "seed": args.probe_seed,
            "sources": dict(sorted(Counter(row["category"] for row in sample).items())),
            "field_scored": dict(
                sorted(Counter(field for row in sample for field in row["expected"]["scored"]).items())
            ),
        }
        print(json.dumps(stats["probe"], indent=2))

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(kept)} docs), {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
