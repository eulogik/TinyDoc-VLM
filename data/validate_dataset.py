#!/usr/bin/env python3
"""
Gold-class validation for the TinyDoc-VLM training manifest.

Checks, per training pair and per document:

  1. Integrity   - image exists & non-trivial, prompt/target non-empty,
                   no placeholder/leak strings (en_US, Lorem, None, {{)
  2. Consistency - QA answers appear in the doc's markdown/text target;
                   invoice/receipt money math is exact; JSON targets parse
  3. Structure   - markdown targets start with a heading; "Extract the
                   document as JSON:" targets are valid JSON
  4. Stats       - coverage by source / doc_type / prompt type, token stats

Usage:
    python data/validate_dataset.py data/training/manifest.jsonl [--sample 2000]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

LEAK_PATTERNS = [
    r"en_us", r"lorem", r"\{\{", r"\}\}", r"\bplaceholder\b",
    r"\bTBD\b", r"\bxxx+\b",
]

MONEY_RE = re.compile(r"^\$\d+(?:\.\d{2})?$")


def money(v):
    return float(v.strip("$").replace(",", ""))


def check_leaks(text: str) -> list:
    return [p for p in LEAK_PATTERNS if re.search(p, text, re.I)]


def validate_pairs(entries, images_root, sample_limit=None):
    errors, warnings = [], []
    n = 0
    by_source = Counter()
    by_prompt = Counter()
    for e in entries:
        if sample_limit and n >= sample_limit:
            break
        n += 1
        src = e.get("source", "synthetic")
        by_source[src] += 1
        by_prompt[e.get("prompt", "")[:40]] += 1

        img = Path(e.get("image_path", ""))
        if img.is_absolute():
            img_abs = img
        elif img.parts and img.parts[0] == "data":
            img_abs = img  # repo-relative (data/training/...)
        else:
            img_abs = Path(images_root) / img
        if not img_abs.exists():
            errors.append(f"{img}: image missing")
            continue
        if img_abs.stat().st_size < 1000:
            warnings.append(f"{img}: suspiciously small image ({img_abs.stat().st_size} B)")

        prompt = e.get("prompt", "")
        target = e.get("target", "")
        if not prompt or not target:
            errors.append(f"{img}: empty prompt or target")
            continue
        if len(target) > 4096:
            warnings.append(f"{img}: very long target ({len(target)} chars)")

        # Leak patterns guard the synthetic generator (en_US, placeholders);
        # real-world docs legitimately contain TBD/XXX/Lorem/placeholder.
        if e.get("source", "synthetic") in ("synthetic", ""):
            for pat in check_leaks(target):
                errors.append(f"{img}: placeholder leak pattern {pat!r}")

        if "\x00" in target or "\ufffd" in target:
            errors.append(f"{img}: control/replacement chars in target")

        low_t = target.strip().lower()
        if e.get("prompt", "").startswith("Convert the document to markdown"):
            if not low_t.startswith("#"):
                errors.append(f"{img}: markdown target does not start with a heading")
        elif e.get("prompt", "").startswith("Extract the document as JSON"):
            try:
                json.loads(target)
            except Exception:
                errors.append(f"{img}: JSON target does not parse")

    return errors, warnings, by_source, by_prompt, n


def validate_doc_math(doc_meta):
    """Replay money math for structured docs; returns list of errors."""
    errors = []
    try:
        if doc_meta.get("subtotal"):
            sub = money(doc_meta["subtotal"])
            tax = money(doc_meta["tax_amount"])
            tot = money(doc_meta["total"])
            if abs(sub + tax - tot) > 0.011:
                errors.append(f"invoice/receipt math: {sub}+{tax} != {tot}")
        if doc_meta.get("discount_amount"):
            sub = money(doc_meta["subtotal"])
            tax = money(doc_meta["tax_amount"])
            disc = money(doc_meta["discount_amount"])
            tot = money(doc_meta["total"])
            if abs(sub + tax - disc - tot) > 0.011:
                errors.append(f"receipt discount math: {sub}+{tax}-{disc} != {tot}")
        if doc_meta.get("amount_tendered"):
            ten = money(doc_meta["amount_tendered"])
            chg = money(doc_meta["change_due"])
            tot = money(doc_meta["total"])
            if abs(ten - tot - chg) > 0.011:
                errors.append(f"receipt tendered math: {ten}-{tot} != {chg}")
    except (TypeError, ValueError) as ex:
        errors.append(f"unparseable money value: {ex}")
    return errors


def validate_qa_consistency(entries):
    """For each doc, QA answers must be reconstructible from the doc's
    markdown/text targets (exact substring, per-part for composite answers,
    per-value for JSON answers)."""
    errors = []
    docs = {}
    for e in entries:
        docs.setdefault(e.get("image_path"), []).append(e)

    def _flatten(v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            out = []
            for x in v:
                out += _flatten(x)
            return out
        if isinstance(v, dict):
            out = []
            for x in v.values():
                out += _flatten(x)
            return out
        return [str(v)]

    for img, pairs in docs.items():
        md_text = ""
        qa_answers = []
        for p in pairs:
            prompt = p.get("prompt", "")
            target = p.get("target", "")
            if prompt.startswith("Convert the document to markdown") or prompt.startswith("Extract all text"):
                md_text += target.lower() + "\n"
            elif prompt.startswith("Answer the question"):
                qa_answers.append((target, p))

        def _present(norm_val: str) -> bool:
            return bool(norm_val) and norm_val in md_text

        for ans, p in qa_answers:
            norm = ans.lower().strip()
            if not norm or norm in ("none", "n/a"):
                errors.append(f"{img}: empty QA answer")
                continue
            stripped = ans.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    vals = [v for v in _flatten(json.loads(stripped)) if isinstance(v, str) and v.strip()]
                except Exception:
                    errors.append(f"{img}: QA answer is invalid JSON")
                    continue
                missing = [v for v in vals if not _present(v.lower().strip())]
                if missing and len(missing) > len(vals) * 0.3:
                    errors.append(f"{img}: JSON QA answer values missing from doc text "
                                  f"({len(missing)}/{len(vals)}): {missing[0][:50]!r} ...")
            elif not _present(norm):
                parts = [p_.strip().lower() for p_ in re.split(r",\s*|\s+and\s+", stripped) if len(p_.strip()) > 2]
                missing = [p_ for p_ in parts if not _present(p_)]
                if missing and len(missing) > len(parts) * 0.3:
                    errors.append(f"{img}: composite QA answer parts missing from doc text "
                                  f"({len(missing)}/{len(parts)}): {missing[0][:50]!r} ...")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=str)
    ap.add_argument("--images-root", type=str, default="data/training")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--qa-consistency", action="store_true",
                    help="check QA answers against doc targets (slower, full scan)")
    args = ap.parse_args()

    entries = [json.loads(l) for l in Path(args.manifest).open() if l.strip()]
    print(f"manifest: {args.manifest} | {len(entries)} pairs")

    errors, warnings, by_source, by_prompt, n = validate_pairs(
        entries, args.images_root, sample_limit=args.sample)

    if args.qa_consistency:
        print("QA-consistency scan (full manifest)...")
        errors += validate_qa_consistency(entries)

    print(f"\nchecked pairs: {n}")
    print(f"by_source: {dict(by_source)}")
    print(f"top prompts: {by_prompt.most_common(6)}")
    print(f"\nERRORS ({len(errors)}):")
    for e in errors[:50]:
        print("  ERR", e)
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more")
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings[:20]:
        print("  WRN", w)
    print(f"\nRESULT: {'FAIL' if errors else 'PASS'} ({len(errors)} errors)")

    # Doc-level math replay (sample of docs with metadata)
    meta_errors = []
    seen = set()
    for e in entries:
        key = e.get("image_path")
        if key in seen or "metadata" not in e:
            continue
        seen.add(key)
        meta_errors += validate_doc_math(e["metadata"])
        if len(seen) >= 500:
            break
    print(f"\ndoc-math replay (500 docs): {len(meta_errors)} errors")
    for me in meta_errors[:20]:
        print("  ERR", me)

    sys.exit(1 if (errors or meta_errors) else 0)


if __name__ == "__main__":
    main()
