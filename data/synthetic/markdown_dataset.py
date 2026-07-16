"""
Markdown-conversion synthetic dataset for TinyDoc-VLM (Tier-2 retrain, item D).

Generates structured documents (reusing the existing ContentGenerator + PIL
renderer), then emits prompt->target training pairs in the SAME format the
model now uses after heads were removed:

  - "Convert the document to markdown:"  -> markdown transcription (majority)
  - "Extract all text:"                  -> plain-text transcription
  - "Answer the question: ..."           -> VQA answer
  - "Extract the document as JSON:"      -> JSON (invoice / table / contract)

The markdown target is generated from the SAME metadata used to render the
image, giving perfect ground truth (olmOCR / MinerU-style pipeline).

Usage:
    python data/synthetic/markdown_dataset.py --num-docs 1000 --output-dir data/training/synthetic
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

sys_path = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(sys_path))

from data.synthetic.generator import ContentGenerator, DOCUMENT_TYPES
from data.synthetic.pil_renderer import render_document, augment_image as pil_augment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown / text transcription of structured metadata
# ---------------------------------------------------------------------------

def _money(v):
    return v if isinstance(v, str) else f"${v:.2f}"


def metadata_to_markdown(doc_type: str, m: Dict) -> str:
    """Build a markdown transcription of a generated document's metadata."""
    if doc_type == "invoice":
        lines = [f"# Invoice", "",
                 f"**Vendor:** {m.get('vendor_name','')}  ",
                 f"**Invoice #:** {m.get('invoice_number','')}  ",
                 f"**Date:** {m.get('invoice_date','')}  ",
                 f"**Due:** {m.get('due_date','')}", "",
                 f"Bill To: {m.get('customer_name','')}",
                 f"Address: {m.get('customer_address','')}", ""]
        lines.append("| Description | Qty | Unit Price | Amount |")
        lines.append("| --- | --- | --- | --- |")
        for it in m.get("items", []):
            lines.append(f"| {it.get('description','')} | {it.get('quantity','')} | {it.get('unit_price','')} | {it.get('amount','')} |")
        lines += ["", f"**Subtotal:** {m.get('subtotal','')}",
                  f"**Tax ({m.get('tax_rate','')}%):** {m.get('tax_amount','')}",
                  f"**Total:** {m.get('total','')}"]
        return "\n".join(lines)

    if doc_type == "receipt":
        lines = [f"# Receipt — {m.get('store_name','')}", "",
                 f"Date: {m.get('txn_date','')}  Time: {m.get('txn_time','')}",
                 f"Txn #: {m.get('txn_id','')}  Cashier: {m.get('cashier_name','')}", ""]
        lines.append("| Item | Qty | Amount |")
        lines.append("| --- | --- | --- |")
        for it in m.get("items", []):
            lines.append(f"| {it.get('name','')} | {it.get('quantity','')} | {it.get('amount','')} |")
        lines += ["", f"Subtotal: {m.get('subtotal','')}",
                  f"Tax: {m.get('tax_amount','')}",
                  f"Total: {m.get('total','')}",
                  m.get('footer_message','')]
        return "\n".join(lines)

    if doc_type == "form":
        lines = [f"# {m.get('form_title','Form')}", "", f"Form ID: {m.get('form_id','')}", "",
                 m.get('instructions',''), ""]
        for f in m.get("fields", []):
            req = " *" if f.get("required") else ""
            lines.append(f"- **{f.get('label','')}{req}:** {f.get('value','')}")
        return "\n".join(lines)

    if doc_type == "table":
        lines = [f"# {m.get('table_title','Table')}", "",
                 "| " + " | ".join(m.get("headers", [])) + " |",
                 "| " + " | ".join(["---"] * len(m.get("headers", []))) + " |"]
        for row in m.get("rows", []):
            cells = [str(row.get(h, "")) for h in m.get("headers", [])]
            lines.append("| " + " | ".join(cells) + " |")
        if m.get("footnote"):
            lines.append("")
            lines.append(m["footnote"])
        return "\n".join(lines)

    if doc_type == "id_card":
        lines = [f"# {m.get('card_type','')} Card", "",
                 f"**Name:** {m.get('first_name','')} {m.get('last_name','')}",
                 f"**DOB:** {m.get('date_of_birth','')}",
                 f"**ID #:** {m.get('id_number','')}",
                 f"**Department:** {m.get('department','')}",
                 f"**Expiry:** {m.get('expiry_date','')}"]
        return "\n".join(lines)

    if doc_type == "chart":
        lines = [f"# {m.get('chart_title','Chart')}", "",
                 "| Label | Value |", "| --- | --- |"]
        for d in m.get("data", []):
            lines.append(f"| {d.get('label','')} | {d.get('value','')} |")
        if m.get("footnote"):
            lines += ["", m["footnote"]]
        return "\n".join(lines)

    if doc_type == "contract":
        lines = [f"# {m.get('contract_title','AGREEMENT')}", "",
                 f"**Party A:** {m.get('party_a','')}",
                 f"**Party B:** {m.get('party_b','')}",
                 f"**Effective:** {m.get('effective_date','')}", ""]
        for s in m.get("sections", []):
            lines.append(f"## {s.get('title','')}")
            for c in s.get("clauses", []):
                lines.append(c)
        return "\n".join(lines)

    if doc_type == "letter":
        lines = [f"# {m.get('company_name','')}", "",
                 f"Date: {m.get('letter_date','')}",
                 f"To: {m.get('recipient_name','')}",
                 f"Re: {m.get('subject','')}", ""]
        for p in m.get("body_paragraphs", []):
            lines.append(p)
        lines += ["", m.get('closing_sentence',''),
                  "Sincerely,", m.get('sender_name',''), m.get('sender_title','')]
        return "\n".join(lines)

    if doc_type == "medical":
        lines = [f"# {m.get('facility_name','')}", "",
                 f"Patient: {m.get('patient_name','')}  DOB: {m.get('patient_dob','')}  MRN: {m.get('mrn','')}",
                 f"Visit: {m.get('visit_date','')}  Provider: {m.get('provider_name','')}", ""]
        v = m.get("vitals", {})
        lines.append(f"Vitals: BP {v.get('bp','')} | HR {v.get('hr','')} | RR {v.get('rr','')} | Temp {v.get('temp','')} | SpO2 {v.get('spo2','')}")
        lines.append(f"Diagnosis: {m.get('diagnosis',{}).get('primary','')}")
        lines.append("Medications:")
        for med in m.get("medications", []):
            lines.append(f"- {med.get('name','')} {med.get('dosage','')} {med.get('frequency','')}")
        lines.append(f"Notes: {m.get('notes','')}")
        return "\n".join(lines)

    if doc_type == "mixed":
        lines = [f"# {m.get('report_title','Report')}", f"_{m.get('report_subtitle','')}_", "",
                 f"Date: {m.get('report_date','')}", "",
                 m.get('summary',''), "",
                 "## Metrics"]
        for met in m.get("metrics", []):
            lines.append(f"- **{met.get('label','')}:** {met.get('value','')}")
        for tname in ("table_a", "table_b"):
            t = m.get(tname)
            if not t:
                continue
            lines.append(f"## {tname}")
            lines.append("| " + " | ".join(t.get("headers", [])) + " |")
            lines.append("| " + " | ".join(["---"] * len(t.get("headers", []))) + " |")
            for row in t.get("rows", []):
                lines.append("| " + " | ".join(str(x) for x in row) + " |")
        if m.get("footer"):
            lines += ["", m["footer"]]
        return "\n".join(lines)

    # fallback
    return json.dumps(m, indent=2)


def metadata_to_text(doc_type: str, m: Dict) -> str:
    """Plain-text transcription (single-spaced, label: value)."""
    md = metadata_to_markdown(doc_type, m)
    # crude markdown -> text
    text = md.replace("#", "").replace("*", "").replace("|", " ")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def build_samples(doc_type: str, m: Dict) -> List[Dict]:
    """Return a list of {prompt, target} training pairs for one document."""
    md = metadata_to_markdown(doc_type, m)
    txt = metadata_to_text(doc_type, m)
    qa = ContentGenerator.generate_qa(m, doc_type)

    samples = []
    # majority: markdown conversion
    samples.append({"prompt": "Convert the document to markdown:", "target": md})
    # plain text extraction
    samples.append({"prompt": "Extract all text:", "target": txt})

    # VQA
    for qa_pair in qa[:3]:
        q = qa_pair.get("question", "")
        a = qa_pair.get("answer", "")
        if q and a:
            samples.append({"prompt": f"Answer the question: {q}", "target": a})

    # JSON extraction for structured docs
    if doc_type in ("invoice", "table", "contract"):
        if doc_type == "invoice":
            js = json.dumps({"items": m.get("items", []), "total": m.get("total", "")}, ensure_ascii=False)
        elif doc_type == "table":
            js = json.dumps({"rows": m.get("rows", []), "headers": m.get("headers", [])}, ensure_ascii=False)
        else:
            js = json.dumps({"title": m.get("contract_title", ""), "parties": [m.get("party_a", ""), m.get("party_b", "")]}, ensure_ascii=False)
        samples.append({"prompt": "Extract the document as JSON:", "target": js})

    return samples


def generate_markdown_documents(
    num_docs: int,
    output_dir: Path,
    doc_types: Optional[List[str]] = None,
    augment: bool = True,
    seed: int = 42,
) -> List[Dict]:
    random.seed(seed)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc_types = doc_types or DOCUMENT_TYPES
    manifest = []

    for i in range(num_docs):
        doc_type = random.choice(doc_types)
        content = ContentGenerator.generate(doc_type)

        try:
            img = render_document(doc_type, content)
        except Exception as e:
            logger.warning(f"render failed for {doc_type}: {e}")
            continue
        if augment:
            img = pil_augment(img)

        image_filename = f"{doc_type}_{i:06d}.png"
        image_path = images_dir / image_filename
        img.save(str(image_path))

        for s in build_samples(doc_type, content):
            manifest.append({
                "image_path": str(image_path.resolve()),
                "prompt": s["prompt"],
                "target": s["target"],
                "doc_type": doc_type,
                "source": "synthetic",
            })

        if (i + 1) % 500 == 0:
            logger.info(f"Generated {i + 1}/{num_docs} markdown docs")

    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for e in manifest:
            f.write(json.dumps(e) + "\n")

    logger.info(f"Wrote {len(manifest)} training pairs to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate markdown-conversion synthetic training data")
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/training/synthetic")
    parser.add_argument("--doc-types", type=str, nargs="+", default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_markdown_documents(
        num_docs=args.num_docs,
        output_dir=Path(args.output_dir),
        doc_types=args.doc_types,
        augment=not args.no_augment,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
