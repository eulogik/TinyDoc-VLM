"""
Synthetic Document Generation Pipeline for TinyDoc-VLM.

Generates realistic document images with perfect ground truth structured data.
Pipeline: Template (HTML/CSS) → Content (Faker/LLM) → Render (WeasyPrint → PDF → Image) → Augment → Save

Usage:
    python data/synthetic/generator.py --num-docs 1000 --output-dir data/synthetic/output
"""

import argparse
import json
import logging
import os
import random
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
from faker import Faker
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.synthetic.pil_renderer import render_document, augment_image as pil_augment

logger = logging.getLogger(__name__)

fake = Faker()
fake.add_provider("en_US")

DOCUMENT_TYPES = ["invoice", "receipt", "form", "table", "id_card", "chart", "contract", "letter", "medical", "mixed"]

TEMPLATE_DIR = Path(__file__).parent / "templates"


class ContentGenerator:
    """Generates realistic document content using Faker."""

    @staticmethod
    def invoice() -> Dict:
        num_items = random.randint(2, 8)
        items = []
        subtotal = 0.0
        for _ in range(num_items):
            qty = random.randint(1, 10)
            unit_price = round(random.uniform(5.0, 500.0), 2)
            amount = round(qty * unit_price, 2)
            subtotal += amount
            items.append({
                "description": fake.catch_phrase(),
                "unit_price": f"${unit_price:.2f}",
                "quantity": str(qty),
                "amount": f"${amount:.2f}",
            })
        tax_rate = random.choice([5.0, 8.0, 10.0, 13.0, 18.0, 20.0, 25.0])
        tax_amount = round(subtotal * tax_rate / 100, 2)
        total = round(subtotal + tax_amount, 2)
        return {
            "vendor_name": fake.company(),
            "vendor_address": fake.address().replace("\n", ", "),
            "vendor_phone": fake.phone_number(),
            "vendor_email": fake.company_email(),
            "invoice_number": f"INV-{fake.random_number(digits=6)}",
            "invoice_date": fake.date_between(start_date="-90d", end_date="today").strftime("%B %d, %Y"),
            "due_date": fake.date_between(start_date="today", end_date="+30d").strftime("%B %d, %Y"),
            "customer_name": fake.name(),
            "customer_address": fake.address().replace("\n", ", "),
            "customer_email": fake.email(),
            "bank_name": fake.company(),
            "account_number": str(fake.random_number(digits=10)),
            "swift_code": f"SW{fake.bothify(text='???')}{fake.random_number(digits=3)}",
            "items": items,
            "subtotal": f"${subtotal:.2f}",
            "tax_rate": f"{tax_rate:.1f}",
            "tax_amount": f"${tax_amount:.2f}",
            "total": f"${total:.2f}",
        }

    @staticmethod
    def receipt() -> Dict:
        num_items = random.randint(1, 6)
        items = []
        subtotal = 0.0
        for _ in range(num_items):
            qty = random.randint(1, 5)
            price = round(random.uniform(1.0, 100.0), 2)
            amount = round(qty * price, 2)
            subtotal += amount
            items.append({
                "name": fake.catch_phrase() if random.random() > 0.3 else f"{fake.word()} {fake.word()}",
                "barcode": str(fake.random_number(digits=12)),
                "quantity": str(qty),
                "amount": f"${amount:.2f}",
                "unit_price": f"${price:.2f}",
            })
        tax_rate = random.choice([5.0, 8.0, 10.0, 13.0])
        tax_amount = round(subtotal * tax_rate / 100, 2)
        discount = round(random.uniform(0, 5), 2) if random.random() > 0.7 else 0.0
        total = round(subtotal + tax_amount - discount, 2)
        return {
            "store_name": fake.company(),
            "store_address": fake.address().replace("\n", ", "),
            "store_phone": fake.phone_number(),
            "store_id": str(fake.random_number(digits=4)),
            "register_id": str(fake.random_number(digits=2)),
            "txn_date": fake.date_between(start_date="-30d", end_date="today").strftime("%Y-%m-%d"),
            "txn_time": fake.time(),
            "cashier_name": fake.first_name(),
            "txn_id": str(fake.random_number(digits=8)),
            "items": items,
            "subtotal": f"${subtotal:.2f}",
            "tax_rate": f"{tax_rate:.1f}",
            "tax_amount": f"${tax_amount:.2f}",
            "discount_amount": f"${discount:.2f}",
            "total": f"${total:.2f}",
            "payment_type": random.choice(["VISA", "MASTERCARD", "AMEX", "CASH", "DEBIT"]),
            "amount_tendered": f"${total + random.uniform(0, 50):.2f}",
            "change_due": f"${random.uniform(0, 50):.2f}",
            "footer_message": random.choice([
                "Refunds accepted within 30 days with receipt.",
                "All sales final. No returns after 14 days.",
                "Price matching available. See store for details.",
                "Thank you for your loyalty!",
            ]),
        }

    @staticmethod
    def form() -> Dict:
        fields = []
        for _ in range(random.randint(5, 12)):
            field_type = random.choice(["text", "date", "select", "checkbox"])
            fields.append({
                "label": fake.catch_phrase().split()[0] + " " + fake.word(),
                "type": field_type,
                "value": fake.name() if field_type == "text" else fake.date() if field_type == "date" else ("Yes" if field_type == "checkbox" else fake.word()),
                "required": random.choice([True, False]),
            })
        return {
            "form_title": f"{fake.word().title()} {fake.word().title()} Form",
            "form_id": f"FORM-{fake.random_number(digits=5)}",
            "instructions": "Please fill out all required fields.",
            "fields": fields,
            "has_signature": random.choice([True, False]),
        }

    @staticmethod
    def table() -> Dict:
        headers = random.sample(["Item", "Description", "Quantity", "Unit Price", "Total", "SKU", "Category", "Date"], k=random.randint(3, 5))
        rows = []
        for _ in range(random.randint(3, 8)):
            row = {}
            for h in headers:
                if h == "Item":
                    row[h] = fake.word().title()
                elif h == "Description":
                    row[h] = fake.catch_phrase()
                elif h == "Quantity":
                    row[h] = str(random.randint(1, 100))
                elif h == "Unit Price":
                    row[h] = f"${random.uniform(1, 1000):.2f}"
                elif h in ("Total", "Amount"):
                    row[h] = f"${random.uniform(10, 5000):.2f}"
                elif h == "SKU":
                    row[h] = f"SKU-{fake.random_number(digits=5)}"
                elif h == "Category":
                    row[h] = fake.word().title()
                elif h == "Date":
                    row[h] = fake.date()
            rows.append(row)
        return {
            "table_title": f"{fake.word().title()} {fake.word().title()} Report",
            "headers": headers,
            "rows": rows,
            "footnote": fake.sentence() if random.random() > 0.5 else "",
        }

    @staticmethod
    def id_card() -> Dict:
        gender = random.choice(["M", "F"])
        first_name = fake.first_name_male() if gender == "M" else fake.first_name_female()
        last_name = fake.last_name()
        return {
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%m/%d/%Y"),
            "id_number": str(fake.random_number(digits=9)),
            "address": fake.address().replace("\n", ", "),
            "expiry_date": fake.date_between(start_date="today", end_date="+10y").strftime("%m/%d/%Y"),
            "card_type": random.choice(["EMPLOYEE", "STUDENT", "MEMBERSHIP", "GOVERNMENT"]),
            "department": random.choice(["Engineering", "Marketing", "Finance", "HR", "Operations", "Sales"]),
        }

    @staticmethod
    def id_card() -> Dict:
        gender = random.choice(["M", "F"])
        first_name = fake.first_name_male() if gender == "M" else fake.first_name_female()
        last_name = fake.last_name()
        return {
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%m/%d/%Y"),
            "id_number": str(fake.random_number(digits=9)),
            "card_type": random.choice(["EMPLOYEE", "STUDENT", "MEMBERSHIP", "GOVERNMENT"]),
            "department": random.choice(["Engineering", "Marketing", "Finance", "HR", "Operations", "Sales"]),
            "expiry_date": fake.date_between(start_date="today", end_date="+10y").strftime("%m/%d/%Y"),
        }

    @staticmethod
    def chart() -> Dict:
        num_bars = random.randint(4, 8)
        labels_pool = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                       "Q1", "Q2", "Q3", "Q4", "Product A", "Product B", "Product C", "Sales", "Revenue", "Cost"]
        selected_labels = random.sample(labels_pool, num_bars)
        data = []
        for label in selected_labels:
            data.append({"label": label, "value": random.randint(10, 100)})
        max_value = max(d["value"] for d in data)
        return {
            "chart_title": f"{fake.word().title()} {fake.word().title()} Report",
            "data": data,
            "max_value": max_value,
            "footnote": fake.sentence() if random.random() > 0.5 else "",
        }

    @staticmethod
    def contract() -> Dict:
        num_sections = random.randint(3, 5)
        section_titles = random.sample([
            "Definitions", "Term", "Payment Terms", "Delivery", "Warranty",
            "Confidentiality", "Termination", "Indemnification", "Governing Law",
            "Dispute Resolution", "Limitation of Liability", "Assignment"
        ], num_sections)
        sections = []
        for title in section_titles:
            num_clauses = random.randint(2, 4)
            clauses = []
            for _ in range(num_clauses):
                clauses.append(f"{random.randint(1, 10)}. {fake.sentence()}")
            sections.append({"title": title, "clauses": clauses})
        return {
            "contract_title": f"{fake.word().title()} Agreement".upper(),
            "party_a": fake.company(),
            "party_b": fake.company(),
            "effective_date": fake.date_between(start_date="-30d", end_date="today").strftime("%B %d, %Y"),
            "sections": sections,
            "signature_date": fake.date_between(start_date="today", end_date="+14d").strftime("%B %d, %Y"),
        }

    @staticmethod
    def letter() -> Dict:
        num_paragraphs = random.randint(2, 4)
        paragraphs = []
        for _ in range(num_paragraphs):
            num_sentences = random.randint(2, 5)
            para = " ".join(fake.sentence() for _ in range(num_sentences))
            paragraphs.append(para)
        return {
            "company_name": fake.company(),
            "company_address": fake.address().replace("\n", ", "),
            "company_phone": fake.phone_number(),
            "company_email": fake.company_email(),
            "letter_date": fake.date_between(start_date="-14d", end_date="today").strftime("%B %d, %Y"),
            "recipient_name": fake.name(),
            "recipient_address": fake.address().replace("\n", ", "),
            "subject": fake.catch_phrase(),
            "body_paragraphs": paragraphs,
            "closing_sentence": random.choice([
                "Thank you for your prompt attention to this matter.",
                "We look forward to your response.",
                "Please do not hesitate to contact us if you have any questions.",
                "We appreciate your continued partnership.",
            ]),
            "sender_name": fake.name(),
            "sender_title": fake.job(),
        }

    @staticmethod
    def medical() -> Dict:
        return {
            "facility_name": fake.company(),
            "facility_address": fake.address().replace("\n", ", "),
            "record_type": random.choice(["Progress Note", "Discharge Summary", "Consultation Report", "Lab Results", "Radiology Report"]),
            "patient_name": fake.name(),
            "patient_dob": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%m/%d/%Y"),
            "mrn": f"MRN-{fake.random_number(digits=7)}",
            "visit_date": fake.date_between(start_date="-60d", end_date="today").strftime("%m/%d/%Y"),
            "provider_name": f"Dr. {fake.last_name()}",
            "vitals": {
                "bp": f"{random.randint(100, 160)}/{random.randint(60, 100)}",
                "hr": str(random.randint(60, 100)),
                "rr": str(random.randint(12, 24)),
                "temp": f"{random.uniform(36.0, 39.0):.1f}C",
                "spo2": f"{random.randint(95, 100)}%",
            },
            "diagnosis": {
                "primary": fake.catch_phrase(),
                "secondary": fake.sentence() if random.random() > 0.5 else "",
            },
            "medications": [{"name": fake.word().title(), "dosage": f"{random.randint(5, 1000)}mg", "frequency": random.choice(["daily", "BID", "TID", "PRN", "qHS"])} for _ in range(random.randint(1, 4))],
            "notes": fake.paragraph(nb_sentences=3),
            "disclaimer": "This document contains confidential patient information. Unauthorized access is prohibited.",
        }

    @staticmethod
    def mixed() -> Dict:
        num_metrics = random.randint(3, 5)
        metrics = []
        for _ in range(num_metrics):
            metrics.append({"label": fake.catch_phrase(), "value": f"${random.randint(1000, 100000):,}"})
        num_chart = random.randint(3, 5)
        chart_data = [{"label": str(random.randint(2018, 2025)), "value": random.randint(30, 90)} for _ in range(num_chart)]
        return {
            "report_title": f"{fake.word().title()} {fake.word().title()} Report",
            "report_subtitle": f"Prepared by {fake.company()}",
            "report_date": fake.date_between(start_date="-30d", end_date="today").strftime("%B %d, %Y"),
            "summary": fake.paragraph(nb_sentences=2),
            "metrics": metrics,
            "chart_data": chart_data,
            "table_a": {
                "headers": ["Quarter", "Revenue", "Growth"],
                "rows": [[f"Q{q}", f"${random.randint(10000, 99999):,}", f"{random.randint(5, 50)}%"] for q in range(1, 5)],
            },
            "table_b": {
                "headers": ["Department", "Headcount"],
                "rows": [[fake.word().title(), str(random.randint(5, 200))] for _ in range(random.randint(3, 5))],
            },
            "footer": fake.sentence(),
        }

    @staticmethod
    def generate(doc_type: str) -> Dict:
        generator = getattr(ContentGenerator, doc_type, None)
        if generator is None:
            raise ValueError(f"Unknown document type: {doc_type}")
        return generator()

    @staticmethod
    def generate_qa(metadata: Dict, doc_type: str) -> List[Dict]:
        qa_pairs = []
        if doc_type == "invoice":
            qa_pairs.append({"question": "What is the total amount?", "answer": metadata.get("total", "")})
            qa_pairs.append({"question": "Who is the vendor?", "answer": metadata.get("vendor_name", "")})
            qa_pairs.append({"question": "What is the invoice number?", "answer": metadata.get("invoice_number", "")})
            qa_pairs.append({"question": "Extract all line items as JSON.", "answer": json.dumps(metadata.get("items", []))})
            qa_pairs.append({"question": "What is the due date?", "answer": metadata.get("due_date", "")})
        elif doc_type == "receipt":
            qa_pairs.append({"question": "What is the total amount?", "answer": metadata.get("total", "")})
            qa_pairs.append({"question": "What store is this receipt from?", "answer": metadata.get("store_name", "")})
            qa_pairs.append({"question": "What items were purchased?", "answer": ", ".join(i["name"] for i in metadata.get("items", []))})
        elif doc_type == "form":
            qa_pairs.append({"question": "What is the form title?", "answer": metadata.get("form_title", "")})
            qa_pairs.append({"question": "What fields are on this form?", "answer": ", ".join(f["label"] for f in metadata.get("fields", []))})
        elif doc_type == "table":
            qa_pairs.append({"question": "What is the table title?", "answer": metadata.get("table_title", "")})
            qa_pairs.append({"question": "Convert table to JSON.", "answer": json.dumps(metadata.get("rows", []))})
        elif doc_type == "id_card":
            qa_pairs.append({"question": "What is the cardholder's name?", "answer": f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}"})
            qa_pairs.append({"question": "What type of card is this?", "answer": metadata.get("card_type", "")})
        elif doc_type == "chart":
            qa_pairs.append({"question": "What is the chart title?", "answer": metadata.get("chart_title", "")})
            qa_pairs.append({"question": "What is the maximum value?", "answer": str(metadata.get("max_value", ""))})
        elif doc_type == "contract":
            qa_pairs.append({"question": "What is the contract title?", "answer": metadata.get("contract_title", "")})
            qa_pairs.append({"question": "Who are the parties?", "answer": f"{metadata.get('party_a', '')} and {metadata.get('party_b', '')}"})
        elif doc_type == "letter":
            qa_pairs.append({"question": "Who is the sender?", "answer": metadata.get("sender_name", "")})
            qa_pairs.append({"question": "What is the subject?", "answer": metadata.get("subject", "")})
        elif doc_type == "medical":
            qa_pairs.append({"question": "What is the patient's name?", "answer": metadata.get("patient_name", "")})
            qa_pairs.append({"question": "What is the diagnosis?", "answer": metadata.get("diagnosis", {}).get("primary", "")})
        elif doc_type == "mixed":
            qa_pairs.append({"question": "What is the report title?", "answer": metadata.get("report_title", "")})
        return qa_pairs


def render_html(template_name: str, context: Dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template(template_name)
    return template.render(**context)


def html_to_image(html: str, output_path: Path, viewport_width: int = 1024, browser=None) -> Optional[Path]:
    if browser is not None:
        try:
            ctx = browser.new_context(viewport={"width": viewport_width, "height": 1})
            page = ctx.new_page()
            page.set_content(html, wait_until="networkidle")
            page.screenshot(path=str(output_path), full_page=True)
            ctx.close()
            return output_path
        except Exception as e:
            logger.warning(f"playwright render failed: {e}")
    return None


def augment_image(image_path: Path, output_path: Path) -> Path:
    img = Image.open(image_path)
    img = img.convert("RGB")

    if random.random() > 0.3:
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    if random.random() > 0.5:
        quality = random.randint(60, 95)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    if random.random() > 0.7:
        img = img.convert("L").convert("RGB")

    img.save(str(output_path), "PNG", optimize=True)
    return output_path


def generate_synthetic_documents(
    num_docs: int,
    output_dir: Path,
    doc_types: Optional[List[str]] = None,
    augment: bool = True,
    browser=None,
) -> List[Dict]:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc_types = doc_types or DOCUMENT_TYPES
    manifest = []

    for i in range(num_docs):
        doc_type = random.choice(doc_types)
        content = ContentGenerator.generate(doc_type)

        template_map = {
            "invoice": "invoice.html",
            "receipt": "receipt.html",
            "form": "form.html",
            "table": "table.html",
            "id_card": "id_card.html",
            "chart": "chart.html",
            "contract": "contract.html",
            "letter": "letter.html",
            "medical": "medical.html",
            "mixed": "mixed.html",
        }
        template_name = template_map.get(doc_type, f"{doc_type}.html")
        template_path = TEMPLATE_DIR / template_name

        if not template_path.exists():
            logger.warning(f"Template {template_name} not found. Adding note to manifest.")
            manifest.append({
                "image_path": None,
                "doc_type": doc_type,
                "metadata": content,
                "text": f"Extract document information: <image>",
                "qa_pairs": ContentGenerator.generate_qa(content, doc_type),
                "error": f"Template {template_name} not found",
            })
            continue

        html = render_html(template_name, content)
        image_filename = f"{doc_type}_{i:06d}.png"
        image_path = images_dir / image_filename

        result = html_to_image(html, image_path, browser=browser)
        if result is None:
            try:
                img = render_document(doc_type, content)
                if augment:
                    img = pil_augment(img)
                img.save(str(image_path))
                result = image_path
            except Exception as e:
                logger.error(f"PIL render failed for {doc_type}: {e}")

        if result is None:
            manifest.append({
                "image_path": str(image_path.relative_to(output_dir.parent)),
                "doc_type": doc_type,
                "metadata": content,
                "text": f"Extract document information: <image>",
                "qa_pairs": ContentGenerator.generate_qa(content, doc_type),
                "error": "Render failed",
            })
            continue

        if augment:
            aug_filename = f"{doc_type}_aug_{i:06d}.png"
            aug_path = images_dir / aug_filename
            augment_image(image_path, aug_path)
            image_path = aug_path

        qa_pairs = ContentGenerator.generate_qa(content, doc_type)

        manifest.append({
            "image_path": str(image_path.relative_to(output_dir.parent)),
            "doc_type": doc_type,
            "metadata": content,
            "text": f"Extract document information: <image>",
            "qa_pairs": qa_pairs,
        })

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_docs} documents")

    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Generated {len(manifest)} document entries. Manifest saved to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic documents for TinyDoc-VLM training")
    parser.add_argument("--num-docs", type=int, default=1000, help="Number of documents to generate")
    parser.add_argument("--output-dir", type=str, default="data/synthetic/output", help="Output directory")
    parser.add_argument("--doc-types", type=str, nargs="+", default=None, help="Document types to generate")
    parser.add_argument("--no-augment", action="store_true", help="Skip augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)
    fake.seed_instance(args.seed)

    generate_synthetic_documents(
        num_docs=args.num_docs,
        output_dir=Path(args.output_dir),
        doc_types=args.doc_types,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
