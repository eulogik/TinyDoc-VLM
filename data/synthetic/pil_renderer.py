"""
PIL-based document image renderer.
Generates document images directly without external rendering dependencies.
Faster and more reliable than HTML→PDF→Image pipeline for initial data generation.
"""

from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
import random
import math

FONT_CACHE = {}


def _get_font(size: int = 12) -> ImageFont.FreeTypeFont:
    cache_key = f"default_{size}"
    if cache_key not in FONT_CACHE:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except (IOError, OSError):
                font = ImageFont.load_default()
        FONT_CACHE[cache_key] = font
    return FONT_CACHE[cache_key]


def _draw_text(draw: ImageDraw.Draw, xy, text, font_size=12, fill=(0, 0, 0), bold=False, align="left"):
    if not text:
        return
    font = _get_font(font_size)
    try:
        draw.text(xy, str(text), font=font, fill=fill, align=align)
    except Exception:
        pass


def _draw_rect(draw: ImageDraw.Draw, xy, fill=None, outline=None, width=1):
    draw.rectangle(xy, fill=fill, outline=outline, width=width)


def _draw_line(draw: ImageDraw.Draw, xy, fill=(200, 200, 200), width=1):
    draw.line(xy, fill=fill, width=width)


def render_invoice(content: Dict) -> Image.Image:
    img = Image.new("RGB", (800, 1050), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (30, 20), content.get("vendor_name", "Company"), font_size=22, fill=(30, 58, 138))
    _draw_text(draw, (30, 50), content.get("vendor_address", ""), font_size=9, fill=(100, 100, 100))
    _draw_text(draw, (30, 65), f"Phone: {content.get('vendor_phone', '')}", font_size=9, fill=(100, 100, 100))

    _draw_text(draw, (600, 20), "INVOICE", font_size=26, fill=(59, 130, 246))
    _draw_text(draw, (560, 55), f"Invoice #: {content.get('invoice_number', '')}", font_size=10, fill=(80, 80, 80))
    _draw_text(draw, (560, 70), f"Date: {content.get('invoice_date', '')}", font_size=10, fill=(80, 80, 80))
    _draw_text(draw, (560, 85), f"Due: {content.get('due_date', '')}", font_size=10, fill=(80, 80, 80))

    _draw_line(draw, [(30, 95), (770, 95)], fill=(59, 130, 246), width=2)

    _draw_text(draw, (30, 110), "BILL TO", font_size=11, fill=(30, 58, 138))
    _draw_text(draw, (30, 128), content.get("customer_name", ""), font_size=11)
    _draw_text(draw, (30, 145), content.get("customer_address", ""), font_size=9, fill=(80, 80, 80))

    _draw_text(draw, (450, 110), "PAYMENT INFO", font_size=11, fill=(30, 58, 138))
    _draw_text(draw, (450, 128), f"Bank: {content.get('bank_name', '')}", font_size=9, fill=(80, 80, 80))
    _draw_text(draw, (450, 143), f"Account: {content.get('account_number', '')}", font_size=9, fill=(80, 80, 80))

    y = 180
    _draw_rect(draw, [(30, y), (770, y + 25)], fill=(243, 244, 246))
    _draw_text(draw, (35, y + 5), "Description", font_size=10, fill=(30, 58, 138))
    _draw_text(draw, (500, y + 5), "Qty", font_size=10, fill=(30, 58, 138))
    _draw_text(draw, (570, y + 5), "Unit Price", font_size=10, fill=(30, 58, 138))
    _draw_text(draw, (690, y + 5), "Amount", font_size=10, fill=(30, 58, 138))
    _draw_line(draw, [(30, y + 25), (770, y + 25)], fill=(59, 130, 246), width=2)

    y += 30
    for item in content.get("items", []):
        _draw_text(draw, (35, y), item.get("description", ""), font_size=10)
        _draw_text(draw, (510, y), item.get("quantity", ""), font_size=10)
        _draw_text(draw, (580, y), item.get("unit_price", ""), font_size=10)
        _draw_text(draw, (700, y), item.get("amount", ""), font_size=10)
        y += 22

    y = max(y + 10, 750)
    _draw_line(draw, [(530, y), (770, y)])
    _draw_text(draw, (535, y + 5), "Subtotal:", font_size=10, fill=(80, 80, 80))
    _draw_text(draw, (700, y + 5), content.get("subtotal", ""), font_size=10)
    _draw_text(draw, (535, y + 22), f"Tax ({content.get('tax_rate', '')}%):", font_size=10, fill=(80, 80, 80))
    _draw_text(draw, (700, y + 22), content.get("tax_amount", ""), font_size=10)

    y += 50
    _draw_rect(draw, [(530, y - 10), (770, y + 20)], fill=(239, 246, 255))
    _draw_line(draw, [(530, y - 10), (770, y - 10)], fill=(59, 130, 246), width=2)
    _draw_line(draw, [(530, y + 20), (770, y + 20)], fill=(59, 130, 246), width=2)
    _draw_text(draw, (535, y), "TOTAL DUE:", font_size=12, fill=(30, 58, 138))
    _draw_text(draw, (700, y), content.get("total", ""), font_size=12, fill=(30, 58, 138))

    _draw_text(draw, (200, y + 50), "Thank you for your business!", font_size=9, fill=(160, 160, 160))
    return img


def render_receipt(content: Dict) -> Image.Image:
    img = Image.new("RGB", (450, 700), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (225, 20), content.get("store_name", "Store"), font_size=16, fill=(0, 0, 0), align="center")
    _draw_text(draw, (225, 42), content.get("store_address", ""), font_size=8, fill=(80, 80, 80), align="center")

    _draw_line(draw, [(20, 60), (430, 60)], fill=(0, 0, 0), width=1)

    _draw_text(draw, (30, 70), f"Date: {content.get('txn_date', '')}", font_size=9)
    _draw_text(draw, (250, 70), f"Time: {content.get('txn_time', '')}", font_size=9)
    _draw_text(draw, (30, 85), f"Txn #: {content.get('txn_id', '')}", font_size=9)
    _draw_text(draw, (250, 85), f"Cashier: {content.get('cashier_name', '')}", font_size=9)

    _draw_line(draw, [(20, 100), (430, 100)], fill=(0, 0, 0), width=1)

    y = 110
    _draw_text(draw, (30, y), "ITEM", font_size=9, bold=True)
    _draw_text(draw, (280, y), "QTY", font_size=9, bold=True)
    _draw_text(draw, (340, y), "PRICE", font_size=9, bold=True)
    _draw_line(draw, [(20, y + 15), (430, y + 15)], fill=(0, 0, 0), width=1)

    y += 20
    for item in content.get("items", []):
        _draw_text(draw, (30, y), item.get("name", ""), font_size=9)
        _draw_text(draw, (290, y), item.get("quantity", ""), font_size=9)
        _draw_text(draw, (350, y), item.get("amount", ""), font_size=9)
        y += 18

    _draw_line(draw, [(20, y), (430, y)], fill=(0, 0, 0), width=1)
    y += 8

    _draw_text(draw, (300, y), "Subtotal:", font_size=9)
    _draw_text(draw, (370, y), content.get("subtotal", ""), font_size=9)
    y += 15
    _draw_text(draw, (300, y), f"Tax:", font_size=9)
    _draw_text(draw, (370, y), content.get("tax_amount", ""), font_size=9)

    if content.get("discount_amount", "$0.00") != "$0.00":
        y += 15
        _draw_text(draw, (300, y), "Discount:", font_size=9)
        _draw_text(draw, (370, y), f"-{content.get('discount_amount', '')}", font_size=9)

    y += 20
    _draw_line(draw, [(280, y - 5), (430, y - 5)], fill=(0, 0, 0), width=2)
    _draw_text(draw, (300, y), "TOTAL:", font_size=12, bold=True)
    _draw_text(draw, (370, y), content.get("total", ""), font_size=12, bold=True)
    _draw_line(draw, [(280, y + 15), (430, y + 15)], fill=(0, 0, 0), width=2)

    y += 30
    _draw_text(draw, (225, y), content.get("footer_message", ""), font_size=7, fill=(80, 80, 80), align="center")
    _draw_text(draw, (225, y + 15), "Thank you for shopping!", font_size=8, fill=(80, 80, 80), align="center")

    return img


def render_form(content: Dict) -> Image.Image:
    img = Image.new("RGB", (800, 1000), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (30, 20), content.get("form_title", "Form"), font_size=18, fill=(30, 58, 138))
    _draw_text(draw, (30, 45), f"Form ID: {content.get('form_id', '')}", font_size=10, fill=(100, 100, 100))

    _draw_rect(draw, [(30, 60), (770, 85)], fill=(243, 244, 246))
    _draw_text(draw, (40, 66), content.get("instructions", ""), font_size=9, fill=(80, 80, 80))

    y = 100
    for field in content.get("fields", []):
        _draw_rect(draw, [(30, y), (770, y + 50)], fill=(255, 255, 255), outline=(220, 220, 220))
        label = field.get("label", "")
        if field.get("required"):
            label += " *"
        _draw_text(draw, (40, y + 5), label, font_size=10, fill=(30, 58, 138))
        _draw_text(draw, (40, y + 25), field.get("value", ""), font_size=10, fill=(60, 60, 60))
        y += 55

    if content.get("has_signature"):
        _draw_line(draw, [(30, y + 50), (400, y + 50)])
        _draw_text(draw, (30, y + 55), "Signature", font_size=9, fill=(100, 100, 100))

    return img


def render_table(content: Dict) -> Image.Image:
    headers = content.get("headers", [])
    rows = content.get("rows", [])
    col_w = min(140, 700 // max(len(headers), 1))
    width = max(800, len(headers) * col_w + 60)
    height = max(400, 60 + len(rows) * 25 + 80)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (30, 15), content.get("table_title", "Table"), font_size=16, fill=(30, 58, 138))

    y = 50
    for col_idx, header in enumerate(headers):
        x = 30 + col_idx * col_w
        _draw_rect(draw, [(x, y), (x + col_w - 2, y + 28)], fill=(30, 58, 138))
        _draw_text(draw, (x + 5, y + 5), header, font_size=9, fill=(255, 255, 255))

    y += 30
    for row_idx, row in enumerate(rows):
        for col_idx, header in enumerate(headers):
            x = 30 + col_idx * col_w
            bg = (249, 250, 251) if row_idx % 2 == 0 else (255, 255, 255)
            _draw_rect(draw, [(x, y), (x + col_w - 2, y + 22)], fill=bg)
            _draw_text(draw, (x + 5, y + 3), row.get(header, ""), font_size=9)
        y += 22

    if content.get("footnote"):
        _draw_text(draw, (30, y + 10), content["footnote"], font_size=8, fill=(100, 100, 100), italic=True)

    return img


def render_id_card(content: Dict) -> Image.Image:
    img = Image.new("RGB", (430, 270), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_rect(draw, [(0, 0), (429, 269)], fill=(224, 231, 255), width=2)

    _draw_rect(draw, [(10, 10), (140, 260)], fill=(200, 210, 240))
    initials = f"{content.get('first_name', '?')[0]}{content.get('last_name', '?')[0]}"
    _draw_text(draw, (75, 125), initials, font_size=36, fill=(99, 102, 241))

    _draw_text(draw, (155, 20), content.get("card_type", ""), font_size=8, fill=(99, 102, 241))
    _draw_text(draw, (155, 40), f"{content.get('first_name', '')} {content.get('last_name', '')}", font_size=14, fill=(30, 58, 138))

    fields = [
        ("DOB", content.get("date_of_birth", "")),
        ("ID #", content.get("id_number", "")),
        ("Dept", content.get("department", "")),
        ("Exp", content.get("expiry_date", "")),
    ]
    y = 75
    for label, value in fields:
        _draw_text(draw, (155, y), label, font_size=7, fill=(100, 100, 100))
        _draw_text(draw, (155, y + 12), value, font_size=10)
        y += 35

    return img


def render_chart(content: Dict) -> Image.Image:
    data = content.get("data", [])
    max_val = content.get("max_value", 100)
    num_bars = len(data)
    bar_w = 50
    width = max(600, num_bars * (bar_w + 20) + 80)
    height = 400

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (width // 2, 15), content.get("chart_title", "Chart"), font_size=16, fill=(30, 58, 138), align="center")

    chart_bottom = height - 60
    chart_top = 50
    chart_height = chart_bottom - chart_top

    _draw_line(draw, [(60, chart_top), (60, chart_bottom)], fill=(200, 200, 200))
    _draw_line(draw, [(60, chart_bottom), (width - 30, chart_bottom)], fill=(200, 200, 200))

    max_val = max(max_val, 1)
    for i, item in enumerate(data):
        bar_h = (item["value"] / max_val) * chart_height
        x = 80 + i * (bar_w + 20)
        _draw_rect(draw, [(x, chart_bottom - bar_h), (x + bar_w, chart_bottom)], fill=(59, 130, 246))
        _draw_text(draw, (x + bar_w // 2, chart_bottom - bar_h - 18), str(item["value"]), font_size=8, fill=(30, 58, 138))
        _draw_text(draw, (x + bar_w // 2, chart_bottom + 5), item["label"], font_size=8, fill=(80, 80, 80))

    if content.get("footnote"):
        _draw_text(draw, (width // 2, height - 20), content["footnote"], font_size=8, fill=(100, 100, 100))

    return img


def render_letter(content: Dict) -> Image.Image:
    img = Image.new("RGB", (800, 1050), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (400, 20), content.get("company_name", ""), font_size=18, fill=(30, 58, 138), align="center")
    _draw_text(draw, (400, 45), content.get("company_address", ""), font_size=9, fill=(100, 100, 100), align="center")
    _draw_line(draw, [(50, 65), (750, 65)], fill=(30, 58, 138), width=2)

    _draw_text(draw, (580, 80), content.get("letter_date", ""), font_size=10)

    _draw_text(draw, (50, 110), content.get("recipient_name", ""), font_size=11)
    _draw_text(draw, (50, 128), content.get("recipient_address", ""), font_size=10, fill=(80, 80, 80))

    _draw_text(draw, (50, 160), f"Re: {content.get('subject', '')}", font_size=11, bold=True)

    y = 195
    for para in content.get("body_paragraphs", []):
        _draw_text(draw, (50, y), para, font_size=11)
        y += 30

    y = max(y + 20, 500)
    _draw_text(draw, (50, y), content.get("closing_sentence", ""), font_size=11)
    _draw_text(draw, (50, y + 35), "Sincerely,", font_size=11)
    _draw_text(draw, (50, y + 80), content.get("sender_name", ""), font_size=11, bold=True)
    _draw_text(draw, (50, y + 98), content.get("sender_title", ""), font_size=9, fill=(100, 100, 100))

    return img


def render_contract(content: Dict) -> Image.Image:
    img = Image.new("RGB", (800, 1100), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (400, 20), content.get("contract_title", "AGREEMENT"), font_size=16, bold=True, align="center")

    _draw_rect(draw, [(50, 55), (400, 90)], fill=(255, 255, 255), outline=(200, 200, 200))
    _draw_text(draw, (60, 60), f"Party A: {content.get('party_a', '')}", font_size=10)
    _draw_rect(draw, [(400, 55), (750, 90)], fill=(255, 255, 255), outline=(200, 200, 200))
    _draw_text(draw, (410, 60), f"Party B: {content.get('party_b', '')}", font_size=10)

    _draw_text(draw, (400, 105), f"Effective: {content.get('effective_date', '')}", font_size=10, align="center")

    y = 130
    for section in content.get("sections", []):
        _draw_line(draw, [(50, y), (750, y)])
        _draw_text(draw, (50, y + 5), section.get("title", ""), font_size=11, bold=True)
        y += 25
        for clause in section.get("clauses", []):
            _draw_text(draw, (65, y), clause, font_size=10)
            y += 20
        y += 10

    y = max(y + 30, 700)
    _draw_line(draw, [(50, y), (750, y)])
    y += 20
    _draw_text(draw, (50, y), f"{content.get('party_a', '')}: _________________________", font_size=10)
    y += 25
    _draw_text(draw, (50, y), f"{content.get('party_b', '')}: _________________________", font_size=10)
    _draw_text(draw, (50, y + 30), "Date: ______________", font_size=9, fill=(100, 100, 100))

    return img


def render_medical(content: Dict) -> Image.Image:
    img = Image.new("RGB", (800, 900), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_text(draw, (30, 15), content.get("facility_name", "Facility"), font_size=18, fill=(5, 150, 105))
    _draw_line(draw, [(30, 42), (770, 42)], fill=(5, 150, 105), width=2)

    _draw_rect(draw, [(30, 55), (770, 130)], fill=(240, 253, 244))
    info_fields = [
        ("Patient", content.get("patient_name", "")),
        ("DOB", content.get("patient_dob", "")),
        ("MRN", content.get("mrn", "")),
        ("Date", content.get("visit_date", "")),
        ("Provider", content.get("provider_name", "")),
    ]
    for i, (label, value) in enumerate(info_fields):
        x = 45 + (i % 3) * 240
        y = 60 + (i // 3) * 30
        _draw_text(draw, (x, y), label, font_size=8, fill=(100, 100, 100))
        _draw_text(draw, (x, y + 12), value, font_size=10)

    y = 145
    _draw_text(draw, (30, y), "VITAL SIGNS", font_size=11, fill=(5, 150, 105))
    _draw_line(draw, [(30, y + 16), (770, y + 16)], fill=(200, 200, 200))
    y += 22
    vitals = content.get("vitals", {})
    v_fields = [("BP", vitals.get("bp", "")), ("HR", vitals.get("hr", "")), ("RR", vitals.get("rr", "")),
                ("Temp", vitals.get("temp", "")), ("SpO2", vitals.get("spo2", ""))]
    for i, (label, value) in enumerate(v_fields):
        x = 45 + i * 145
        _draw_text(draw, (x, y), f"{label}: {value}", font_size=10)

    y += 30
    _draw_text(draw, (30, y), "DIAGNOSIS", font_size=11, fill=(5, 150, 105))
    _draw_line(draw, [(30, y + 16), (770, y + 16)], fill=(200, 200, 200))
    y += 22
    _draw_text(draw, (45, y), content.get("diagnosis", {}).get("primary", ""), font_size=10)

    y += 30
    _draw_text(draw, (30, y), "MEDICATIONS", font_size=11, fill=(5, 150, 105))
    _draw_line(draw, [(30, y + 16), (770, y + 16)], fill=(200, 200, 200))
    y += 22
    for med in content.get("medications", []):
        _draw_text(draw, (45, y), f"{med.get('name', '')} - {med.get('dosage', '')} - {med.get('frequency', '')}", font_size=10)
        y += 18

    y += 20
    _draw_text(draw, (30, y), "NOTES", font_size=11, fill=(5, 150, 105))
    _draw_line(draw, [(30, y + 16), (770, y + 16)], fill=(200, 200, 200))
    y += 22
    _draw_text(draw, (45, y), content.get("notes", ""), font_size=10)

    return img


RENDERERS = {
    "invoice": render_invoice,
    "receipt": render_receipt,
    "form": render_form,
    "table": render_table,
    "id_card": render_id_card,
    "chart": render_chart,
    "letter": render_letter,
    "contract": render_contract,
    "medical": render_medical,
}


def render_document(doc_type: str, content: Dict) -> Image.Image:
    renderer = RENDERERS.get(doc_type)
    if renderer is None:
        img = Image.new("RGB", (800, 400), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        _draw_text(draw, (50, 50), f"Document Type: {doc_type}", font_size=18, fill=(30, 58, 138))
        y = 80
        for k, v in content.items():
            if isinstance(v, str):
                _draw_text(draw, (50, y), f"{k}: {v[:80]}", font_size=10)
                y += 18
        return img
    return renderer(content)


def augment_image(img: Image.Image) -> Image.Image:
    if random.random() > 0.5:
        angle = random.uniform(-2, 2)
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    if random.random() > 0.3:
        img = img.convert("L").convert("RGB")

    return img
