"""
Grounding training pairs from synthetic documents.

Turns renderer annotations ({label, text, bbox_px, bbox}) into prompt-routed
training pairs in the same manifest schema as the markdown/VQA data:
    {image_path, prompt, target, source}

Three task families (LocateAnything-inspired, document-specialized):

1. Locate one field      "Where is the total amount? ... <box>x1 y1 x2 y2</box>"
2. Grounded KIE (JSON)   all labeled fields + normalized bboxes in one JSON object
3. Line-item boxes       each invoice/receipt line item located individually

Coordinates are integers in [0, 1000] relative to image width/height
(LocateAnything / Qwen-VL convention). Boxes are computed BEFORE rotation
augmentation; grounding docs therefore skip rotation (grayscale/JPEG noise
only) so targets stay pixel-accurate.
"""

import json
import random
from typing import Dict, List

# Human-phrased questions per semantic label. Labels not listed here are only
# used in the grounded-KIE JSON task, never in single-field locate prompts.
LOCATE_QUESTIONS = {
    "total": [
        "Where is the total amount?",
        "Locate the total.",
        "Point to the box containing the total amount.",
    ],
    "subtotal": ["Where is the subtotal?"],
    "tax": ["Where is the tax amount?"],
    "discount": ["Where is the discount?"],
    "vendor": ["Where is the vendor name?"],
    "store": ["Where is the store name?"],
    "customer": ["Where is the customer name?"],
    "invoice_number": ["Where is the invoice number?"],
    "invoice_date": ["Where is the invoice date?"],
    "due_date": ["Where is the due date?"],
    "account_number": ["Where is the account number?"],
    "date": ["Where is the transaction date?"],
    "transaction_id": ["Where is the transaction ID?"],
    "date_of_birth": ["Where is the date of birth?"],
    "id_number": ["Where is the ID number?"],
    "department": ["Where is the department?"],
    "expiry_date": ["Where is the expiry date?"],
}

BOX_OPEN = "<box>"
BOX_CLOSE = "</box>"


def _box_str(bbox: List[int]) -> str:
    return " ".join(str(v) for v in bbox)


def locate_pairs(annotations: List[Dict], rng: random.Random) -> List[Dict]:
    """One pair per labeled field with a known question phrasing."""
    pairs = []
    seen = set()
    for ann in annotations:
        label = ann.get("label", "")
        questions = LOCATE_QUESTIONS.get(label)
        if not questions or label in seen:
            continue
        seen.add(label)
        q = rng.choice(questions)
        target = f"{BOX_OPEN}{_box_str(ann['bbox'])}{BOX_CLOSE}"
        pairs.append({
            "prompt": f"{q} Answer with {BOX_OPEN}x1 y1 x2 y2{BOX_CLOSE} coordinates.",
            "target": target,
            "grounding_label": label,
        })
    return pairs


def grounded_kie_pair(annotations: List[Dict], rng: random.Random) -> Dict:
    """All fields + positions as one JSON object (the production KIE output)."""
    fields = {}
    for ann in annotations:
        if ann["label"].startswith("line_item_"):
            continue
        fields[ann["label"]] = {"value": ann["text"], "bbox": ann["bbox"]}
    target = json.dumps(fields, ensure_ascii=False)
    return {
        "prompt": "Extract all key fields with their positions as JSON. "
                  f"Each value is an object {{\"value\": str, \"bbox\": [x1, y1, x2, y2]}} "
                  f"with coordinates in [0, 1000].",
        "target": target,
        "grounding_label": "__all_fields__",
    }


def line_item_pairs(annotations: List[Dict], rng: random.Random) -> List[Dict]:
    """Locate individual line items by content."""
    items = [a for a in annotations if a["label"].startswith("line_item_")]
    pairs = []
    for ann in items[:3]:  # cap per doc to avoid over-weighting long invoices
        item = ann.get("item", {})
        first_desc = str(item.get("description", item.get("name", "")))[:40]
        if not first_desc:
            continue
        q = f"Locate the line item for \"{first_desc}\"."
        target = f"{BOX_OPEN}{_box_str(ann['bbox'])}{BOX_CLOSE}"
        pairs.append({
            "prompt": f"{q} Answer with {BOX_OPEN}x1 y1 x2 y2{BOX_CLOSE} coordinates.",
            "target": target,
            "grounding_label": ann["label"],
        })
    # One aggregate JSON of all line-item boxes per doc
    if items:
        agg = [
            {"item": a.get("item", {}), "bbox": a["bbox"]}
            for a in items
        ]
        pairs.append({
            "prompt": "Extract all line items with their bounding boxes as JSON. "
                      "Each entry is {\"item\": {...}, \"bbox\": [x1, y1, x2, y2]} "
                      "with coordinates in [0, 1000].",
            "target": json.dumps(agg, ensure_ascii=False),
            "grounding_label": "__all_line_items__",
        })
    return pairs


def build_grounding_pairs(annotations: List[Dict], rng: random.Random,
                          p_locate: float = 1.0, p_kie: float = 1.0) -> List[Dict]:
    """Full pair set for one annotated document."""
    pairs = []
    if annotations:
        if rng.random() <= p_locate:
            pairs.extend(locate_pairs(annotations, rng))
            pairs.extend(line_item_pairs(annotations, rng))
        if rng.random() <= p_kie and len(annotations) >= 2:
            pairs.append(grounded_kie_pair(annotations, rng))
    return pairs


def verify_boxes(img, annotations: List[Dict], min_ink: float = 0.005) -> List[str]:
    """Sanity check: each labeled text bbox must contain non-white pixels.

    Crops bbox_px from the (pre-augment) render and measures the fraction of
    pixels darker than 200. Returns list of problems (empty = pass).
    """
    from PIL import Image
    problems = []
    gray = img.convert("L")
    w, h = gray.size
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox_px"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            problems.append(f"{ann['label']}: degenerate bbox {ann['bbox_px']}")
            continue
        crop = gray.crop((x1, y1, x2, y2))
        px = list(crop.getdata())
        dark = sum(1 for v in px if v < 200)
        frac = dark / max(len(px), 1)
        if frac < min_ink:
            problems.append(f"{ann['label']}: bbox looks empty (ink={frac:.4f})")
    return problems
