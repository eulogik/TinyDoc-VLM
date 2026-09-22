"""Draw field evidence boxes + labels on a receipt image (A6 demo overlay)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image, ImageDraw, ImageFont

# Distinct colors per field (RGB)
_FIELD_COLORS = {
    "company": (0, 140, 255),
    "date": (0, 180, 80),
    "address": (255, 140, 0),
    "total": (220, 30, 60),
}


def _font(size: int = 14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def draw_overlay(
    image_path: Union[str, Path],
    field_results: List,
    out_path: Union[str, Path],
    *,
    draw_empty: bool = False,
) -> Path:
    """Rasterize evidence bboxes + field labels onto a copy of the page.

    ``field_results`` is a list of ``FieldResult`` (or dicts with
    name/value/evidence/confidence). Writes PNG to ``out_path``.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(max(12, img.size[1] // 60))
    pad = 3

    for fr in field_results:
        if isinstance(fr, dict):
            name = fr.get("name", "")
            value = fr.get("value", "")
            ev = fr.get("evidence")
            conf = fr.get("confidence", 0.0)
        else:
            name = getattr(fr, "name", "")
            value = getattr(fr, "value", "")
            ev = getattr(fr, "evidence", None)
            conf = getattr(fr, "confidence", 0.0)

        if not value and not draw_empty:
            continue
        bbox = (ev or {}).get("bbox") if isinstance(ev, dict) else None
        color = _FIELD_COLORS.get(name, (120, 120, 120))

        label = f"{name}: {value[:48]}"
        if conf:
            label += f" ({conf:.2f})"

        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            # slight inflate
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(img.size[0] - 1, x1 + pad), min(img.size[1] - 1, y1 + pad)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            # label above box if room, else below
            ty = y0 - 18 if y0 >= 18 else y1 + 2
            _draw_label(draw, img.size, label, x0, ty, color, font)
        else:
            # no bbox — chip goes into the top-right legend below
            pass

    # legend / no-bbox fields stacked top-right
    y = 8
    for fr in field_results:
        if isinstance(fr, dict):
            name, value, ev, conf = fr.get("name"), fr.get("value"), fr.get("evidence"), fr.get("confidence", 0)
        else:
            name, value = fr.name, fr.value
            ev, conf = fr.evidence, fr.confidence
        if not value:
            continue
        if ev and isinstance(ev, dict) and ev.get("bbox"):
            continue  # already drawn on page
        color = _FIELD_COLORS.get(name, (120, 120, 120))
        label = f"{name}: {value[:40]}"
        if conf:
            label += f" ({conf:.2f})"
        # right-aligned chip
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x1 = img.size[0] - 8
        x0 = x1 - tw - 10
        draw.rectangle([x0, y, x1, y + th + 6], fill=color)
        draw.text((x0 + 5, y + 2), label, fill=(255, 255, 255), font=font)
        y += th + 10

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def _draw_label(draw, img_size, text: str, x: float, y: float, color, font) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(0, min(x, img_size[0] - tw - 8))
    y = max(0, min(y, img_size[1] - th - 8))
    draw.rectangle([x, y, x + tw + 8, y + th + 6], fill=color)
    draw.text((x + 4, y + 2), text, fill=(255, 255, 255), font=font)
