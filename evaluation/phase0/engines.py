"""Inference engines for Phase-0 baselines.

Engines:
  - ocr_regex: Tesseract/pytesseract if available, else a trivial empty baseline
    (reports the floor). Field rules: company=first non-empty header heuristic
    is intentionally weak — this is a *floor*, not a product parser.
  - smolvlm2: local HF weights (KIOXIA path or HF id)
  - ollama: any chat/vision model (e.g. qwen2.5vl:3b)
  - ppdocbee: PaddlePaddle/PP-DocBee-2B via transformers if downloaded
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from metrics import extract_json, FIELDS

KIOXIA_SMOL = "/Volumes/KIOXIA 1TB/smolvlm2-2.2b-instruct"
KIOXIA_HF_SMOL = (
    "/Volumes/KIOXIA 1TB/huggingface_cache/hub/"
    "models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots"
)
PPDOC_PATH = "/Volumes/KIOXIA 1TB/models/PP-DocBee-2B"

# Identical text to sdk/tinydoc/pipeline.py EXTRACT_PROMPT, which is the single
# source of truth. tests/test_prompt_parity.py fails if these ever drift, so the
# phase-0 harness and the product path always score under the same instructions.
EXTRACT_PROMPT = (
    'Extract receipt fields. Reply with JSON only, exactly four keys:\n'
    '{"company":"...","date":"...","address":"...","total":"..."}\n'
    "Rules:\n"
    "- company: merchant name exactly as printed (header/logo line).\n"
    "- date: transaction date exactly as printed.\n"
    "- address: full postal address as printed (street, city, postcode). "
    "Transcribe character-by-character; do not guess. "
    "Watch lookalikes: O/0, I/1/l, G/6, B/8, S/5, Z/2. "
    "Copy postcode digits exactly. Include every address line; "
    "do not invent a second street.\n"
    "- total: grand total / amount due.\n"
    "Never omit total. JSON only, no markdown."
)


class BaseEngine:
    name = "base"

    def extract(self, image_path: str) -> Dict[str, Any]:
        t0 = time.time()
        raw = self._raw(image_path)
        fields = extract_json(raw) or {}
        # keep only known fields as strings
        out = {k: str(fields.get(k, "")) for k in FIELDS if k in fields}
        for k in FIELDS:
            out.setdefault(k, "")
        return {
            "engine": self.name,
            "fields": out,
            "raw": raw[:2000],
            "schema_ready": extract_json(raw) is not None,
            "latency_ms": (time.time() - t0) * 1000,
        }

    def _raw(self, image_path: str) -> str:
        raise NotImplementedError


class OcrRegexEngine(BaseEngine):
    """Floor baseline: OCR text → regex-ish field scrape (deliberately simple)."""

    name = "ocr_regex"

    def _raw(self, image_path: str) -> str:
        text = ""
        try:
            import pytesseract

            text = pytesseract.image_to_string(Image.open(image_path))
        except Exception as e:
            return json.dumps({"error": f"ocr_failed: {e}"})
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        company = lines[0] if lines else ""
        date_m = re.search(
            r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}|\d{1,2}\s+\w+\s+\d{4})",
            text,
        )
        total_m = re.search(
            r"(?:grand\s+total|total|amount\s+due)\s*[:\s]*\s*([\$]?\s*\d[\d,]*\.?\d*)",
            text,
            re.I,
        )
        if not total_m:
            ms = re.findall(r"\$?\s*\d[\d,]*\.\d{2}", text)
            total_m_vals = ms[-1:] if ms else []
        else:
            total_m_vals = [total_m.group(1)]
        fields = {
            "company": company,
            "date": date_m.group(1) if date_m else "",
            "address": ", ".join(lines[1:3]),
            "total": total_m_vals[0] if total_m_vals else "",
        }
        return json.dumps(fields)


class SmolVLM2Engine(BaseEngine):
    name = "smolvlm2_base"

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = model_path or self._discover()
        self.device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        # fp16 on cuda/mps where possible; fp32 on cpu for correctness
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        print(f"[smolvlm2] loading {path} on {self.device} dtype={dtype}")
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.name = "smolvlm2_base"

    @staticmethod
    def _discover() -> str:
        p = Path(KIOXIA_SMOL)
        if (p / "config.json").exists():
            return str(p)
        snaps = Path(KIOXIA_HF_SMOL)
        if snaps.exists():
            subs = sorted([d for d in snaps.iterdir() if d.is_dir()])
            if subs:
                return str(subs[-1])
        return "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def _raw(self, image_path: str) -> str:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        try:
            inputs = self.processor(
                text=[text], images=[[img]], return_tensors="pt"
            )
        except TypeError:
            inputs = self.processor(
                text=[text], images=[[img]], return_tensors="pt",
                processor_kwargs=None,
            )
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
            if k != "image_token_id"
        }
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.12,
            )
        seq = out[0]
        n_in = inputs["input_ids"].shape[-1]
        gen = seq[n_in:] if seq.shape[0] >= n_in else seq
        tok = getattr(self.processor, "tokenizer", self.processor)
        return tok.decode(gen, skip_special_tokens=True).strip()


class OllamaEngine(BaseEngine):
    def __init__(self, model: str = "qwen2.5vl:3b", host: Optional[str] = None):
        import os

        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.name = f"ollama:{model}"

    def _raw(self, image_path: str) -> str:
        import base64
        import urllib.error
        import urllib.request

        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT,
                    "images": [b64],
                }
            ],
            "stream": False,
            # NOTE: do NOT set format="json" — Ollama constrained decoding
            # truncates long address fields and often returns empty total on
            # qwen2.5vl:3b. Unconstrained + extract_json salvage is the fix
            # (Phase-1 SDK uses the same approach).
            "options": {"temperature": 0, "num_predict": 512},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode())
            return (body.get("message") or {}).get("content", "") or ""
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"ollama_chat: {e}"})


class PPDocBeeEngine(BaseEngine):
    name = "ppdocbee_2b"

    def __init__(self, model_path: str = PPDOC_PATH):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not Path(model_path, "config.json").exists():
            raise FileNotFoundError(model_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        print(f"[ppdocbee] loading {model_path} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def _raw(self, image_path: str) -> str:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        # PP-DocBee uses Qwen2-VL chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[[img]], return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        n_in = inputs["input_ids"].shape[-1]
        gen = out[0][n_in:]
        tok = getattr(self.processor, "tokenizer", self.processor)
        return tok.decode(gen, skip_special_tokens=True).strip()


def build_engine(name: str, **kwargs) -> BaseEngine:
    if name == "ocr_regex":
        return OcrRegexEngine()
    if name == "smolvlm2":
        return SmolVLM2Engine(**kwargs)
    if name.startswith("ollama"):
        model = name.split(":", 1)[1] if ":" in name.split("ollama", 1)[-1].lstrip(":") else kwargs.get("model", "qwen2.5vl:3b")
        # allow "ollama" or "ollama:qwen2.5vl:3b"
        if name == "ollama":
            model = kwargs.get("model", "qwen2.5vl:3b")
        elif name.startswith("ollama:"):
            model = name[len("ollama:") :]
        return OllamaEngine(model=model)
    if name == "ppdocbee":
        return PPDocBeeEngine(**kwargs)
    raise ValueError(f"unknown engine {name}")
