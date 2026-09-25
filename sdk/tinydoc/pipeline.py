"""Phase-1 product pipeline: engine router + JSON schema + evidence + confidence.

Engines (measured Phase 0, see evaluation/phase0/results/baseline_table.md):
  - ollama:qwen2.5vl:3b  (default; best free field-F1 on SROIE)
  - ocr_regex             (floor / offline fallback)
  - smolvlm2              (optional local HF weights)

Large model weights live on the KIOXIA external disk (never the system volume).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .evidence import Evidence, attach_evidence

FIELDS = ("company", "date", "address", "total")

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
# Prefer packaged schema (pip install); fall back to repo evaluation path.
_SCHEMA_NAME = "sroie_receipt.schema.json"
SCHEMA_PATH = Path(__file__).absolute().parent / "schemas" / _SCHEMA_NAME
if not SCHEMA_PATH.exists():
    # evaluation/ is a symlink; never resolve() — absolute(__file__) only
    SCHEMA_PATH = (
        Path(__file__).absolute().parent.parent.parent
        / "evaluation"
        / "phase0"
        / "schemas"
        / _SCHEMA_NAME
    )

# KIOXIA model roots (external disk)
KIOXIA_ROOT = Path("/Volumes/KIOXIA 1TB")
SMOL_PATH = KIOXIA_ROOT / "smolvlm2-2.2b-instruct"
PPDOC2_PATH = KIOXIA_ROOT / "models" / "PPDocBee2-3B"


@dataclass
class FieldResult:
    name: str
    value: str
    present: bool
    evidence: Optional[Dict[str, Any]] = None
    confidence: float = 0.0


@dataclass
class DocumentResult:
    image_path: str
    engine: str
    fields: Dict[str, str]
    field_results: List[FieldResult]
    schema_valid: bool
    confidence: float
    raw: str
    latency_ms: float
    evidence_coverage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "engine": self.engine,
            "fields": self.fields,
            "schema_valid": self.schema_valid,
            "confidence": self.confidence,
            "evidence_coverage": self.evidence_coverage,
            "latency_ms": self.latency_ms,
            "raw": self.raw[:2000],
            "fields_detail": [
                {
                    "name": f.name,
                    "value": f.value,
                    "present": f.present,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                }
                for f in self.field_results
            ],
        }


def load_schema() -> Optional[Dict[str, Any]]:
    try:
        if SCHEMA_PATH.exists():
            return json.loads(SCHEMA_PATH.read_text())
    except Exception:
        pass
    return None


def schema_is_valid(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    schema = load_schema()
    if schema is None:
        return all(isinstance(obj.get(k), str) and obj.get(k) for k in FIELDS)
    try:
        import jsonschema

        jsonschema.validate(instance=obj, schema=schema)
        return True
    except Exception:
        return False


_FIELD_MAXLEN = {"company": 200, "date": 40, "address": 400, "total": 40}


def collapse_repetition(text: str, min_phrase_reps: int = 3) -> str:
    """Cut degenerate n-gram loops from VLM output (not metric gaming — garbage in)."""
    if not text:
        return text
    words = text.split()
    for plen in range(2, 10):
        if len(words) < plen * min_phrase_reps:
            continue
        i = 0
        out: List[str] = []
        while i < len(words):
            unit = words[i : i + plen]
            if len(unit) < plen:
                out.extend(words[i:])
                break
            reps = 1
            while True:
                nxt = words[i + plen * reps : i + plen * (reps + 1)]
                if nxt == unit:
                    reps += 1
                else:
                    break
            if reps >= min_phrase_reps:
                out.extend(unit)  # keep one copy
                i += plen * reps
            else:
                out.append(words[i])
                i += 1
        words = out
    # single-token runs
    i = 0
    out = []
    while i < len(words):
        j = i
        while j < len(words) and words[j] == words[i]:
            j += 1
        if (j - i) >= 4:
            out.append(words[i])
        else:
            out.extend(words[i:j])
        i = j
    # adjacent duplicate tokens (comma glued): "PERAK, PERAK,"
    words2 = []
    for w in out:
        if words2 and w == words2[-1] and len(w) > 3:
            continue
        words2.append(w)
    return " ".join(words2)


def sanitize_fields(fields: Dict[str, Any]) -> Dict[str, str]:
    clean: Dict[str, str] = {}
    for k in FIELDS:
        v = str(fields.get(k, "") or "")
        v = collapse_repetition(v)
        v = re.sub(r"\s+", " ", v).strip()
        max_len = _FIELD_MAXLEN.get(k, 400)
        if len(v) > max_len:
            v = v[:max_len].rstrip(" ,;")
        clean[k] = v
    return clean


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    if start < 0:
        return None
    # If multiple top-level objects appear, keep only the first complete one
    # (or the first fragment through end of text for salvage).
    depth = 0
    in_str = esc = False
    end = None
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is not None:
        try:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        t = t[start : end + 1]
    else:
        t = t[start:]
    # salvage: close open braces/strings
    in_str = esc = False
    for ch in t:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
    if in_str:
        t = t + '"'
    body = t
    open_b = 0
    in_s = e = False
    for ch in body:
        if in_s:
            if e:
                e = False
            elif ch == "\\":
                e = True
            elif ch == '"':
                in_s = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "{":
            open_b += 1
        elif ch == "}":
            open_b -= 1
    cand = body + "}" * max(open_b, 0)

    def _close_braces(s: str) -> str:
        ob = 0
        ins = esc2 = False
        for ch in s:
            if ins:
                if esc2:
                    esc2 = False
                elif ch == "\\":
                    esc2 = True
                elif ch == '"':
                    ins = False
                continue
            if ch == '"':
                ins = True
            elif ch == "{":
                ob += 1
            elif ch == "}":
                ob -= 1
        return s + "}" * max(ob, 0)

    for _ in range(8):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
            break
        except json.JSONDecodeError:
            pass
        # strip trailing closing braces, then drop incomplete tail, then reclose
        stripped = cand.rstrip().rstrip("}")
        # drop incomplete key / key:value tail
        stripped = re.sub(r",\s*\"(?:[^\"\\]|\\.)*\"\s*:?\s*$", "", stripped)
        stripped = re.sub(r",\s*$", "", stripped)
        stripped = re.sub(r"[,:]\s*$", "", stripped)
        # drop dangling bare key after {
        stripped = re.sub(r"\{\s*\"(?:[^\"\\]|\\.)*\"\s*:?\s*$", "{", stripped)
        if not stripped.strip().endswith("}"):
            nxt = _close_braces(stripped)
        else:
            nxt = stripped
        if nxt == cand:
            # last resort: cut to last complete "key": "value" pair
            pairs = list(
                re.finditer(
                    r"\"(?:[^\"\\]|\\.)*\"\s*:\s*\"(?:[^\"\\]|\\.)*\"", cand
                )
            )
            if pairs:
                nxt = "{" + cand[pairs[0].start() : pairs[-1].end()] + "}"
            else:
                break
        cand = nxt
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class BaseEngine:
    name = "base"

    def extract_fields(self, image_path: str) -> Dict[str, str]:
        raise NotImplementedError


class OllamaEngine(BaseEngine):
    def __init__(self, model: str = "qwen2.5vl:3b", host: Optional[str] = None):
        import os

        self.model = model
        self.host = (
            host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.name = f"ollama:{model}"

    def extract_fields(self, image_path: str) -> Dict[str, str]:
        import base64
        import urllib.error
        import urllib.request

        # Single source of truth (tests/test_prompt_parity.py enforces that the
        # phase-0 harness and this product path score under identical text).
        prompt = EXTRACT_PROMPT
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        # Client disconnects mid-generation leave the single-slot server wedged;
        # keep this generous (env-overridable) for slow local GPUs.
        timeout = float(os.environ.get("TINYDOC_OLLAMA_TIMEOUT", "180"))

        def _call(p: str) -> Dict[str, str]:
            # NOTE: do NOT set format="json" — Ollama constrained decoding
            # truncates long address fields and drops total on this model.
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": p, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 512},
            }
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
            raw = (body.get("message") or {}).get("content", "") or ""
            obj = _extract_json(raw) or {}
            return {k: str(obj.get(k, "")) for k in FIELDS}

        fields = _call(prompt)
        if not fields.get("total"):
            retry = (
                prompt
                + '\nRequired total is non-empty, e.g. "total":"12.34". JSON only.'
            )
            fields2 = _call(retry)
            if fields2.get("total"):
                return fields2
        return fields


class OcrRegexEngine(BaseEngine):
    name = "ocr_regex"

    def extract_fields(self, image_path: str) -> Dict[str, str]:
        try:
            import pytesseract
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(image_path))
        except Exception:
            return {k: "" for k in FIELDS}
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        date_m = re.search(
            r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}|\d{1,2}\s+\w+\s+\d{4})", text
        )
        total_m = re.search(
            r"(?:grand\s+total|total|amount\s+due)\s*[:\s]*\s*([\$]?\s*\d[\d,]*\.?\d*)",
            text,
            re.I,
        )
        if not total_m:
            ms = re.findall(r"\$?\s*\d[\d,]*\.\d{2}", text)
            total = ms[-1] if ms else ""
        else:
            total = total_m.group(1)
        return {
            "company": lines[0] if lines else "",
            "date": date_m.group(1) if date_m else "",
            "address": ", ".join(lines[1:3]),
            "total": total,
        }


class SmolVLM2Engine(BaseEngine):
    name = "smolvlm2"

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = str(model_path or SMOL_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(f"SmolVLM2 weights not on disk: {path}")
        self.device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.name = "smolvlm2_base"

    def extract_fields(self, image_path: str) -> Dict[str, str]:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        prompt = (
            'Extract receipt fields. Reply with JSON only, exactly four keys:\n'
            '{"company":"...","date":"...","address":"...","total":"..."}\n'
            "Fill every key from the image. Never omit total."
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[[img]], return_tensors="pt")
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
        n_in = inputs["input_ids"].shape[-1]
        gen = out[0][n_in:]
        tok = getattr(self.processor, "tokenizer", self.processor)
        raw = tok.decode(gen, skip_special_tokens=True).strip()
        obj = _extract_json(raw) or {}
        return {k: str(obj.get(k, "")) for k in FIELDS}


class RoutedEngine(BaseEngine):
    """Primary engine with OCR-floor fill for schema-breaking empty fields.

    Only runs the fallback when the primary result fails ``schema_is_valid``
    (required keys non-empty / maxLength). Never replaces non-empty primary
    values — avoids demoting a good VLM answer to a weak regex scrape.
    """

    def __init__(self, primary: BaseEngine, fallback: BaseEngine):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fb:{fallback.name}"

    def extract_fields(self, image_path: str) -> Dict[str, str]:
        fields = {k: str(v or "") for k, v in self.primary.extract_fields(image_path).items()}
        for k in FIELDS:
            fields.setdefault(k, "")
        if schema_is_valid(fields):
            return fields
        fb = {k: str(v or "") for k, v in self.fallback.extract_fields(image_path).items()}
        for k in FIELDS:
            if not fields.get(k) and fb.get(k):
                fields[k] = fb[k]
        return fields


def build_engine(name: str = "auto", **kwargs) -> BaseEngine:
    if name == "auto":
        # prefer measured best free engine; OCR floor only if schema would fail
        try:
            eng = OllamaEngine(**kwargs)
            # cheap ping
            import urllib.request

            urllib.request.urlopen(eng.host + "/api/tags", timeout=2)
            return RoutedEngine(eng, OcrRegexEngine())
        except Exception:
            return OcrRegexEngine()
    if name == "ollama" or name.startswith("ollama:"):
        model = kwargs.pop("model", "qwen2.5vl:3b")
        if name.startswith("ollama:"):
            model = name[len("ollama:") :]
        return OllamaEngine(model=model, **kwargs)
    if name == "ocr_regex":
        return OcrRegexEngine()
    if name == "smolvlm2":
        return SmolVLM2Engine(**kwargs)
    raise ValueError(f"unknown engine {name}")


def _field_confidence(name: str, value: str, ev: Optional[Evidence], schema_ok: bool) -> float:
    if not value:
        return 0.0
    base = 0.55 if schema_ok else 0.35
    if ev is not None:
        base = 0.35 + 0.55 * float(ev.score)
    # type priors
    if name == "total" and re.search(r"\d", value):
        base = min(1.0, base + 0.05)
    if name == "date" and re.search(r"\d", value):
        base = min(1.0, base + 0.05)
    if len(value) > 120 and name in ("company", "total", "date"):
        base *= 0.85
    return round(float(min(max(base, 0.0), 1.0)), 4)


class ReceiptPipeline:
    """Drop-folder → schema-validated fields + evidence + confidence."""

    def __init__(self, engine: Union[str, BaseEngine] = "auto", **engine_kwargs):
        if isinstance(engine, BaseEngine):
            self.engine = engine
        else:
            self.engine = build_engine(engine, **engine_kwargs)
        self.attach_evidence = True

    def extract(
        self,
        image_path: Union[str, Path],
        with_evidence: Optional[bool] = None,
    ) -> DocumentResult:
        image_path = str(image_path)
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)
        t0 = time.time()
        fields = self.engine.extract_fields(image_path)
        fields = sanitize_fields(fields)
        schema_ok = schema_is_valid(fields)

        do_ev = self.attach_evidence if with_evidence is None else with_evidence
        ev_map: Dict[str, Optional[Evidence]] = {}
        if do_ev:
            try:
                ev_map = attach_evidence(fields, image_path)
            except Exception:
                ev_map = {k: None for k in FIELDS}

        fr: List[FieldResult] = []
        confs = []
        n_present_ev = 0
        for k in FIELDS:
            ev = ev_map.get(k)
            conf = _field_confidence(k, fields[k], ev, schema_ok)
            present = bool(fields[k])
            if present and ev is not None:
                n_present_ev += 1
            fr.append(
                FieldResult(
                    name=k,
                    value=fields[k],
                    present=present,
                    evidence=ev.to_dict() if ev else None,
                    confidence=conf,
                )
            )
            if present:
                confs.append(conf)

        coverage = n_present_ev / len(FIELDS)
        # overall: schema gate × mean present-field confidence × evidence coverage bonus
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        overall = mean_conf * (0.7 + 0.3 * coverage)
        if not schema_ok:
            overall *= 0.75
        overall = round(float(min(max(overall, 0.0), 1.0)), 4)

        return DocumentResult(
            image_path=image_path,
            engine=self.engine.name,
            fields=fields,
            field_results=fr,
            schema_valid=schema_ok,
            confidence=overall,
            raw=json.dumps(fields),
            latency_ms=(time.time() - t0) * 1000,
            evidence_coverage=round(coverage, 4),
        )

    def extract_folder(
        self,
        folder: Union[str, Path],
        patterns: tuple = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.tif", "*.tiff"),
        limit: Optional[int] = None,
        with_evidence: Optional[bool] = None,
    ) -> List[DocumentResult]:
        folder = Path(folder)
        paths: List[Path] = []
        for pat in patterns:
            paths.extend(sorted(folder.glob(pat)))
            paths.extend(sorted(folder.glob("**/" + pat)))
        # unique preserve order
        seen = set()
        uniq = []
        for p in paths:
            rp = str(p)
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)
        if limit:
            uniq = uniq[:limit]
        results: List[DocumentResult] = []
        for p in uniq:
            try:
                results.append(self.extract(p, with_evidence=with_evidence))
            except Exception as e:
                # keep batch alive; surface failure as empty schema-invalid result
                results.append(
                    DocumentResult(
                        image_path=str(p),
                        engine=getattr(self.engine, "name", "unknown"),
                        fields={k: "" for k in FIELDS},
                        field_results=[
                            FieldResult(name=k, value="", present=False)
                            for k in FIELDS
                        ],
                        schema_valid=False,
                        confidence=0.0,
                        raw=f"ERROR: {e}",
                        latency_ms=0.0,
                        evidence_coverage=0.0,
                    )
                )
        return results
