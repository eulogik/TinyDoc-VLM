"""Phase-0 metrics: ANLS, field-level P/R/F1, schema-valid rate."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_PATH = Path(__file__).parent / "schemas" / "sroie_receipt.schema.json"
FIELDS = ("company", "date", "address", "total")


def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]  # dp[i-1][0]
        dp[0] = i
        for j in range(1, n + 1):
            old_diag = prev  # dp[i-1][j-1]
            prev = dp[j]  # becomes dp[i-1][j] after this assignment for next j
            if s1[i - 1] == s2[j - 1]:
                dp[j] = old_diag
            else:
                dp[j] = 1 + min(old_diag, prev, dp[j - 1])
    return dp[n]


def anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA-style, thr 0.5)."""
    if not prediction or not ground_truth:
        return 0.0
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    if not pred and not gt:
        return 1.0
    dist = levenshtein(pred, gt)
    nl = 1.0 - dist / max(len(pred), len(gt), 1)
    return nl if nl >= threshold else 0.0


def normalize_field(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.replace("\n", " ")
    s = re.sub(r"[\$£€]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:\"'`!?]+$", "", s)
    return s.strip()


def normalize_text(s: Any) -> str:
    """Alphanumeric-only normalize for company/address (matches evidence.py).

    Addresses differ by OCR punctuation (comma vs period, spacing around
    '&', missing trailing comma). Content-equal strings must match; date
    and money keep type-aware rules below.
    """
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("\n", " ")
    s = re.sub(r"[\$£€]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def address_metrics(prediction: Any, ground_truth: Any) -> Dict[str, float]:
    pred = str(prediction or "").strip()
    gold = str(ground_truth or "").strip()
    if not pred and not gold:
        raw_cer = 0.0
    elif not pred or not gold:
        raw_cer = 1.0
    else:
        raw_cer = levenshtein(pred, gold) / max(len(pred), len(gold))
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        normalized_f1 = 1.0
    elif not pred_tokens or not gold_tokens:
        normalized_f1 = 0.0
    else:
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        normalized_f1 = 2.0 * overlap / (len(pred_tokens) + len(gold_tokens))
    return {
        "raw_cer": raw_cer,
        "normalized_f1": normalized_f1,
        "raw_exact": float(pred == gold),
    }


def field_match(pred: str, gold: str, kind: str) -> bool:
    """Binary match used for P/R: exact after normalize, or gold ⊆ pred for text,
    or numeric equality for money/date-ish fields.

    company/address use normalize_text (punctuation-insensitive content match).
    """
    p, g = normalize_field(pred), normalize_field(gold)
    if not g:
        return not p
    if not p:
        return False
    if kind in ("company", "address"):
        pt, gt = normalize_text(pred), normalize_text(gold)
        if not gt:
            return not pt
        if not pt:
            return False
        return pt == gt or gt in pt or pt in gt
    if p == g:
        return True
    if kind == "total":
        return _money_eq(p, g)
    if kind == "date":
        return _date_eq(p, g)
    # fallback: containment either way after normalize
    return g in p or p in g


def _money_eq(p: str, g: str) -> bool:
    def nums(s: str) -> List[str]:
        return re.findall(r"\d+(?:[.,]\d+)?", s.replace(",", ""))

    pn, gn = nums(p), nums(g)
    if pn and gn:
        try:
            return abs(float(pn[-1].replace(",", ".")) - float(gn[-1].replace(",", "."))) < 0.011
        except ValueError:
            pass
    return p == g


def _date_eq(p: str, g: str) -> bool:
    def parts(s: str) -> Tuple:
        d = [int(x) for x in re.findall(r"\d+", s)[:3]]
        return tuple(d)

    if p == g:
        return True
    # same three numeric components regardless of separators/order ambiguity:
    # only accept if the sets of numbers match (day, month, year)
    pn, gn = parts(p), parts(g)
    if len(pn) == 3 and len(gn) == 3 and sorted(pn) == sorted(gn):
        return True
    return g in p or p in g


def _json_loads_maybe(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _salvage_truncated_object(s: str) -> Optional[Dict[str, Any]]:
    """Close unclosed strings/braces and drop trailing incomplete fields."""
    # close unclosed string
    in_str = esc = False
    for ch in s:
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
        s = s + '"'
    # repeatedly drop trailing incomplete fragments and close braces
    for _ in range(8):
        body = s.rstrip()
        # strip trailing comma / colon
        body = re.sub(r"[,:]\s*$", "", body)
        # strip dangling bare key (", \"foo\"" or ", \"foo\":")
        body = re.sub(r",\s*\"(?:[^\"\\]|\\.)*\"\s*:?\s*$", "", body)
        # count braces outside strings
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
        obj = _json_loads_maybe(cand)
        if obj is not None:
            return obj
        # if still failing, cut back to last comma boundary
        m = re.search(r",", body)
        if not m:
            break
        # cut last key-value attempt
        cut = body.rfind(",")
        if cut <= 0:
            break
        s = body[:cut]
    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
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
        obj = _json_loads_maybe(t[start : end + 1])
        if obj is not None:
            return obj
    # truncated (or nested failure): salvage from first brace to end of text
    return _salvage_truncated_object(t[start:])


def schema_valid(obj: Optional[Dict[str, Any]]) -> bool:
    if obj is None:
        return False
    try:
        import jsonschema
    except ImportError:
        # fallback: required keys + types
        return all(k in obj and isinstance(obj[k], str) and obj[k] for k in FIELDS)
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        jsonschema.validate(instance=obj, schema=schema)
        return True
    except jsonschema.ValidationError:
        return False


def score_example(pred_fields: Dict[str, Any], gold_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Per-field hit/miss + ANLS. pred may be partial/invalid."""
    per_field = {}
    for f in FIELDS:
        pv = pred_fields.get(f) if isinstance(pred_fields, dict) else None
        gv = gold_fields.get(f, "")
        hit = field_match(str(pv) if pv is not None else "", str(gv), f)
        per_field[f] = {
            "hit": hit,
            "pred": pv,
            "gold": gv,
            "anls": anls(str(pv) if pv is not None else "", str(gv)),
        }
    hits = sum(1 for v in per_field.values() if v["hit"])
    return {
        "fields": per_field,
        "precision": hits / len(FIELDS),  # micro over present fields treated uniformly
        "recall": hits / len(FIELDS),
        "f1": hits / len(FIELDS),
        "anls_mean": sum(v["anls"] for v in per_field.values()) / len(FIELDS),
        "schema_valid": schema_valid(pred_fields if isinstance(pred_fields, dict) else None),
    }


def macro_prf(example_scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """Micro P/R/F1 over all (example, field) pairs where gold field is non-empty."""
    tp = fp = fn = 0
    for ex in example_scores:
        for f, v in ex["fields"].items():
            gold_nonempty = normalize_field(v["gold"]) != ""
            pred_nonempty = normalize_field(str(v["pred"] if v["pred"] is not None else "")) != ""
            if v["hit"]:
                tp += 1
            else:
                if pred_nonempty and gold_nonempty:
                    fp += 1
                    fn += 1  # wrong prediction counts as both miss on gold and spurious
                elif gold_nonempty and not pred_nonempty:
                    fn += 1
                elif pred_nonempty and not gold_nonempty:
                    fp += 1
    # For KIE field F1, simpler correct definition: per field slot
    # Recompute cleanly: each (ex, field) is one decision
    tp = fp = fn = 0
    for ex in example_scores:
        for f, v in ex["fields"].items():
            if v["hit"]:
                tp += 1
            else:
                pred_empty = normalize_field(str(v["pred"] if v["pred"] is not None else "")) == ""
                gold_empty = normalize_field(v["gold"]) == ""
                if not gold_empty and pred_empty:
                    fn += 1
                elif not gold_empty and not pred_empty:
                    fn += 1  # missed correct value
                    fp += 1  # emitted wrong value
                elif gold_empty and not pred_empty:
                    fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = len(example_scores) or 1
    return {
        "n_examples": len(example_scores),
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "anls_mean": sum(e["anls_mean"] for e in example_scores) / n,
        "schema_valid_rate": sum(1 for e in example_scores if e["schema_valid"]) / n,
        "exact_json_all_fields": sum(
            1
            for e in example_scores
            if all(e["fields"][f]["hit"] for f in FIELDS)
        )
        / n,
    }


def normalize_gold_target(target: str) -> Dict[str, Any]:
    try:
        obj = json.loads(target)
        if isinstance(obj, dict):
            return {k: str(obj.get(k, "")) for k in FIELDS}
    except json.JSONDecodeError:
        pass
    return {k: "" for k in FIELDS}
