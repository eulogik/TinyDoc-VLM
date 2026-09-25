"""Recompute every published score from committed per-example artifacts.

No model calls, no OCR, no network. For each (preds jsonl, scores json) pair in
PAIRS, rebuild macro field F1 / schema rate / ANLS / per-field hit rates from the
per-row `score` blocks and assert they match the published aggregate (tolerance
covers the 2-decimal rounding used in some published files).

Exit 0 = every claim recomputes. This is the G8 claim-audit gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from metrics import macro_prf  # noqa: E402

RESULTS = ROOT / "results"

PAIRS = [
    ("preds_ollama.jsonl", "scores_ollama.json"),
    ("preds_ollama_unseen.jsonl", "scores_ollama_unseen.json"),
    ("preds_funsd_ollama.jsonl", "scores_funsd_ollama.json"),
    ("preds_ocr_regex.jsonl", "scores_ocr_regex.json"),
    ("preds_ocr_regex_unseen.jsonl", "scores_ocr_regex_unseen.json"),
    ("preds_smolvlm2.jsonl", "scores_smolvlm2.json"),
    ("preds_ppocr_heuristics.jsonl", "scores_ppocr_heuristics.json"),
    ("preds_pipeline_ollama_qwen2.5vl_3b.jsonl", "scores_pipeline_ollama_qwen2.5vl_3b.json"),
    ("preds_pipeline_ollama_qwen2.5vl_3b_unseen.jsonl", "scores_pipeline_ollama_qwen2.5vl_3b_unseen.json"),
]

TOL = 0.006  # published aggregates are sometimes rounded to 2 decimals
FIELDS = ("company", "date", "address", "total")


def recompute(rows: list[dict]) -> dict:
    scores = [r["score"] for r in rows]
    agg = macro_prf(scores)
    out = {
        "n": len(rows),
        "field_f1": agg["field_f1"],
        "schema_valid_rate": sum(1 for s in scores if s.get("schema_valid")) / len(scores),
        "anls_mean": sum(s.get("anls_mean", 0.0) for s in scores) / len(scores),
        "per_field_hit_rate": {
            f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores)
            for f in FIELDS
        },
    }
    return out


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= TOL


def main() -> int:
    failures: list[str] = []
    print(f"{'pair':<56} {'n':>4} {'F1 pub':>7} {'F1 re':>7} {'status'}")
    print("-" * 90)
    for preds_name, scores_name in PAIRS:
        p_path, s_path = RESULTS / preds_name, RESULTS / scores_name
        if not p_path.exists() or not s_path.exists():
            failures.append(f"{preds_name}: missing artifact")
            print(f"{preds_name:<56} {'--':>4} {'--':>7} {'--':>7} MISSING")
            continue
        rows = [json.loads(l) for l in p_path.open() if l.strip()]
        pub = json.loads(s_path.read_text())
        re = recompute(rows)

        ok = True
        reasons = []
        if int(pub.get("n_examples", re["n"])) != re["n"]:
            ok = False
            reasons.append(f"n {pub.get('n_examples')}!={re['n']}")
        if "field_f1" in pub and not close(pub["field_f1"], re["field_f1"]):
            ok = False
            reasons.append(f"f1 {pub['field_f1']}!={re['field_f1']:.4f}")
        if "schema_valid_rate" in pub and not close(pub["schema_valid_rate"], re["schema_valid_rate"]):
            ok = False
            reasons.append(f"schema {pub['schema_valid_rate']}!={re['schema_valid_rate']:.4f}")
        if "anls_mean" in pub and not close(pub["anls_mean"], re["anls_mean"]):
            ok = False
            reasons.append(f"anls {pub['anls_mean']}!={re['anls_mean']:.4f}")
        pub_hits = pub.get("per_field_hit_rate") or pub.get("field_hit_rates")
        if isinstance(pub_hits, dict):
            for f in FIELDS:
                if f in pub_hits and not close(pub_hits[f], re["per_field_hit_rate"][f]):
                    ok = False
                    reasons.append(f"{f} {pub_hits[f]}!={re['per_field_hit_rate'][f]:.4f}")

        status = "OK" if ok else "MISMATCH: " + "; ".join(reasons)
        if not ok:
            failures.append(f"{scores_name}: {status}")
        print(
            f"{preds_name:<56} {re['n']:>4} "
            f"{pub.get('field_f1', float('nan')):>7} {re['field_f1']:>7.4f} {status}"
        )

    # product-level claims that must hold across pairs
    print("-" * 90)
    def f1_of(scores_name: str) -> float:
        return json.loads((RESULTS / scores_name).read_text())["field_f1"]

    try:
        ours, comp = f1_of("scores_ollama.json"), f1_of("scores_ppocr_heuristics.json")
        win = ours > comp
        print(f"claim: pipeline {ours:.3f} > ppocr_baseline {comp:.3f} -> {'PASS' if win else 'FAIL'}")
        if not win:
            failures.append("competitive claim: pipeline does not beat PP-OCR baseline")
    except Exception as e:  # noqa: BLE001
        failures.append(f"competitive claim check errored: {e}")

    ev = RESULTS / "evidence_ab.json"
    if ev.exists():
        e = json.loads(ev.read_text())["summary"]
        gain = e["rapidocr"]["gold_in_quote_rate"] > e["tesseract"]["gold_in_quote_rate"]
        print(f"claim: evidence rapidocr gold_in_quote {e['rapidocr']['gold_in_quote_rate']} > "
              f"tesseract {e['tesseract']['gold_in_quote_rate']} -> {'PASS' if gain else 'FAIL'}")
        if not gain:
            failures.append("evidence A/B: adopted engine does not win")

    ocr = RESULTS / "ocrbench_256m_full.json"
    if ocr.exists():
        o = json.loads(ocr.read_text())
        acc = o.get("ocrbench_score_strict")
        print(f"claim: ocrbench_256m_full strict accuracy == {acc} -> "
              f"{'PASS' if acc == 0.0 else 'FAIL (expected 0.0)'}")
        if acc != 0.0:
            failures.append(f"ocrbench artifact accuracy {acc} != 0.0")

    print("-" * 90)
    if failures:
        print(f"CLAIM AUDIT FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("CLAIM AUDIT PASSED — every published number recomputes from artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
