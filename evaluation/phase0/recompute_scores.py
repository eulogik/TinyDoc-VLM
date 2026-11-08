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
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from metrics import macro_prf

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

# 2026-11-06 base bake-off. Optional: these artifacts may be absent from a
# source checkout, but when present every published number must recompute.
PILOT_PAIRS = [
    (
        "preds_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.jsonl",
        "scores_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.json",
    ),
    (
        "preds_pipeline_ollama_qwen3-vl_2b-instruct_clean_ctx8k.jsonl",
        "scores_pipeline_ollama_qwen3-vl_2b-instruct_clean_ctx8k.json",
    ),
    (
        "preds_pipeline_ollama_qwen2.5vl_3b_val_ctx8k.jsonl",
        "scores_pipeline_ollama_qwen2.5vl_3b_val_ctx8k.json",
    ),
    (
        "preds_pipeline_ollama_qwen3-vl_2b-instruct_val_ctx8k.jsonl",
        "scores_pipeline_ollama_qwen3-vl_2b-instruct_val_ctx8k.json",
    ),
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


def audit_pair(
    preds_name: str,
    scores_name: str,
    failures: list[str],
    missing_is_failure: bool = True,
) -> None:
    p_path, s_path = RESULTS / preds_name, RESULTS / scores_name
    if not p_path.exists() or not s_path.exists():
        if missing_is_failure:
            failures.append(f"{preds_name}: missing artifact")
            print(f"{preds_name:<56} {'--':>4} {'--':>7} {'--':>7} MISSING")
        else:
            print(f"{preds_name:<56} {'--':>4} {'--':>7} {'--':>7} absent (optional)")
        return

    rows = [json.loads(line) for line in p_path.open() if line.strip()]
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
    if "schema_valid_rate" in pub and not close(
        pub["schema_valid_rate"], re["schema_valid_rate"]
    ):
        ok = False
        reasons.append(f"schema {pub['schema_valid_rate']}!={re['schema_valid_rate']:.4f}")
    if "anls_mean" in pub and not close(pub["anls_mean"], re["anls_mean"]):
        ok = False
        reasons.append(f"anls {pub['anls_mean']}!={re['anls_mean']:.4f}")
    pub_hits = pub.get("per_field_hit_rate") or pub.get("field_hit_rates")
    if isinstance(pub_hits, dict):
        for field in FIELDS:
            if field in pub_hits and not close(pub_hits[field], re["per_field_hit_rate"][field]):
                ok = False
                reasons.append(
                    f"{field} {pub_hits[field]}!={re['per_field_hit_rate'][field]:.4f}"
                )

    status = "OK" if ok else "MISMATCH: " + "; ".join(reasons)
    if not ok:
        failures.append(f"{scores_name}: {status}")
    print(
        f"{preds_name:<56} {re['n']:>4} "
        f"{pub.get('field_f1', float('nan')):>7} {re['field_f1']:>7.4f} {status}"
    )


def main() -> int:
    failures: list[str] = []
    print(f"{'pair':<56} {'n':>4} {'F1 pub':>7} {'F1 re':>7} {'status'}")
    print("-" * 90)
    for preds_name, scores_name in PAIRS:
        audit_pair(preds_name, scores_name, failures)

    print("-" * 90)
    print("2026-11-06 base bake-off (optional artifacts)")
    for preds_name, scores_name in PILOT_PAIRS:
        audit_pair(preds_name, scores_name, failures, missing_is_failure=False)

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

    # Negative probe results are published claims too. Require the artifacts and
    # check the numbers quoted in README / docs/BENCHMARKS / docs/v3_plan.
    grounding_claims = [
        ("qwen2.5vl_3b", "grounding_probe_qwen2.5vl_3b.json", 0.041, 0.04),
        ("qwen3-vl_2b", "grounding_probe_qwen3-vl_2b-instruct.json", 0.308, 0.20),
    ]
    for label, name, mean_iou, hit in grounding_claims:
        path = RESULTS / name
        if not path.exists():
            failures.append(f"grounding probe artifact missing: {name}")
            continue
        g = json.loads(path.read_text())
        ok = (
            abs(float(g.get("mean_iou", -1)) - mean_iou) <= 5e-4
            and abs(float(g.get("hit_at_0.5", -1)) - hit) <= 1e-9
            and g.get("parse_rate") == 1.0
            and g.get("n") == 100
        )
        print(
            f"claim: grounding {label} mean IoU {g.get('mean_iou'):.3f}, "
            f"hit@0.5 {g.get('hit_at_0.5'):.2f} (gate 0.70) -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"grounding probe numbers changed for {label}")

    confidence_claims = [
        ("qwen2.5vl_3b", "confidence_probe_qwen2.5vl_3b_clean.json", 0.545),
        ("qwen3-vl_2b", "confidence_probe_qwen3-vl_2b-instruct_clean.json", 0.518),
    ]
    for label, name, ece_value in confidence_claims:
        path = RESULTS / name
        if not path.exists():
            failures.append(f"confidence probe artifact missing: {name}")
            continue
        c = json.loads(path.read_text())
        field = c.get("field_level") or {}
        ok = abs(float(field.get("ece_10bin", -1)) - ece_value) <= 5e-4 and c.get("n_docs") == 100
        print(
            f"claim: confidence {label} field ECE {field.get('ece_10bin'):.3f} "
            f"(uncalibrated) -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"confidence probe numbers changed for {label}")

    # Evidence-on default + document-grouped isotonic OOF. These are the numbers
    # the public docs now quote; they must match both the probe and calibration
    # artifacts.
    evidence_confidence_claims = [
        (
            "qwen2.5vl_3b",
            "confidence_probe_qwen2.5vl_3b_clean_evidence.json",
            "calibration_qwen2.5vl_3b_clean.json",
            0.1355,
            0.0140,
            0.0969,
            0.0603,
        ),
        (
            "qwen3-vl_2b",
            "confidence_probe_qwen3-vl_2b-instruct_clean_evidence.json",
            "calibration_qwen3-vl_2b-instruct_clean.json",
            0.1154,
            0.0049,
            0.0917,
            0.1064,
        ),
    ]
    for label, probe_name, cal_name, raw_field, iso_field, raw_doc, iso_doc in (
        evidence_confidence_claims
    ):
        probe_path, cal_path = RESULTS / probe_name, RESULTS / cal_name
        if not probe_path.exists() or not cal_path.exists():
            failures.append(f"evidence-on confidence artifacts missing for {label}")
            continue
        probe = json.loads(probe_path.read_text())
        cal = json.loads(cal_path.read_text())
        field = cal.get("fields") or {}
        doc = cal.get("doc_exact4") or {}
        ok = (
            probe.get("used_field_confidences") is True
            and abs(float(probe["field_level"]["ece_10bin"]) - raw_field) <= 5e-4
            and abs(float(field["raw"]["ece_10bin"]) - raw_field) <= 5e-4
            and abs(float(field["isotonic_oof"]["ece_10bin"]) - iso_field) <= 5e-4
            and abs(float(probe["doc_exact4"]["ece_10bin"]) - raw_doc) <= 5e-4
            and abs(float(doc["isotonic_oof"]["ece_10bin"]) - iso_doc) <= 5e-4
            and cal.get("n_docs") == 100
            and cal.get("folds") == 5
        )
        print(
            f"claim: confidence {label} evidence-on field ECE {raw_field:.3f} "
            f"-> isotonic OOF {iso_field:.3f}; doc {raw_doc:.3f} -> {iso_doc:.3f} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"evidence-on confidence numbers changed for {label}")

    invoice_summary = ROOT.parent / "invoice/results/summary_invoice_qwen2.5vl_3b.json"
    invoice_scores = ROOT.parent / "invoice/results/scores_invoice_qwen2.5vl_3b.json"
    if not invoice_summary.exists() or not invoice_scores.exists():
        failures.append("invoice probe artifacts missing")
    else:
        inv = json.loads(invoice_summary.read_text())
        inv_scores_data = json.loads(invoice_scores.read_text())
        key = inv_scores_data.get("summary", {}).get("key_fields", {})
        ok = (
            inv.get("n_docs") == 181
            and abs(float(inv.get("key_header_accuracy", -1)) - 0.918782) <= 1e-5
            and abs(float(inv.get("fully_correct_rate", -1)) - 0.657459) <= 1e-5
            and key.get("correct") == 1448
            and key.get("scored") == 1576
        )
        print(
            f"claim: invoice synthetic probe key-header {inv.get('key_header_accuracy'):.4f}, "
            f"fully correct {inv.get('fully_correct_rate'):.4f}, n={inv.get('n_docs')} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("invoice probe numbers changed")

    invoice_summary_2b = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_instruct.json"
    invoice_compare = ROOT.parent / "invoice/results/compare_invoice_3b_vs_2b.json"
    if not invoice_summary_2b.exists() or not invoice_compare.exists():
        failures.append("2B invoice probe artifacts missing")
    else:
        inv2 = json.loads(invoice_summary_2b.read_text())
        cmp_data = json.loads(invoice_compare.read_text())
        ok = (
            inv2.get("n_docs") == 181
            and abs(float(inv2.get("key_header_accuracy", -1)) - 0.935279) <= 1e-5
            and abs(float(inv2.get("fully_correct_rate", -1)) - 0.718232) <= 1e-5
            and inv2.get("key_header_correct") == 1474
            and inv2.get("fully_correct") == 130
            and abs(cmp_data["key"]["diff"] - 0.016497) <= 1e-5
            and abs(cmp_data["fully_correct"]["diff"] - 0.060773) <= 1e-5
            and abs(cmp_data["key"]["sign_p"] - 0.091915) <= 1e-5
        )
        print(
            f"claim: invoice 2B key-header {inv2.get('key_header_accuracy'):.4f}, "
            f"fully correct {inv2.get('fully_correct_rate'):.4f}; paired diff "
            f"{cmp_data['key']['diff']:+.4f} (p={cmp_data['key']['sign_p']:.3f}) "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("2B invoice probe or tier comparison changed")

    multipage_summary = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_mp.json"
    multipage_scores = ROOT.parent / "invoice/results/scores_invoice_qwen3-vl_2b_mp.json"
    multipage_compare = ROOT.parent / "invoice/results/compare_invoice_page1_vs_multipage_2b.json"
    multipage_rows = ROOT.parent / "invoice/results/rows_invoice_qwen3-vl_2b_mp.jsonl"
    multipage_missing = [
        path.name
        for path in (multipage_summary, multipage_scores, multipage_compare, multipage_rows)
        if not path.exists()
    ]
    if multipage_missing:
        failures.append(f"multipage artifacts missing: {multipage_missing}")
    else:
        try:
            msum = json.loads(multipage_summary.read_text())
            msc = json.loads(multipage_scores.read_text())
            mcmp = json.loads(multipage_compare.read_text())
            mrows = [json.loads(line) for line in multipage_rows.open() if line.strip()]
            invoice_key_fields = (
                "vendor_name", "invoice_number", "invoice_date", "due_date",
                "currency", "subtotal", "tax", "total", "document_type",
            )
            key_correct = sum(
                1
                for doc in msc["documents"]
                for name in invoice_key_fields
                if (doc.get("fields", {}).get(name) or {}).get("correct") is True
            )
            key_scored = sum(
                1
                for doc in msc["documents"]
                for name in invoice_key_fields
                if name in (doc.get("fields") or {})
            )
            fully = sum(1 for doc in msc["documents"] if doc.get("fully_correct") is True)
            failed_ids = {row["id"] for row in mrows if row.get("error")}
            silent_cuts = [
                row["id"]
                for row in mrows
                if not row.get("error") and row.get("done_reason") != "stop"
            ]
            overflow = [
                row["id"]
                for row in mrows
                if not row.get("error")
                and (row.get("prompt_eval_count") or 0) + (row.get("eval_count") or 0)
                > (row.get("num_ctx") or 16384)
            ]
            expected_failures = {"multipage-8p-us-9", "multipage-8p-na-10"}
            ok = (
                msum.get("n_docs") == 10
                and abs(float(msum.get("key_header_accuracy", -1)) - 0.711111) <= 1e-5
                and abs(float(msum.get("fully_correct_rate", -1)) - 0.4) <= 1e-5
                and msum.get("parse_errors") == 2
                and (key_correct, key_scored, fully) == (64, 90, 4)
                and (msc["summary"]["key_fields"]["correct"], msc["summary"]["key_fields"]["scored"]) == (64, 90)
                and failed_ids == expected_failures
                and not silent_cuts
                and not overflow
                and mcmp["page1"]["key_correct"] == 57
                and mcmp["page1"]["fully_correct"] == 0
                and mcmp["all_pages"]["key_correct"] == 64
                and mcmp["all_pages"]["fully_correct"] == 4
                and abs(float(mcmp["transitions"]["key_sign_p"]) - 0.2890625) <= 1e-9
                and abs(float(mcmp["transitions"]["fully_correct_sign_p"]) - 0.125) <= 1e-9
            )
            print(
                f"claim: multipage 2B all pages key-header {key_correct}/{key_scored} "
                f"({msum.get('key_header_accuracy'):.4f}), fully correct {fully}/10; "
                f"page 1 {mcmp['page1']['key_correct']}/90, {mcmp['page1']['fully_correct']}/10; "
                f"failed rows {sorted(failed_ids)} -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("multipage probe numbers or failure accounting changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"multipage check errored: {exc}")

    # Paired hybrid-vs-base counts on the shipped product path. The earlier
    # 31/8/61 came from the harness path and must not be quoted as shipped.
    product = RESULTS / "preds_product_hybrid_clean.jsonl"
    base = RESULTS / "preds_smolvlm2-2.2b_base_clean.jsonl"
    decision = RESULTS / "v2_product_decision.json"
    if product.exists() and base.exists():
        try:
            prod = {json.loads(line)["id"]: json.loads(line) for line in product.open() if line.strip()}
            bas = {json.loads(line)["id"]: json.loads(line) for line in base.open() if line.strip()}
            ids = sorted(set(prod) & set(bas))
            aw = al = at = 0
            fw = fl = ft = 0
            for doc_id in ids:
                pa = bool(prod[doc_id]["score"]["fields"]["address"]["hit"])
                ba = bool(bas[doc_id]["score"]["fields"]["address"]["hit"])
                if pa and not ba:
                    aw += 1
                elif ba and not pa:
                    al += 1
                else:
                    at += 1
                pf = float(prod[doc_id]["score"]["f1"])
                bf = float(bas[doc_id]["score"]["f1"])
                if pf > bf + 1e-9:
                    fw += 1
                elif bf > pf + 1e-9:
                    fl += 1
                else:
                    ft += 1
            n = aw + al
            p = min(1.0, 2 * sum(comb(n, k) for k in range(min(aw, al) + 1)) / 2 ** n) if n else 1.0
            ok = (aw, al, at) == (31, 7, 62) and (fw, fl, ft) == (32, 7, 61)
            print(
                f"claim: hybrid address paired {aw}/{al}/{at}, p={p:.6f}; "
                f"per-doc F1 {fw}/{fl}/{ft} -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(
                    f"paired hybrid counts changed: address {aw}/{al}/{at}, F1 {fw}/{fl}/{ft}"
                )
            if decision.exists():
                d = json.loads(decision.read_text())
                if abs(float(d.get("address_sign_test_p", -1)) - p) > 1e-5:
                    failures.append(
                        f"v2_product_decision.json p={d.get('address_sign_test_p')} != recomputed {p:.6f}"
                    )
        except Exception as e:  # noqa: BLE001
            failures.append(f"paired hybrid check errored: {e}")

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
