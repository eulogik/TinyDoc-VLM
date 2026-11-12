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
sys.path.insert(0, str(ROOT.parent / "docvqa"))

from metrics import macro_prf
from run_docvqa_probe import anls_score, normalize_answer

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

    docvqa_rows = ROOT.parent / "docvqa/results/rows_docvqa_qwen3-vl_2b_instruct.jsonl"
    docvqa_summary = ROOT.parent / "docvqa/results/summary_docvqa_qwen3-vl_2b_instruct.json"
    if not docvqa_rows.exists() or not docvqa_summary.exists():
        failures.append("DocVQA probe artifacts missing")
    else:
        try:
            rows = [json.loads(line) for line in docvqa_rows.open() if line.strip()]
            summary = json.loads(docvqa_summary.read_text())
            anls = sum(anls_score(row["answer"], row["target"]) for row in rows) / len(rows)
            exact = sum(
                1
                for row in rows
                if normalize_answer(row["answer"]) == normalize_answer(row["target"])
            ) / len(rows)
            ok = (
                summary.get("n") == 200
                and abs(float(summary.get("anls_mean", -1)) - 0.829397) <= 1e-5
                and abs(float(summary.get("exact_rate", -1)) - 0.765) <= 1e-9
                and summary.get("errors") == 0
                and summary.get("stored_anls_mismatches") == 0
                and summary.get("single_reference") is True
                and abs(anls - 0.8293969454993905) <= 1e-9
                and abs(exact - 0.765) <= 1e-9
            )
            print(
                f"claim: DocVQA reading probe (single-reference, n={len(rows)}) "
                f"ANLS {anls:.4f}, exact {exact:.4f}, errors {summary.get('errors')} "
                f"-> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("DocVQA probe numbers changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"DocVQA check errored: {exc}")

    pagewise_summary = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise.json"
    pagewise_scores = ROOT.parent / "invoice/results/scores_invoice_qwen3-vl_2b_mp_pagewise.json"
    pagewise_compare = ROOT.parent / "invoice/results/compare_invoice_mp_allpages_vs_pagewise_2b.json"
    pagewise_rows = ROOT.parent / "invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise.jsonl"
    pagewise_raw_rows = ROOT.parent / "invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise_raw.jsonl"
    pagewise_audit = ROOT.parent / "invoice/results/audit_invoice_qwen3-vl_2b_mp_pagewise.json"
    pagewise_rebuild = ROOT.parent / "invoice/results/rebuild_invoice_qwen3-vl_2b_mp_pagewise.json"
    pagewise_missing = [
        path.name
        for path in (
            pagewise_summary,
            pagewise_scores,
            pagewise_compare,
            pagewise_rows,
            pagewise_raw_rows,
            pagewise_audit,
            pagewise_rebuild,
        )
        if not path.exists()
    ]
    if pagewise_missing:
        failures.append(f"pagewise artifacts missing: {pagewise_missing}")
    else:
        try:
            psum = json.loads(pagewise_summary.read_text())
            psc = json.loads(pagewise_scores.read_text())
            pcmp = json.loads(pagewise_compare.read_text())
            prows = [json.loads(line) for line in pagewise_rows.open() if line.strip()]
            praw = [json.loads(line) for line in pagewise_raw_rows.open() if line.strip()]
            paudit = json.loads(pagewise_audit.read_text())
            prebuild = json.loads(pagewise_rebuild.read_text())
            invoice_key_fields = (
                "vendor_name", "invoice_number", "invoice_date", "due_date",
                "currency", "subtotal", "tax", "total", "document_type",
            )
            key_correct = sum(
                1
                for doc in psc["documents"]
                for name in invoice_key_fields
                if (doc.get("fields", {}).get(name) or {}).get("correct") is True
            )
            key_scored = sum(
                1
                for doc in psc["documents"]
                for name in invoice_key_fields
                if name in (doc.get("fields") or {})
            )
            fully = sum(1 for doc in psc["documents"] if doc.get("fully_correct") is True)
            page_count = sum(len(row.get("pages") or []) for row in prows)
            silent_cuts = [
                row["id"]
                for row in prows
                for page in row.get("pages") or []
                if page.get("done_reason") != "stop"
            ]
            overflow = [
                f"{row['id']} p{page.get('page')}"
                for row in prows
                for page in row.get("pages") or []
                if (page.get("prompt_eval_count") or 0) + (page.get("eval_count") or 0) > 8192
            ]
            corrections = {
                row["id"]: (row.get("pred") or {}).get("corrections")
                for row in prows
                if (row.get("pred") or {}).get("corrections")
            }
            ok = (
                psum.get("n_docs") == 10
                and psum.get("parse_errors") == 0
                and abs(float(psum.get("key_header_accuracy", -1)) - 0.933333) <= 1e-5
                and abs(float(psum.get("fully_correct_rate", -1)) - 0.6) <= 1e-9
                and abs(float(psum.get("line_items", {}).get("f1_mean", -1)) - 0.848) <= 1e-9
                and (key_correct, key_scored, fully) == (84, 90, 6)
                and psc["summary"]["key_fields"] == {"correct": 84, "scored": 90}
                and page_count == 46
                and len(prows) == 10
                and len(praw) == 10
                and all(row.get("error") is None for row in prows)
                and all(row.get("error") is None for row in praw)
                and not silent_cuts
                and not overflow
                and corrections
                == {
                    "multipage-3p-na-3": ["subtotal_from_total_tax_residual"],
                    "multipage-6p-za-8": ["tax_from_total_subtotal_truncated"],
                }
                and prebuild.get("corrections") == 2
                and prebuild.get("review_flags") == 8
                and paudit.get("ok") is True
                and pcmp["before"]["all_pages"]["key_correct"] == 64
                and pcmp["before"]["all_pages"]["fully_correct"] == 4
                and pcmp["after"]["pagewise"]["key_correct"] == 84
                and pcmp["after"]["pagewise"]["fully_correct"] == 6
                and abs(float(pcmp["after"]["pagewise"]["mean_line_f1"]) - 0.848) <= 1e-9
                and abs(float(pcmp["delta"]["key_accuracy"]) - 0.222222) <= 1e-5
                and abs(float(pcmp["delta"]["fully_correct_rate"]) - 0.2) <= 1e-9
                and abs(float(pcmp["delta"]["mean_line_f1"]) - 0.526) <= 1e-9
                and abs(float(pcmp["transitions"]["key_sign_p"]) - 0.21875) <= 1e-9
                and abs(float(pcmp["transitions"]["fully_correct_sign_p"]) - 0.625) <= 1e-9
            )
            print(
                f"claim: pagewise multipage 2B key-header {key_correct}/{key_scored} "
                f"({psum.get('key_header_accuracy'):.4f}), fully correct {fully}/10, "
                f"line F1 {psum.get('line_items', {}).get('f1_mean'):.3f}; "
                f"all-pages {pcmp['before']['all_pages']['key_correct']}/90, "
                f"{pcmp['before']['all_pages']['fully_correct']}/10; "
                f"corrections {len(corrections)} -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("pagewise probe numbers or corrections changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"pagewise check errored: {exc}")

    ocr_summary = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise_ocr.json"
    ocr_scores = ROOT.parent / "invoice/results/scores_invoice_qwen3-vl_2b_mp_pagewise_ocr.json"
    ocr_rows = ROOT.parent / "invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise_ocr.jsonl"
    ocr_compare_pagewise = ROOT.parent / "invoice/results/compare_invoice_pagewise_vs_ocr_2b.json"
    ocr_compare_allpages = ROOT.parent / "invoice/results/compare_invoice_mp_allpages_vs_ocr_2b.json"
    ocr_audit = ROOT.parent / "invoice/results/audit_invoice_qwen3-vl_2b_mp_pagewise_ocr.json"
    ocr_rebuild = ROOT.parent / "invoice/results/rebuild_invoice_qwen3-vl_2b_mp_pagewise_ocr.json"
    reconciled_summary = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise_reconciled.json"
    promptv2_raw_summary = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise_v2_raw.json"
    promptv2_summary = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise_v2.json"
    ocr_missing = [
        path.name
        for path in (
            ocr_summary, ocr_scores, ocr_rows, ocr_compare_pagewise,
            ocr_compare_allpages, ocr_audit, ocr_rebuild, reconciled_summary,
            promptv2_raw_summary, promptv2_summary,
        )
        if not path.exists()
    ]
    if ocr_missing:
        failures.append(f"pagewise OCR artifacts missing: {ocr_missing}")
    else:
        try:
            osum = json.loads(ocr_summary.read_text())
            osc = json.loads(ocr_scores.read_text())
            orows = [json.loads(line) for line in ocr_rows.open() if line.strip()]
            ocp = json.loads(ocr_compare_pagewise.read_text())
            oca = json.loads(ocr_compare_allpages.read_text())
            oaudit = json.loads(ocr_audit.read_text())
            orebuild = json.loads(ocr_rebuild.read_text())
            rsum = json.loads(reconciled_summary.read_text())
            v2raw = json.loads(promptv2_raw_summary.read_text())
            v2sum = json.loads(promptv2_summary.read_text())
            symbols = {
                "N$": "NAD", "NAD": "NAD", "R": "ZAR", "ZAR": "ZAR",
                "$": "USD", "USD": "USD", "EUR": "EUR", "£": "GBP",
                "GBP": "GBP", "K": "ZMW", "ZMW": "ZMW",
            }
            ocr_pages = sum(len(row.get("pages") or []) for row in orows)
            ocr_blocks = [
                row["ocr_totals"]
                for row in orows
                if (row.get("ocr_totals") or {}).get("consistent") is True
            ]
            amount_mismatches = []
            currency_mismatches = []
            for row in orows:
                pred = row.get("pred") or {}
                block = row.get("ocr_totals") or {}
                for name, key in (("subtotal", "subtotal"), ("tax", "tax"), ("total", "total_amount")):
                    if abs(float(pred.get(key)) - float(block.get(name))) > 0.005:
                        amount_mismatches.append(f"{row['id']}:{key}")
                expected = symbols.get(str(block.get("currency_symbol", "")).strip().upper())
                if expected is not None and pred.get("currency") != expected:
                    currency_mismatches.append(f"{row['id']}:{pred.get('currency')}!={expected}")
            corrections = {
                row["id"]: (row.get("pred") or {}).get("corrections")
                for row in orows
                if (row.get("pred") or {}).get("corrections")
            }
            ok = (
                osum.get("n_docs") == 10
                and osum.get("parse_errors") == 0
                and osum.get("key_header_correct") == 90
                and osum.get("key_header_scored") == 90
                and abs(float(osum.get("key_header_accuracy", -1)) - 1.0) <= 1e-9
                and osum.get("fully_correct") == 10
                and abs(float(osum.get("fully_correct_rate", -1)) - 1.0) <= 1e-9
                and abs(float(osum.get("line_items", {}).get("f1_mean", -1)) - 0.848) <= 1e-9
                and osum.get("line_items", {}).get("all_lines_right") == 1
                and abs(osum["ci95_exact"]["fully_correct_rate"][0] - 0.6915) < 1e-3
                and osc["summary"]["key_fields"] == {"correct": 90, "scored": 90}
                and osc["summary"]["fully_correct"] == 10
                and len(orows) == 10
                and ocr_pages == 46
                and all(row.get("error") is None for row in orows)
                and all(page.get("done_reason") == "stop" for row in orows for page in row["pages"])
                and len(ocr_blocks) == 10
                and not amount_mismatches
                and not currency_mismatches
                and corrections == {
                    "multipage-2p-za-1": ["totals_from_ocr:subtotal+tax+total"],
                    "multipage-3p-na-3": [
                        "subtotal_from_total_tax_residual",
                        "currency_from_ocr_totals",
                    ],
                    "multipage-6p-za-8": ["tax_from_total_subtotal_truncated"],
                }
                and oaudit.get("ok") is True
                and orebuild.get("ocr_totals_consistent") == 10
                and orebuild.get("corrections") == 3
                and orebuild.get("review_flags") == 8
                and rsum.get("key_header_correct") == 86
                and rsum.get("fully_correct") == 8
                and abs(float(rsum.get("line_items", {}).get("f1_mean", -1)) - 0.848) <= 1e-9
                and v2raw.get("key_header_correct") == 81
                and v2raw.get("fully_correct") == 5
                and abs(float(v2raw.get("line_items", {}).get("f1_mean", -1)) - 0.844) <= 1e-9
                and v2sum.get("key_header_correct") == 90
                and v2sum.get("fully_correct") == 10
                and abs(float(ocp["transitions"]["key_sign_p"]) - 0.125) <= 1e-9
                and abs(float(ocp["transitions"]["fully_correct_sign_p"]) - 0.125) <= 1e-9
                and abs(float(oca["transitions"]["key_sign_p"]) - 0.03125) <= 1e-9
                and abs(float(oca["transitions"]["fully_correct_sign_p"]) - 0.03125) <= 1e-9
            )
            print(
                f"claim: pagewise+OCR 2B key-header {osum.get('key_header_correct')}/90 "
                f"({osum.get('key_header_accuracy'):.4f}), fully correct "
                f"{osum.get('fully_correct')}/10, line F1 "
                f"{osum.get('line_items', {}).get('f1_mean'):.3f}; OCR blocks "
                f"{len(ocr_blocks)}/10, corrections {len(corrections)}; "
                f"prompt v2 raw {v2raw.get('key_header_correct')}/90 "
                f"{v2raw.get('fully_correct')}/10 -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("pagewise OCR numbers, corrections, or prompt-v2 comparison changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"pagewise OCR check errored: {exc}")

    # Degraded multipage probe: same 10 documents, deterministic photo/scan
    # degradation. The OCR parser is held out from these pages, and two page
    # failures are part of the result, not hidden.
    deg_summary = ROOT.parent / "invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise_degraded_ocr.json"
    deg_scores = ROOT.parent / "invoice/results/scores_invoice_qwen3-vl_2b_mp_pagewise_degraded_ocr.json"
    deg_rows = ROOT.parent / "invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise_degraded_ocr.jsonl"
    deg_compare = ROOT.parent / "invoice/results/compare_invoice_clean_vs_degraded_2b.json"
    deg_audit = ROOT.parent / "invoice/results/audit_invoice_qwen3-vl_2b_mp_pagewise_degraded_ocr.json"
    deg_recipe = ROOT.parent / "invoice/results/degrade_mp_recipe_42.json"
    deg_retry = ROOT.parent / "invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise_degraded_retry.jsonl"
    deg_retry6p = ROOT.parent / "invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise_degraded_retry6p.jsonl"
    deg_missing = [
        path.name
        for path in (
            deg_summary, deg_scores, deg_rows, deg_compare, deg_audit, deg_recipe,
            deg_retry, deg_retry6p,
        )
        if not path.exists()
    ]
    if deg_missing:
        failures.append(f"degraded multipage artifacts missing: {deg_missing}")
    else:
        try:
            dsum = json.loads(deg_summary.read_text())
            dsc = json.loads(deg_scores.read_text())
            drows = [json.loads(line) for line in deg_rows.open() if line.strip()]
            dcmp = json.loads(deg_compare.read_text())
            daudit = json.loads(deg_audit.read_text())
            drecipe = json.loads(deg_recipe.read_text())
            dretry = [json.loads(line) for line in deg_retry.open() if line.strip()]
            dretry6p = [json.loads(line) for line in deg_retry6p.open() if line.strip()]
            failed_rows = sorted(row["id"] for row in drows if row.get("error"))
            parsed_pages = [
                page
                for row in drows
                for page in row.get("pages") or []
                if not page.get("error")
            ]
            ocr_blocks = [
                row["ocr_totals"]
                for row in drows
                if (row.get("ocr_totals") or {}).get("consistent") is True
            ]
            corrections = {
                row["id"]: (row.get("pred") or {}).get("corrections")
                for row in drows
                if (row.get("pred") or {}).get("corrections")
            }
            ok = (
                dsum.get("n_docs") == 8
                and dsum.get("key_header_correct") == 71
                and dsum.get("key_header_scored") == 72
                and abs(float(dsum.get("key_header_accuracy", -1)) - 0.9861111111111112) <= 1e-9
                and dsum.get("fully_correct") == 7
                and abs(float(dsum.get("fully_correct_rate", -1)) - 0.875) <= 1e-9
                and abs(float(dsum.get("line_items", {}).get("f1_mean", -1)) - 0.795) <= 1e-9
                and dsum.get("line_items", {}).get("all_lines_right") == 0
                and dsum.get("parse_errors") == 2
                and abs(dsum["ci95_exact"]["fully_correct_rate"][0] - 0.4735) < 1e-3
                and abs(float(dsum.get("latency_ms", {}).get("p50", -1)) - 101094.658) < 1.0
                and dsc["summary"]["key_fields"] == {"correct": 71, "scored": 72}
                and dsc["summary"]["fully_correct"] == 7
                and dsc["summary"]["documents"] == 8
                and len(drows) == 10
                and sum(len(row.get("pages") or []) for row in drows) == 46
                and failed_rows == ["multipage-3p-gb-4", "multipage-6p-za-8"]
                and all(page.get("done_reason") == "stop" for page in parsed_pages)
                and len(ocr_blocks) == 10
                and corrections == {
                    "multipage-2p-za-1": ["totals_from_ocr:subtotal+tax+total"],
                    "multipage-3p-na-3": ["totals_from_ocr:tax"],
                    "multipage-5p-na-7": ["currency_from_ocr_totals"],
                }
                and daudit.get("ok") is True
                and daudit.get("allowed_failures") == 2
                and sorted(row["id"] for row in daudit.get("failed_rows") or []) == failed_rows
                and dcmp.get("n_docs") == 8
                and sorted(dcmp.get("missing_from_after") or []) == failed_rows
                and abs(float(dcmp["delta"]["key_accuracy"]) + 0.01388888888888884) < 1e-9
                and abs(float(dcmp["delta"]["fully_correct_rate"]) + 0.125) < 1e-9
                and abs(float(dcmp["delta"]["mean_line_f1"]) + 0.0575) < 1e-3
                and drecipe.get("documents") == 10
                and drecipe.get("pages") == 46
                and drecipe.get("seed") == 42
                and drecipe.get("recipe", {}).get("quality") == 58
                and len(dretry) == 1
                and dretry[0]["id"] == "multipage-3p-gb-4"
                and dretry[0].get("error") == "page 1: no JSON object"
                and len(dretry6p) == 1
                and dretry6p[0]["id"] == "multipage-6p-za-8"
                and dretry6p[0].get("error")
            )
            print(
                f"claim: degraded multipage 2B complete {dsum.get('n_docs')}/10, "
                f"key-header {dsum.get('key_header_correct')}/72 "
                f"({dsum.get('key_header_accuracy'):.4f}), fully correct "
                f"{dsum.get('fully_correct')}/8, line F1 "
                f"{dsum.get('line_items', {}).get('f1_mean'):.3f}; OCR blocks "
                f"{len(ocr_blocks)}/10, failures {failed_rows} -> "
                f"{'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("degraded multipage numbers, failures, or comparison changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"degraded multipage check errored: {exc}")

    # Real Portuguese probe: 100 real phone photos, five fields. This is a
    # development-set measurement; the seed-43 holdout is prepared but not run.
    pt_manifest = ROOT.parent / "invoice/manifest_pt_real_probe100.jsonl"
    pt_manifest_1400 = ROOT.parent / "invoice/manifest_pt_real_probe100_1400.jsonl"
    pt_holdout = ROOT.parent / "invoice/manifest_pt_real_holdout100.jsonl"
    pt_holdout_1400 = ROOT.parent / "invoice/manifest_pt_real_holdout100_1400.jsonl"
    pt_rows = ROOT.parent / "invoice/results/rows_invoice_pt_probe100_1400.jsonl"
    pt_scores = ROOT.parent / "invoice/results/scores_invoice_pt_probe100_1400_full.json"
    pt_summary = ROOT.parent / "invoice/results/summary_invoice_pt_probe100_1400_full.json"
    pt_fix_scores = ROOT.parent / "invoice/results/scores_invoice_pt_probe100_1400_mergefix5.json"
    pt_fix_summary = ROOT.parent / "invoice/results/summary_invoice_pt_probe100_1400_mergefix5.json"
    pt_audit = ROOT.parent / "invoice/results/audit_invoice_pt_probe100_1400.json"
    pt_prep = ROOT.parent / "invoice/results/pt_real_probe_prep_1400.json"
    pt_holdout_prep = ROOT.parent / "invoice/results/pt_real_holdout_prep_1400.json"
    pt_missing = [
        path.name
        for path in (
            pt_manifest, pt_manifest_1400, pt_holdout, pt_holdout_1400, pt_rows,
            pt_scores, pt_summary, pt_fix_scores, pt_fix_summary, pt_audit,
            pt_prep, pt_holdout_prep,
        )
        if not path.exists()
    ]
    if pt_missing:
        failures.append(f"real PT probe artifacts missing: {pt_missing}")
    else:
        try:
            pman = [json.loads(line) for line in pt_manifest.open() if line.strip()]
            pman1400 = [json.loads(line) for line in pt_manifest_1400.open() if line.strip()]
            phold = [json.loads(line) for line in pt_holdout.open() if line.strip()]
            phold1400 = [json.loads(line) for line in pt_holdout_1400.open() if line.strip()]
            prows = [json.loads(line) for line in pt_rows.open() if line.strip()]
            psc = json.loads(pt_scores.read_text())
            psum = json.loads(pt_summary.read_text())
            pfsc = json.loads(pt_fix_scores.read_text())
            pfsum = json.loads(pt_fix_summary.read_text())
            paud = json.loads(pt_audit.read_text())
            pprep = json.loads(pt_prep.read_text())
            phprep = json.loads(pt_holdout_prep.read_text())

            def _pt_counts(scores):
                docs = scores.get("documents") or []
                correct = scored = fully = six = six_scored = 0
                per_field = {}
                for doc in docs:
                    fields = doc.get("fields") or {}
                    correct += sum(1 for item in fields.values() if item.get("correct"))
                    scored += len(fields)
                    fully += int(bool(doc.get("fully_correct")))
                    six += int(bool(doc.get("fully_correct_six")))
                    six_scored += int(bool(doc.get("all_six_scored")))
                    for name, item in fields.items():
                        bucket = per_field.setdefault(name, [0, 0])
                        bucket[1] += 1
                        bucket[0] += int(bool(item.get("correct")))
                return correct, scored, fully, six, six_scored, per_field

            correct, scored, fully, six, six_scored, per_field = _pt_counts(psc)
            fcorrect, fscored, ffully, fsix, fsix_scored, fper = _pt_counts(pfsc)
            expected_full = {
                "currency": [96, 100],
                "invoice_date": [93, 97],
                "invoice_number": [44, 98],
                "tax": [67, 87],
                "total": [19, 100],
                "vendor_name": [55, 95],
            }
            expected_fix = {
                "currency": [96, 100],
                "invoice_date": [93, 97],
                "invoice_number": [44, 98],
                "tax": [68, 87],
                "total": [72, 100],
                "vendor_name": [55, 95],
            }
            dev_ids = {str(row["id"]) for row in pman}
            ok = (
                len(pman) == 100
                and len(pman1400) == 100
                and dev_ids == {str(row["id"]) for row in pman1400}
                and len(prows) == 100
                and (correct, scored) == (374, 577)
                and fully == 3
                and (six, six_scored) == (2, 80)
                and (fcorrect, fscored) == (428, 577)
                and ffully == 18
                and (fsix, fsix_scored) == (13, 80)
                and per_field == expected_full
                and fper == expected_fix
                and abs(float(psum.get("field_accuracy", -1)) - 374 / 577) < 1e-9
                and abs(float(pfsum.get("field_accuracy", -1)) - 428 / 577) < 1e-9
                and psum.get("fully_correct") == 3
                and pfsum.get("fully_correct") == 18
                and psum.get("fully_correct_six") == 2
                and pfsum.get("fully_correct_six") == 13
                and paud.get("documents") == 100
                and paud.get("rows") == 100
                and paud.get("failed") == 1
                and paud.get("allowed_failed") == 1
                and [item.get("doc_id") for item in paud.get("failures") or []]
                == ["ptreal-personal-20210215_164929"]
                and (paud.get("failures") or [{}])[0].get("detail") == "no JSON object; done_reason=length"
                and (paud.get("truncated_pages") or [{}])[0].get("eval_count") == 2048
                and pprep.get("documents") == 100
                and pprep.get("max_px") == 1400
                and pprep.get("resized") == 100
                and len(phold) == 100
                and len(phold1400) == 100
                and {str(row["id"]) for row in phold} == {str(row["id"]) for row in phold1400}
                and not (dev_ids & {str(row["id"]) for row in phold})
                and phprep.get("documents") == 100
                and phprep.get("max_px") == 1400
            )
            print(
                f"claim: real PT probe 2B 1400px 100 docs -> field {correct}/{scored} "
                f"({correct / scored:.4f}), fully {fully}/100; merge fix {fcorrect}/{fscored} "
                f"({fcorrect / fscored:.4f}), fully {ffully}/100; holdout "
                f"{len(phold)} docs disjoint -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("real PT probe numbers, failure accounting, or holdout split changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"real PT probe check errored: {exc}")

    # Real Portuguese holdout: disjoint seed-43 documents, fixed merge, run once.
    hold_scores = ROOT.parent / "invoice/results/scores_invoice_pt_holdout100_1400_full.json"
    hold_summary = ROOT.parent / "invoice/results/summary_invoice_pt_holdout100_1400_full.json"
    hold_rows = ROOT.parent / "invoice/results/rows_invoice_pt_holdout100_1400.jsonl"
    hold_audit = ROOT.parent / "invoice/results/audit_invoice_pt_holdout100_1400.json"
    hold_compare = ROOT.parent / "invoice/results/compare_pt_dev_vs_holdout.json"
    hold_missing = [
        path.name
        for path in (hold_scores, hold_summary, hold_rows, hold_audit, hold_compare)
        if not path.exists()
    ]
    if hold_missing:
        failures.append(f"real PT holdout artifacts missing: {hold_missing}")
    else:
        try:
            hsc = json.loads(hold_scores.read_text())
            hsum = json.loads(hold_summary.read_text())
            hrows = [json.loads(line) for line in hold_rows.open() if line.strip()]
            haud = json.loads(hold_audit.read_text())
            hcmp = json.loads(hold_compare.read_text())
            hdocs = hsc.get("documents") or []
            hcorrect = sum(
                1 for doc in hdocs for item in (doc.get("fields") or {}).values() if item.get("correct")
            )
            hscored = sum(len(doc.get("fields") or {}) for doc in hdocs)
            hfully = sum(1 for doc in hdocs if doc.get("fully_correct"))
            hsix = sum(1 for doc in hdocs if doc.get("fully_correct_six"))
            hsix_scored = sum(1 for doc in hdocs if doc.get("all_six_scored"))
            hper = {}
            for doc in hdocs:
                for name, item in (doc.get("fields") or {}).items():
                    bucket = hper.setdefault(name, [0, 0])
                    bucket[1] += 1
                    bucket[0] += int(bool(item.get("correct")))
            expected_hold = {
                "currency": [96, 100],
                "invoice_date": [91, 98],
                "invoice_number": [41, 98],
                "tax": [59, 91],
                "total": [76, 100],
                "vendor_name": [56, 95],
            }
            hfail = haud.get("failures") or []
            ok = (
                (hcorrect, hscored) == (419, 582)
                and hfully == 22
                and (hsix, hsix_scored) == (15, 84)
                and hper == expected_hold
                and abs(float(hsum.get("field_accuracy", -1)) - 419 / 582) < 1e-9
                and hsum.get("fully_correct") == 22
                and hsum.get("fully_correct_six") == 15
                and len(hrows) == 100
                and haud.get("documents") == 100
                and haud.get("failed") == 2
                and haud.get("allowed_failed") == 2
                and [item.get("doc_id") for item in hfail]
                == ["ptreal-company-20210323_184527", "ptreal-personal-20210215_165018"]
                and [item.get("check") for item in hfail] == ["done_reason", "page_error"]
                and len(haud.get("merge_sanity") or []) == 3
                and (haud.get("truncated_pages") or [{}])[0].get("eval_count") == 2048
                and abs(float(hcmp["delta"]["field_accuracy"]) + 0.0218364928204303) < 1e-9
                and abs(float(hcmp["delta"]["fully_correct_rate"]) - 0.04) < 1e-9
            )
            print(
                f"claim: real PT holdout 2B 1400px -> field {hcorrect}/{hscored} "
                f"({hcorrect / hscored:.4f}), fully {hfully}/100, six {hsix}/{hsix_scored}; "
                f"2 hard failures -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("real PT holdout numbers, failures, or comparison changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"real PT holdout check errored: {exc}")

    # Prompt pilot: a 30-document development check of two receipt-aware prompts.
    # Both were rejected; v1 remains canonical.
    pilot_manifest = ROOT.parent / "invoice/manifest_pt_pilot30_1400.jsonl"
    pilot_scores = {
        name: ROOT.parent / f"invoice/results/scores_invoice_pt_pilot30_{name}.json"
        for name in ("v1", "v3", "v4")
    }
    pilot_summaries = {
        name: ROOT.parent / f"invoice/results/summary_invoice_pt_pilot30_{name}.json"
        for name in ("v1", "v3", "v4")
    }
    pilot_compare = {
        name: ROOT.parent / f"invoice/results/compare_pt_pilot30_v1_vs_{name}.json"
        for name in ("v3", "v4")
    }
    pilot_rows = {
        name: ROOT.parent / f"invoice/results/rows_invoice_pt_pilot30_{name}.jsonl"
        for name in ("v3", "v4")
    }
    pilot_missing = [
        path.name
        for path in (pilot_manifest, *pilot_scores.values(), *pilot_summaries.values(),
                     *pilot_compare.values(), *pilot_rows.values())
        if not path.exists()
    ]
    if pilot_missing:
        failures.append(f"prompt pilot artifacts missing: {pilot_missing}")
    else:
        try:
            pman = [json.loads(line) for line in pilot_manifest.open() if line.strip()]

            def _pilot_totals(scores):
                docs = scores.get("documents") or []
                correct = sum(
                    1 for doc in docs for item in (doc.get("fields") or {}).values() if item.get("correct")
                )
                scored = sum(len(doc.get("fields") or {}) for doc in docs)
                fully = sum(1 for doc in docs if doc.get("fully_correct"))
                six = sum(1 for doc in docs if doc.get("fully_correct_six"))
                six_scored = sum(1 for doc in docs if doc.get("all_six_scored"))
                return correct, scored, fully, six, six_scored

            vals = {
                name: _pilot_totals(json.loads(path.read_text()))
                for name, path in pilot_scores.items()
            }
            v3rows = [json.loads(line) for line in pilot_rows["v3"].open() if line.strip()]
            v4rows = [json.loads(line) for line in pilot_rows["v4"].open() if line.strip()]
            v3caps = sorted(
                row["id"] for row in v3rows
                if row.get("pages") and (row["pages"][0].get("eval_count") or 0) >= 2048
            )
            v4caps = [
                row["id"] for row in v4rows
                if row.get("pages") and (row["pages"][0].get("eval_count") or 0) >= 2048
            ]
            c3 = json.loads(pilot_compare["v3"].read_text())
            c4 = json.loads(pilot_compare["v4"].read_text())
            ok = (
                len(pman) == 30
                and vals["v1"] == (133, 174, 7, 6, 24)
                and vals["v3"] == (138, 174, 7, 6, 24)
                and vals["v4"] == (131, 174, 6, 5, 24)
                and v3caps == ["ptreal-company-20210318_172727", "ptreal-company-20210508_174017"]
                and not v4caps
                and c3["delta"]["fully_correct"] == 0
                and c3["per_field"]["total"]["delta"] == 2
                and c3["per_field"]["tax"]["delta"] == 0
                and c4["delta"]["fully_correct"] == -1
                and c4["per_field"]["tax"]["delta"] == -2
                and c4["per_field"]["total"]["delta"] == -2
            )
            print(
                f"claim: prompt pilot 30 docs -> v1 {vals['v1'][0]}/174 fully {vals['v1'][2]}, "
                f"v3 {vals['v3'][0]}/174 fully {vals['v3'][2]} with {len(v3caps)} cap loops, "
                f"v4 {vals['v4'][0]}/174 fully {vals['v4'][2]} -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("prompt-pilot counts or loop accounting changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"prompt pilot check errored: {exc}")

    # Resolution pilot: same 30 development documents at 1400 px vs 2400 px.
    res_manifest = ROOT.parent / "invoice/manifest_pt_pilot30_2400.jsonl"
    res_scores = ROOT.parent / "invoice/results/scores_invoice_pt_pilot30_2400_v1.json"
    res_summary = ROOT.parent / "invoice/results/summary_invoice_pt_pilot30_2400_v1.json"
    res_rows = ROOT.parent / "invoice/results/rows_invoice_pt_pilot30_2400_v1.jsonl"
    res_compare = ROOT.parent / "invoice/results/compare_pt_pilot30_1400_vs_2400.json"
    res_missing = [
        path.name
        for path in (res_manifest, res_scores, res_summary, res_rows, res_compare)
        if not path.exists()
    ]
    if res_missing:
        failures.append(f"resolution pilot artifacts missing: {res_missing}")
    else:
        try:
            rman = [json.loads(line) for line in res_manifest.open() if line.strip()]
            rsc = json.loads(res_scores.read_text())
            rsum = json.loads(res_summary.read_text())
            rrows = [json.loads(line) for line in res_rows.open() if line.strip()]
            rcmp = json.loads(res_compare.read_text())
            rdocs = rsc.get("documents") or []
            rcorrect = sum(
                1 for doc in rdocs for item in (doc.get("fields") or {}).values() if item.get("correct")
            )
            rscored = sum(len(doc.get("fields") or {}) for doc in rdocs)
            rfully = sum(1 for doc in rdocs if doc.get("fully_correct"))
            rsix = sum(1 for doc in rdocs if doc.get("fully_correct_six"))
            rsix_scored = sum(1 for doc in rdocs if doc.get("all_six_scored"))
            rper = {}
            for doc in rdocs:
                for name, item in (doc.get("fields") or {}).items():
                    bucket = rper.setdefault(name, [0, 0])
                    bucket[1] += 1
                    bucket[0] += int(bool(item.get("correct")))
            expected_res = {
                "currency": [29, 30],
                "invoice_date": [30, 30],
                "invoice_number": [18, 30],
                "tax": [21, 25],
                "total": [21, 30],
                "vendor_name": [22, 29],
            }
            no_total = [row for row in rrows if row.get("error") == "no total"]
            ok = (
                len(rman) == 30
                and (rcorrect, rscored) == (141, 174)
                and rfully == 11
                and (rsix, rsix_scored) == (9, 24)
                and rper == expected_res
                and abs(float(rsum.get("field_accuracy", -1)) - 141 / 174) < 1e-9
                and rsum.get("fully_correct") == 11
                and abs(float(rsum.get("latency_ms", {}).get("p50", -1)) - 33798.19703102112) < 1.0
                and len(no_total) == 1
                and no_total[0]["id"] == "ptreal-company-20210318_174038"
                and no_total[0]["pages"][0].get("done_reason") == "stop"
                and abs(float(rcmp["delta"]["field_accuracy"]) - 0.04597701149425282) < 1e-9
                and rcmp["delta"]["fully_correct"] == 4
                and rcmp["field_pairs"] == {"both": 130, "a_only": 3, "b_only": 11, "neither": 30}
                and rcmp["doc_pairs"] == {"both": 7, "a_only": 0, "b_only": 4, "neither": 19}
            )
            print(
                f"claim: resolution pilot 30 docs -> 1400 133/174 fully 7; 2400 {rcorrect}/174 "
                f"fully {rfully}, six {rsix}/{rsix_scored}, p50 "
                f"{rsum.get('latency_ms', {}).get('p50', 0) / 1000:.1f}s -> "
                f"{'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("resolution-pilot counts or latency changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"resolution pilot check errored: {exc}")

    # Resolution holdout-50: fixed subset of the seed-43 holdout, 1400 vs 2400.
    h50_manifest = ROOT.parent / "invoice/manifest_pt_holdout50_2400.jsonl"
    h50_scores = {
        name: ROOT.parent / f"invoice/results/scores_invoice_pt_holdout50_{name}.json"
        for name in ("1400", "2400_v1", "hybrid")
    }
    h50_summaries = {
        name: ROOT.parent / f"invoice/results/summary_invoice_pt_holdout50_{name}.json"
        for name in ("1400", "2400_v1", "hybrid")
    }
    h50_rows = {
        name: ROOT.parent / f"invoice/results/rows_invoice_pt_holdout50_{name}.jsonl"
        for name in ("1400", "2400_v1")
    }
    h50_compare = ROOT.parent / "invoice/results/compare_pt_holdout50_1400_vs_2400.json"
    h50_prep = ROOT.parent / "invoice/results/pt_real_holdout_prep_2400.json"
    h50_missing = [
        path.name
        for path in (h50_manifest, *h50_scores.values(), *h50_summaries.values(),
                     *h50_rows.values(), h50_compare, h50_prep)
        if not path.exists()
    ]
    if h50_missing:
        failures.append(f"resolution holdout-50 artifacts missing: {h50_missing}")
    else:
        try:
            hman = [json.loads(line) for line in h50_manifest.open() if line.strip()]

            def _h50_totals(scores):
                docs = scores.get("documents") or []
                correct = sum(
                    1 for doc in docs for item in (doc.get("fields") or {}).values() if item.get("correct")
                )
                scored = sum(len(doc.get("fields") or {}) for doc in docs)
                fully = sum(1 for doc in docs if doc.get("fully_correct"))
                six = sum(1 for doc in docs if doc.get("fully_correct_six"))
                six_scored = sum(1 for doc in docs if doc.get("all_six_scored"))
                return correct, scored, fully, six, six_scored

            hv = {name: _h50_totals(json.loads(path.read_text())) for name, path in h50_scores.items()}
            hcmp = json.loads(h50_compare.read_text())
            hprep = json.loads(h50_prep.read_text())
            h2400_rows = [json.loads(line) for line in h50_rows["2400_v1"].open() if line.strip()]
            ok = (
                len(hman) == 50
                and hv["1400"] == (211, 289, 11, 8, 40)
                and hv["2400_v1"] == (221, 289, 12, 11, 40)
                and hv["hybrid"] == (226, 289, 13, 11, 40)
                and abs(float(hcmp["delta"]["field_accuracy"]) - 10 / 289) < 1e-9
                and hcmp["delta"]["fully_correct"] == 1
                and hcmp["field_pairs"] == {"both": 198, "a_only": 13, "b_only": 23, "neither": 55}
                and hcmp["doc_pairs"] == {"both": 7, "a_only": 4, "b_only": 5, "neither": 34}
                and abs(float(json.loads(h50_summaries["2400_v1"].read_text())["latency_ms"]["p50"]) - 36599.40004348755) < 1.0
                and len(h2400_rows) == 50
                and all(not row.get("error") for row in h2400_rows)
                and hprep.get("documents") == 100
                and hprep.get("max_px") == 2400
            )
            print(
                f"claim: resolution holdout50 -> 1400 {hv['1400'][0]}/289 fully {hv['1400'][2]}, "
                f"2400 {hv['2400_v1'][0]}/289 fully {hv['2400_v1'][2]}, hybrid "
                f"{hv['hybrid'][0]}/289 fully {hv['hybrid'][2]} -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append("resolution holdout-50 counts or comparison changed")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"resolution holdout-50 check errored: {exc}")

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
