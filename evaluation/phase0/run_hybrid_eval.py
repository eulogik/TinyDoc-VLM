"""Hybrid v2 eval: 2.2B-base VLM fields + tesseract-OCR address (frozen rule F3).

Frozen rule F3 (selected on val-40 ONLY, never test):
  address = VLM address iff normalize(VLM) is a non-empty substring of
            normalize(OCR window), else the OCR window, where
  OCR window = tesseract PSM-6 lines[max(0, i-4)..i+1] around the first line
             containing a 5-digit postcode (Malaysian postcode anchor).
Val result: 20/40 address hits (vs VLM 6/40, OCR-window-only 18/40).
Company/date/total always come from the VLM (same base zero-shot run as the
paired baseline, so the comparison isolates the address channel).

Usage:
  ocr_venv/bin/python evaluation/phase0/run_hybrid_eval.py \
      --vlm-preds results/preds_smolvlm2-2.2b_base_clean.jsonl \
      --eval-path results/sroie_eval_clean.json --out-suffix _hybrid_clean
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from metrics import FIELDS, field_match, macro_prf, normalize_text

RESULTS = ROOT / "evaluation/phase0/results"
RULE = "F3: tesseract psm-6, postcode window -4/+1, keep VLM iff norm(VLM) in norm(window)"


def ocr_lines(image_path: str) -> list[str]:
    out = subprocess.run(
        ["tesseract", image_path, "stdout", "--psm", "6", "-l", "eng"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def ocr_window(lines: list[str]) -> str:
    idx = next((i for i, l in enumerate(lines) if re.search(r"\b\d{5}\b", l)), None)
    if idx is None:
        return ""
    return " ".join(lines[max(0, idx - 4):idx + 2])


def hybrid_address(vlm_addr: str, window: str) -> str:
    v = normalize_text(vlm_addr)
    if v and v in normalize_text(window):
        return vlm_addr.strip()
    return window


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-preds", required=True)
    ap.add_argument("--eval-path", default=str(RESULTS / "sroie_eval_clean.json"))
    ap.add_argument("--out-suffix", default="_hybrid_clean")
    args = ap.parse_args()

    from metrics import score_example

    V = {r["id"]: r for r in
         map(json.loads, Path(args.vlm_preds).read_text().splitlines()) if True}
    items = json.loads(Path(args.eval_path).read_text())

    preds, scores = [], []
    t0_all = time.time()
    for n, item in enumerate(items):
        v = V[item["id"]]
        t0 = time.time()
        try:
            W = ocr_window(ocr_lines(item["image_path"]))
            addr = hybrid_address(v["pred"].get("address", ""), W)
            error = ""
        except Exception as e:  # noqa: BLE001
            addr, W, error = v["pred"].get("address", ""), "", f"{type(e).__name__}: {e}"
        lat = (time.time() - t0) * 1000
        pred = {f: v["pred"].get(f, "") for f in FIELDS}
        pred["address"] = addr
        gold = {f: item["gold"].get(f, "") for f in FIELDS}
        sc = score_example(pred, gold)
        preds.append({"id": item["id"], "image_path": item["image_path"],
                      "pred": pred, "gold": gold, "latency_ms": lat,
                      "ocr_window": W[:500], "vlm_address": v["pred"].get("address", ""),
                      "rule": RULE, "error": error, "score": sc})
        scores.append(sc)
        if (n + 1) % 20 == 0 or n + 1 == len(items):
            print(f"  [{n + 1}/{len(items)}] f1={macro_prf(scores)['field_f1']:.3f}",
                  flush=True)
    agg = macro_prf(scores)
    out = {
        "engine": f"hybrid-2.2b-tess{args.out_suffix}",
        "rule": RULE,
        "vlm_source": str(args.vlm_preds),
        "n_examples": len(preds),
        "field_f1": agg["field_f1"],
        "schema_valid_rate": agg["schema_valid_rate"],
        "anls_mean": agg["anls_mean"],
        "avg_latency_ms": sum(p["latency_ms"] for p in preds) / max(len(preds), 1),
        "total_wall_s": round(time.time() - t0_all, 1),
        "per_field_hit_rate": {
            f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores)
            for f in FIELDS
        },
        "address_vlm_only_hits": sum(
            1 for p in preds
            if field_match(p["vlm_address"], p["gold"]["address"], "address")),
        "errors": sum(1 for p in preds if p["error"]),
    }
    (RESULTS / f"scores_hybrid-2.2b-tess{args.out_suffix}.json").write_text(
        json.dumps(out, indent=2))
    (RESULTS / f"preds_hybrid-2.2b-tess{args.out_suffix}.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
