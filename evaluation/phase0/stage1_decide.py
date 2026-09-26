"""Stage-1 decision: paired adapter-vs-3B verdict on the clean 100.

Inputs are the two scored runs (same docs, same scorer). Reports aggregates
AND paired per-doc stats (W/L/T on per-doc F1, exact binomial sign test on
address hits) AND the char-exact address panel (raw exact-match rate, mean
CER) — because the gate's address hit-rate uses containment-tolerant matching
while the product goal is char-exactness.
Writes evaluation/phase0/results/stage1_decision.json (ship/no_ship per the
precommitted bar: address-hit >= 0.80 AND F1 >= 3B clean F1 AND latency <= 1s/doc).
Usage:
  python evaluation/phase0/stage1_decide.py \
      --adapter-preds results/preds_smolvlm500m_lora_clean.jsonl \
      --baseline-preds results/preds_pipeline_ollama_qwen2.5vl_3b_clean.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from metrics import address_metrics, macro_prf

RESULTS = ROOT / "evaluation/phase0/results"


def sign_test(b: int, c: int) -> float:
    """Two-sided exact binomial p for b wins vs c losses (ties discarded)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def load_preds(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-preds", required=True)
    ap.add_argument("--baseline-preds", required=True)
    ap.add_argument("--address-bar", type=float, default=0.80)
    ap.add_argument("--latency-bar-ms", type=float, default=1000.0)
    ap.add_argument("--out", default=str(RESULTS / "stage1_decision.json"))
    args = ap.parse_args()

    A = load_preds(Path(args.adapter_preds))
    B = load_preds(Path(args.baseline_preds))
    paired_ids = sorted(set(A) & set(B))
    assert len(paired_ids) == 100, f"expected 100 paired docs, got {len(paired_ids)}"

    agg_a = macro_prf([A[i]["score"] for i in paired_ids])
    agg_b = macro_prf([B[i]["score"] for i in paired_ids])

    wins = losses = ties = 0
    ab = ba = 0  # address-hit discordants: adapter-only hit / baseline-only hit
    for i in paired_ids:
        fa = A[i]["score"]["f1"]
        fb = B[i]["score"]["f1"]
        if fa > fb:
            wins += 1
        elif fa < fb:
            losses += 1
        else:
            ties += 1
        ha = A[i]["score"]["fields"]["address"]["hit"]
        hb = B[i]["score"]["fields"]["address"]["hit"]
        ab += ha and not hb
        ba += hb and not ha

    def addr_panel(rows: dict) -> dict:
        hits = sum(1 for i in paired_ids if rows[i]["score"]["fields"]["address"]["hit"])
        exact = sum(1 for i in paired_ids
                    if address_metrics(rows[i]["pred"].get("address", ""),
                                       rows[i]["gold"].get("address", ""))["raw_exact"])
        cers = [address_metrics(rows[i]["pred"].get("address", ""),
                                rows[i]["gold"].get("address", ""))["raw_cer"]
                for i in paired_ids]
        return {"hit_rate": hits / len(paired_ids),
                "raw_exact_rate": exact / len(paired_ids),
                "mean_cer": sum(cers) / len(cers)}

    lat_a = sum(A[i]["latency_ms"] for i in paired_ids) / len(paired_ids)
    lat_b = sum(B[i]["latency_ms"] for i in paired_ids) / len(paired_ids)

    addr_a, addr_b = addr_panel(A), addr_panel(B)
    crit = {
        "address_hit_ge_bar": addr_a["hit_rate"] >= args.address_bar,
        "f1_no_regression": agg_a["field_f1"] >= agg_b["field_f1"],
        "latency_le_bar": lat_a <= args.latency_bar_ms,
    }
    decision = "ship" if all(crit.values()) else "no_ship"
    out = {
        "n_paired": len(paired_ids),
        "f1_adapter": round(agg_a["field_f1"], 4),
        "f1_3b_clean": round(agg_b["field_f1"], 4),
        "paired_doc_wins_adapter": wins,
        "paired_doc_losses_adapter": losses,
        "paired_doc_ties": ties,
        "address_sign_test_p": round(sign_test(ab, ba), 4),
        "address_discordants": {"adapter_only_hit": ab, "baseline_only_hit": ba},
        "address_adapter": addr_a,
        "address_baseline": addr_b,
        "latency_ms_adapter": round(lat_a, 1),
        "latency_ms_baseline": round(lat_b, 1),
        "errors_adapter": sum(1 for i in paired_ids if A[i].get("error")),
        "errors_baseline": sum(1 for i in paired_ids if B[i].get("error")),
        "criteria": crit,
        "bars": {"address_hit": args.address_bar, "latency_ms": args.latency_bar_ms},
        "decision": decision,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
