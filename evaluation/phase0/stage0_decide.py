"""Stage-0 decision: paired comparison of zero-shot SmolVLM-500M vs the 3B baseline.

Precommitted rule (written before the run, in .unlazy/stage0-500m-probe/GATES.md):
    f1_500m >= 0.50            -> GO        (base strong enough to fine-tune)
    0.30 <= f1_500m < 0.50     -> BORDERLINE (char-error analysis decides)
    f1_500m < 0.30             -> NO-GO     (base too weak; stop before training)

Also reports paired per-doc win/loss vs qwen2.5vl:3b, address character error
rate, and latency, so the verdict rests on more than one aggregate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from metrics import FIELDS, levenshtein, macro_prf

RESULTS = ROOT / "evaluation/phase0/results"
PRED500 = RESULTS / "preds_smolvlm500m_zeroshot.jsonl"
PRED3B = RESULTS / "preds_pipeline_ollama_qwen2.5vl_3b.jsonl"


def cer(pred: str, gold: str) -> float:
    if not gold:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gold) / max(len(gold), 1)


def main() -> int:
    rows500 = [json.loads(l) for l in PRED500.open() if l.strip()]
    rows3b = [json.loads(l) for l in PRED3B.open() if l.strip()]
    by_id3b = {r["id"]: r for r in rows3b}

    scores500 = [r["score"] for r in rows500]
    agg500 = macro_prf(scores500)

    # paired subset: only docs present in both runs
    paired = [r for r in rows500 if r["id"] in by_id3b]
    scores3b_paired = [by_id3b[r["id"]]["score"] for r in paired]
    agg3b = macro_prf(scores3b_paired) if paired else None

    wins = losses = ties = 0
    for r in paired:
        f5 = r["score"]["f1"]
        f3 = by_id3b[r["id"]]["score"]["f1"]
        if f5 > f3:
            wins += 1
        elif f5 < f3:
            losses += 1
        else:
            ties += 1

    addr_cer = [cer(r["pred"].get("address", ""), r["gold"].get("address", "")) for r in rows500]
    addr_exact = sum(1 for r in rows500 if r["score"]["fields"]["address"]["hit"])
    mean_cer = sum(addr_cer) / len(addr_cer) if addr_cer else 0.0
    near_miss_cer = [c for c in addr_cer if 0 < c < 0.05]
    errors = sum(1 for r in rows500 if r.get("error"))
    avg_lat = sum(r["latency_ms"] for r in rows500) / max(len(rows500), 1)

    f1_500m = agg500["field_f1"]
    if f1_500m >= 0.50:
        verdict = "GO"
        reason = "zero-shot field F1 >= 0.50: base model reads receipts well enough that SFT on real data can plausibly close the gap to the 3B baseline (0.870) and attack the char-exact address path."
    elif f1_500m >= 0.30:
        verdict = "BORDERLINE"
        reason = (
            f"zero-shot field F1 {f1_500m:.3f} in [0.30, 0.50): char-error analysis decides. "
            f"address exact {addr_exact}/{len(rows500)}, mean address CER {mean_cer:.4f}, "
            f"{len(near_miss_cer)} near-misses (CER<0.05). "
            + ("Errors are near-misses (precision, learnable by SFT) rather than hallucinations -> GO."
               if len(near_miss_cer) >= 0.5 * max(addr_exact, 1) and mean_cer < 0.25
               else "Errors include gross hallucinations or empty fields -> NO-GO, retrain-free base too weak.")
        )
    else:
        verdict = "NO-GO"
        reason = f"zero-shot field F1 {f1_500m:.3f} < 0.30: base too weak to justify SFT compute; stop here."

    out = {
        "verdict": verdict,
        "reason": reason,
        "f1_500m": round(f1_500m, 4),
        "f1_3b": 0.87,
        "f1_3b_paired_subset": round(agg3b["field_f1"], 4) if agg3b else None,
        "n_500m": len(rows500),
        "n_paired": len(paired),
        "paired_doc_wins_500m": wins,
        "paired_doc_losses_500m": losses,
        "paired_doc_ties": ties,
        "per_field_hit_500m": {
            f: round(sum(1 for s in scores500 if s["fields"][f]["hit"]) / len(scores500), 4)
            for f in FIELDS
        },
        "address_exact_hits": addr_exact,
        "address_mean_cer": round(mean_cer, 4),
        "address_near_miss_cer_lt_0.05": len(near_miss_cer),
        "schema_valid_rate": agg500["schema_valid_rate"],
        "avg_latency_ms": round(avg_lat, 1),
        "errors": errors,
        "artifacts": {
            "preds_500m": str(PRED500.relative_to(ROOT)),
            "preds_3b": str(PRED3B.relative_to(ROOT)),
        },
        "rule": "f1>=0.50 GO; 0.30-0.50 BORDERLINE (char analysis decides); <0.30 NO-GO (precommitted)",
    }
    dest = RESULTS / "stage0_decision.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
