#!/usr/bin/env bash
# Rebuild page-wise predictions from saved raws with OCR total reconciliation,
# then score, summarize, and audit. No model calls; OCR sees only document pages.
#
# Usage: scripts/reconcile_invoice_totals.sh <raw_rows.jsonl> <tag> [model]
# Example:
#   scripts/reconcile_invoice_totals.sh \
#     evaluation/invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise_v2_raw.jsonl \
#     qwen3-vl_2b_mp_pagewise_v2
#
# ALLOW_FAILED defaults to 0. Set it only for a documented probe whose page
# failures are reported; the audit still lists every failed row.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

ROWS_IN="${1:?usage: reconcile_invoice_totals.sh <raw_rows.jsonl> <tag> [model]}"
TAG="${2:?usage: reconcile_invoice_totals.sh <raw_rows.jsonl> <tag> [model]}"
MODEL="${3:-qwen3-vl:2b-instruct}"
MANIFEST="${INVOICE_MANIFEST:-evaluation/invoice/manifest_multipage.jsonl}"
BENCH="${INVOICE_BENCHMARK:-/Volumes/KIOXIA 1TB/tinydoc/invoice-extraction-benchmark}"
ROOT="$(pwd)"
R="$ROOT/evaluation/invoice/results"
mkdir -p "$R" logs

if [ ! -f "$ROWS_IN" ]; then
  echo "ERROR: rows missing: $ROWS_IN" >&2
  exit 2
fi
if [ ! -d "$BENCH/scorer" ]; then
  echo "ERROR: benchmark scorer missing: $BENCH/scorer" >&2
  exit 2
fi

echo "START reconcile tag=$TAG $(date '+%Y-%m-%d %H:%M:%S')"
python3 evaluation/invoice/rebuild_pagewise.py \
  --rows "$ROWS_IN" \
  --out-rows "$R/rows_invoice_${TAG}.jsonl" \
  --out-preds "$R/preds_invoice_${TAG}.json" \
  --out-report "$R/rebuild_invoice_${TAG}.json" \
  --model "$MODEL" \
  --ocr-totals \
  > "logs/reconcile_${TAG}.log" 2>&1
rc=$?
echo "REBUILD rc=$rc"
if [ "$rc" -ne 0 ]; then
  tail -20 "logs/reconcile_${TAG}.log" >&2
  exit "$rc"
fi

(
  cd "$BENCH" || exit 1
  python3 -m scorer.evaluate "$R/preds_invoice_${TAG}.json" \
    --out "$R/scores_invoice_${TAG}.json" \
    --markdown "$R/scores_invoice_${TAG}.md" \
    --allow-missing >> "$ROOT/logs/reconcile_${TAG}.log" 2>&1
)
score_rc=$?
echo "SCORE rc=$score_rc"
if [ "$score_rc" -ne 0 ]; then
  tail -20 "logs/reconcile_${TAG}.log" >&2
  exit "$score_rc"
fi

python3 evaluation/invoice/summarize_scores.py \
  --scores "$R/scores_invoice_${TAG}.json" \
  --rows "$R/rows_invoice_${TAG}.jsonl" \
  --out "$R/summary_invoice_${TAG}.json" >/dev/null 2>&1
summary_rc=$?
echo "SUMMARY rc=$summary_rc"
if [ "$summary_rc" -ne 0 ]; then
  echo "ERROR: summary failed for $TAG" >&2
  exit "$summary_rc"
fi

python3 evaluation/invoice/audit_pagewise_rows.py \
  --manifest "$MANIFEST" \
  --rows "$R/rows_invoice_${TAG}.jsonl" \
  --num-ctx 8192 --num-predict 2048 \
  --allow-failed "${ALLOW_FAILED:-0}" \
  > "$R/audit_invoice_${TAG}.json" 2>&1
audit_rc=$?
echo "AUDIT rc=$audit_rc"

python3 - <<PY
import json
from pathlib import Path
r = Path("$R")
summary = json.loads((r / "summary_invoice_${TAG}.json").read_text())
rebuild = json.loads((r / "rebuild_invoice_${TAG}.json").read_text())
audit = json.loads((r / "audit_invoice_${TAG}.json").read_text())
print(json.dumps({
    "tag": "$TAG",
    "key_header": [summary["key_header_correct"], summary["key_header_scored"]],
    "key_header_accuracy": summary["key_header_accuracy"],
    "fully_correct": summary["fully_correct"],
    "fully_correct_rate": summary["fully_correct_rate"],
    "line_f1": summary["line_items"]["f1_mean"],
    "parse_errors": summary["parse_errors"],
    "ocr_totals_consistent": rebuild.get("ocr_totals_consistent"),
    "corrections": rebuild.get("corrections"),
    "review_flags": rebuild.get("review_flags"),
    "audit_ok": audit.get("ok"),
    "p50_ms": summary["latency_ms"]["p50"],
}, indent=2))
PY

if [ "$audit_rc" -ne 0 ]; then
  echo "ERROR: audit failed for $TAG; see $R/audit_invoice_${TAG}.json" >&2
  exit "$audit_rc"
fi
echo "END reconcile tag=$TAG $(date '+%Y-%m-%d %H:%M:%S')"
