#!/usr/bin/env bash
# Run the real Portuguese probe with a macOS memory watchdog.
#
# Usage:
#   scripts/run_pt_probe.sh <tag> <manifest> [num_predict] [limit]
#
# The watchdog checks free+inactive+speculative memory every 10 s and kills the
# runner below TINYDOC_MIN_FREE_MB (default 800). This is a 16 GB M4; sustained
# 7B-on-MPS generation has crashed it before, and the 2B multipage run hit
# Ollama HTTP 500 under memory pressure.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:?usage: run_pt_probe.sh <tag> <manifest> [num_predict] [limit]}"
MANIFEST="${2:?usage: run_pt_probe.sh <tag> <manifest> [num_predict] [limit]}"
NUM_PREDICT="${3:-2048}"
LIMIT="${4:-0}"

RESULTS="$ROOT/evaluation/invoice/results"
ROWS="$RESULTS/rows_invoice_${TAG}.jsonl"
PREDS="$RESULTS/preds_invoice_${TAG}.json"
LOG="$RESULTS/run_${TAG}.log"
MIN_FREE_MB="${TINYDOC_MIN_FREE_MB:-800}"
PAGESIZE="$(sysctl -n hw.pagesize)"

mkdir -p "$RESULTS"
: >"$LOG"

ARGS=(
  python3 "$ROOT/evaluation/invoice/run_invoice_pagewise.py"
  --manifest "$MANIFEST"
  --model qwen3-vl:2b-instruct
  --out "$ROWS"
  --preds "$PREDS"
  --prompt-version 1
  --num-ctx "${TINYDOC_OLLAMA_NUM_CTX:-8192}"
  --num-predict "$NUM_PREDICT"
  --timeout "${TINYDOC_OLLAMA_TIMEOUT:-600}"
)
if [ "$LIMIT" -gt 0 ]; then
  ARGS+=(--limit "$LIMIT")
fi

"${ARGS[@]}" >>"$LOG" 2>&1 &
PID=$!

cleanup() {
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "run tag=$TAG pid=$PID min_free_mb=$MIN_FREE_MB" | tee -a "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  FREE_MB="$(vm_stat | awk -v ps="$PAGESIZE" '
    /Pages free/        {gsub(/\./,"",$3); free=$3}
    /Pages inactive/    {gsub(/\./,"",$3); inactive=$3}
    /Pages speculative/ {gsub(/\./,"",$3); spec=$3}
    END {printf "%d", (free+inactive+spec)*ps/1048576}
  ')"
  if [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
    {
      echo "watchdog: free=${FREE_MB}MB below ${MIN_FREE_MB}MB; killing runner"
      vm_stat | head -8
      ps -o pid,rss,comm -p "$PID" || true
    } | tee -a "$LOG"
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    exit 3
  fi
  sleep 10
done

wait "$PID"
echo "run complete tag=$TAG rows=$ROWS preds=$PREDS" | tee -a "$LOG"
