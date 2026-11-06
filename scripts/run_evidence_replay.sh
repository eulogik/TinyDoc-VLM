#!/usr/bin/env bash
# Replay the OCR/evidence stage on saved clean-100 predictions (no VLM calls).
# Usage: scripts/run_evidence_replay.sh [ocr_venv]
# Requires the RapidOCR venv; see AGENTS.md for the rebuild note.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

VENV="${1:-/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/ocr_venv}"
RESULTS="evaluation/phase0/results"
mkdir -p logs

run_one() {
  local preds="$1" out="$2" log="$3"
  echo "START $preds $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  "$VENV/bin/python" evaluation/phase0/attach_evidence_from_preds.py \
    --preds "$preds" --out "$out" >> "$log" 2>&1 &
  local pid=$!
  scripts/watch_mem.sh "$pid" 6 5 &
  local wpid=$!
  wait "$pid"; local rc=$?
  kill "$wpid" 2>/dev/null || true
  echo "END rc=$rc $(date '+%Y-%m-%d %H:%M:%S')" >> "$log"
  return "$rc"
}

run_one "$RESULTS/preds_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.jsonl" \
  "$RESULTS/preds_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k_evidence.jsonl" \
  "logs/evidence_replay_qwen2.5vl_3b.log"
r1=$?
run_one "$RESULTS/preds_pipeline_ollama_qwen3-vl_2b-instruct_clean_ctx8k.jsonl" \
  "$RESULTS/preds_pipeline_ollama_qwen3-vl_2b-instruct_clean_ctx8k_evidence.jsonl" \
  "logs/evidence_replay_qwen3-vl_2b.log"
r2=$?

echo "DONE 3b=$r1 2b=$r2"
exit $(( r1 || r2 ))
