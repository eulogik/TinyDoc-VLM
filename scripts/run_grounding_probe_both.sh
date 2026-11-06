#!/usr/bin/env bash
# Run the grounding probe for the fast tier and the quality tier.
# Usage: scripts/run_grounding_probe_both.sh [manifest] [limit]
# Inference is long-running; watch_mem.sh guards system memory.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
export TINYDOC_OLLAMA_TIMEOUT="${TINYDOC_OLLAMA_TIMEOUT:-600}"
export TINYDOC_OLLAMA_NUM_CTX="${TINYDOC_OLLAMA_NUM_CTX:-8192}"

MANIFEST="${1:-/Volumes/KIOXIA 1TB/tinydoc/grounding_probe_20261106/manifest.jsonl}"
LIMIT="${2:-100}"
RESULTS="evaluation/phase0/results"
mkdir -p logs

run_one() {
  local model="$1" out="$2" log="$3"
  echo "START $model $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  python3 evaluation/phase0/run_grounding_probe.py \
    --manifest "$MANIFEST" --model "$model" --limit "$LIMIT" --out "$out" >> "$log" 2>&1 &
  local pid=$!
  scripts/watch_mem.sh "$pid" 8 5 &
  local wpid=$!
  wait "$pid"; local rc=$?
  kill "$wpid" 2>/dev/null || true
  echo "END $model rc=$rc $(date '+%Y-%m-%d %H:%M:%S')" >> "$log"
  return "$rc"
}

run_one qwen3-vl:2b-instruct \
  "$RESULTS/grounding_probe_qwen3-vl_2b-instruct.json" \
  "logs/grounding_qwen3-vl_2b-instruct.log"
r1=$?
run_one qwen2.5vl:3b \
  "$RESULTS/grounding_probe_qwen2.5vl_3b.json" \
  "logs/grounding_qwen2.5vl_3b.log"
r2=$?

echo "DONE 2b=$r1 3b=$r2"
exit $(( r1 || r2 ))
