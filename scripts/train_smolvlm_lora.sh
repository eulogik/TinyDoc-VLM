#!/usr/bin/env bash
# Stage-1: LoRA SFT of SmolVLM-500M for char-exact receipt extraction (M4/MLX).
#
# Trains ONLY on the leakage-filtered SROIE train split (420 docs, whole duplicate
# components assigned to eval/val/train so no near-duplicate crosses splits) with
# eval prompt (EXTRACT_PROMPT) as the question and compact 4-key JSON as the
# answer. Vision tower frozen; completion-only loss; watchdog-guarded.
#
# Usage:
#   bash scripts/train_smolvlm_lora.sh smoke   # ~10 iters: validates + times a step
#   bash scripts/train_smolvlm_lora.sh full    # the real run (needs approval)
#
# The dataset carries a prebuilt `messages` column (user turn = EXTRACT_PROMPT with
# the image, assistant turn = compact 4-key JSON), so no --custom-prompt-format:
# the 0.7.3 formatter emits a dict, which its own VisionDataset cannot consume.
set -euo pipefail

MODE="${1:-smoke}"
SCRATCH="/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode"
VENV="$SCRATCH/venv_mlx"
DATA="$SCRATCH/stage1_data"
MODEL="$SCRATCH/models/smolvlm500m"
ADAPTER_DIR="$SCRATCH/adapters"
mkdir -p "$ADAPTER_DIR"

case "$MODE" in
  smoke) ITERS=10; OUT="$ADAPTER_DIR/smoke"; RESUME="" ;;
  full)  ITERS=1200; OUT="$ADAPTER_DIR/full"; RESUME="" ;;
  resume)
    # wave-tolerance: warm-start from the latest checkpoint after a watchdog
    # kill. $2 = iters already banked (read from the last "Iter N" save line).
    # Optimizer state resets (standard); ≤1 save-interval of overlap re-trains.
    DONE_ITERS="${2:?usage: $0 resume <done_iters>}"
    ITERS=$((1200 - DONE_ITERS)); OUT="$ADAPTER_DIR/full"; RESUME="--adapter-path $OUT"
    # archive prior segment's numbered saves BEFORE they get overwritten:
    # filenames are segment-relative, so tag them with global iters now.
    mkdir -p "$OUT/banked"
    PREV_START=$(cat "$OUT/banked/CURRENT_START" 2>/dev/null || echo 0)
    for f in "$OUT"/0000*_adapters.safetensors; do
      [ -e "$f" ] || continue
      nnn=$(basename "$f" | grep -oE '^[0-9]+')
      g=$((PREV_START + 10#$nnn))
      mv "$f" "$OUT/banked/global_${g}_adapters.safetensors"
      echo "global_${g} <= segment(start=$PREV_START)+seg-iter $((10#$nnn)) [auto-archived $(date -u +%FT%TZ)]" >> "$OUT/banked/MAPPING.txt"
    done
    echo "$DONE_ITERS" > "$OUT/banked/CURRENT_START" ;;
  *) echo "usage: $0 [smoke|full|resume <done_iters>]"; exit 2 ;;
esac
mkdir -p "$OUT"

# Ship-bar checkpointing: save every 200 iters so a crash never loses the run,
# and evaluate on the clean val split periodically via steps-per-eval.
exec "$VENV/bin/python" -m mlx_vlm.lora \
  --model-path "$MODEL" \
  $RESUME \
  --dataset "$DATA/hf_train" \
  --split train \
  --lora-rank 16 \
  --lora-alpha 32 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --grad-checkpoint \
  --max-seq-length 1024 \
  --train-on-completions \
  --iters "$ITERS" \
  --steps-per-report 10 \
  --steps-per-eval 200 \
  --val-batches 5 \
  --steps-per-eval 200 \
  --val-batches 5 \
  --steps-per-save 50 \
  --output-path "$OUT"
