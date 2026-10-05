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

# Split-disk layout (2026-09-30): cleaners/purges eat scratch, so only the
# regenerable venv + logs live there. Everything slow or irreplaceable
# (weights, datasets, stage1 splits, adapters) lives under DATA_ROOT, which
# defaults to KIOXIA (persistent, reboot-safe). Override via environment.
SCRATCH="${SCRATCH:-/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode}"
DATA_ROOT="${DATA_ROOT:-/Volumes/KIOXIA 1TB/tinydoc}"
VENV="$SCRATCH/venv_mlx"
DATA="$DATA_ROOT/stage1_data"
MODEL="${MODEL:-$DATA_ROOT/models/smolvlm500m}"
ADAPTER_DIR="$DATA_ROOT/adapters"
SROIE_LABELS="$DATA_ROOT/data/sroie-labels"
mkdir -p "$ADAPTER_DIR"

MODE="${1:-smoke}"
# Hyperparams are env-overridable; defaults = Stage-1 values.
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_DROPOUT="${TRAIN_DROPOUT:-0.0}"
TRAIN_DATA="${TRAIN_DATA:-$DATA/hf_train}"
case "$MODE" in
  smoke) ITERS=10; OUT="$ADAPTER_DIR/smoke"; RESUME="" ;;
  full) ITERS=1200; OUT="$ADAPTER_DIR/full"; RESUME="" ;;
  # s2: Stage-2 twin-mirror (2 epochs over 1920 docs @ eff.batch 8 = 480 iters)
  # Explicit TRAIN_* env wins; else S2_* stage defaults; else Stage-1 values.
  s2) ITERS="${S2_ITERS:-480}"; OUT="${S2_OUT:-$ADAPTER_DIR/stage2a}"; RESUME="" ;
      TRAIN_LR="${TRAIN_LR:-${S2_LR:-1e-4}}"; TRAIN_DROPOUT="${TRAIN_DROPOUT:-${S2_DROPOUT:-0.05}}" ;
      TRAIN_DATA="${TRAIN_DATA:-${S2_DATA:-$DATA_ROOT/stage2_data/hf_train2/data}}" ;;
  resume)
    # wave-tolerance: warm-start from the latest checkpoint after a watchdog
    # kill. $2 = iters already banked (read from the last "Iter N" save line).
    # Optimizer state resets (standard); ≤1 save-interval of overlap re-trains.
    DONE_ITERS="${2:?usage: $0 resume <done_iters>}"
    RTARGET="${RESUME_TARGET:-1200}"; ROUT="${RESUME_OUT:-$ADAPTER_DIR/full}"
    ITERS=$((RTARGET - DONE_ITERS)); OUT="$ROUT"; RESUME=1
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
  *) echo "usage: $0 [smoke|full|s2|resume <done_iters>]"; exit 2 ;;
esac
mkdir -p "$OUT"

# Ship-bar checkpointing: save every 200 iters so a crash never loses the run,
# and evaluate on the clean val split periodically via steps-per-eval.
# Build argv with set -- (POSIX, bash-3.2/set -u safe: never empty, always
# quoted). Paths may contain spaces (KIOXIA) — never interpolate unquoted.
# Optional resize pre-processing: TRAIN_RESIZE_ARGS="--image-resize-shape 512 512"
# (aspect-preserving fit-inside-box; verified in mlx_vlm/utils.resize_image).
# Intentionally UNQUOTED at use (expands to zero args when empty, N args when set).
TRAIN_RESIZE_ARGS="${TRAIN_RESIZE_ARGS:-}"
TRAIN_MAXSEQ="${TRAIN_MAXSEQ:-1024}"
set -- "$VENV/bin/python" -m mlx_vlm.lora \
  --model-path "$MODEL"
if [ -n "$RESUME" ]; then
  set -- "$@" --adapter-path "$OUT"
fi
set -- "$@" \
  --dataset "$TRAIN_DATA" \
  --split train \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout "$TRAIN_DROPOUT" \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate "$TRAIN_LR" \
  --grad-checkpoint \
  --max-seq-length "$TRAIN_MAXSEQ" \
  --train-on-completions \
  --iters "$ITERS" \
  --steps-per-report 10 \
  --steps-per-eval 200 \
  --val-batches 5 \
  --steps-per-eval 200 \
  --val-batches 5 \
  --steps-per-save 25 \
  --output-path "$OUT"
set -- "$@" $TRAIN_RESIZE_ARGS
exec "$@"
