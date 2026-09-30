#!/usr/bin/env bash
# Score banked LoRA checkpoints on the VAL split (40 docs) for checkpoint
# selection. NEVER scores test here: the clean-100 is reported exactly once,
# for the selected adapter. Run AFTER training completes (MLX inference needs
# the GPU to itself; ~3 min per checkpoint).
#
# Usage: bash scripts/score_checkpoints_val.sh "350 400 450 500 ..."
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${SCRATCH:-/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode}"
DATA_ROOT="${DATA_ROOT:-/Volumes/KIOXIA 1TB/tinydoc}"
VENV="$SCRATCH/venv_mlx"
BANKED="${BANKED:-$DATA_ROOT/adapters/full/banked}"
MODEL="${MODEL:-$DATA_ROOT/models/smolvlm500m}"
# out-suffix prefix per stage (Stage-1 _valg* names are taken by committed files)
VALPREFIX="${VALPREFIX:-_valg}"

for g in $1; do
  ckpt="$BANKED/global_${g}_adapters.safetensors"
  [ -f "$ckpt" ] || { echo "missing $ckpt — skipping"; continue; }
  if [ -f "$REPO/evaluation/phase0/results/scores_smolvlm500m_lora${VALPREFIX}$g.json" ]; then
    echo "global_$g already scored — skipping"
    continue
  fi
  # load_model(adapter_path=dir) needs dir/adapter_config.json + dir/adapters.safetensors
  STAGE="$SCRATCH/valstage/global_$g"
  mkdir -p "$STAGE"
  ln -sf "$(dirname "$ckpt")/../adapter_config.json" "$STAGE/adapter_config.json"
  ln -sf "$ckpt" "$STAGE/adapters.safetensors"
  echo "=== val-scoring global_$g ==="
  "$VENV/bin/python" "$REPO/evaluation/phase0/run_mlx_adapter_eval.py" \
    --model "$MODEL" \
    ${EVAL_RESIZE:+--image-resize $EVAL_RESIZE} \
    --adapter "$STAGE" \
    --eval-path "$REPO/evaluation/phase0/results/sroie_val.json" \
    --out-suffix "${VALPREFIX}$g" --max-tokens 256
done
echo "--- val F1 summary ---"
for g in $1; do
  f="$REPO/evaluation/phase0/results/scores_smolvlm500m_lora${VALPREFIX}$g.json"
  [ -f "$f" ] && python3 -c "import json;d=json.load(open('$f'));print('global_$g val_f1=%.4f schema=%.3f lat=%dms' % (d['field_f1'], d['schema_valid_rate'], d['avg_latency_ms']))"
done
