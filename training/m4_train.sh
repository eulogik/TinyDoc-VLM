#!/bin/bash
# Quick training launcher for M4 Mac
# Usage: bash training/m4_train.sh [steps]

STEPS=${1:-5000}

echo "=== TinyDoc-VLM M4 Training ==="
echo "Steps: $STEPS"
echo "Device: mps"
echo "Time estimate: ~$(echo "$STEPS / 1800" | bc) hours"
echo ""

# Generate 500 docs if needed
if [ ! -f data/synthetic/m4_output/manifest.jsonl ]; then
    echo "Generating 500 synthetic documents..."
    python3 data/synthetic/generator.py --num-docs 500 --output-dir data/synthetic/m4_output
fi

# Train
echo "Starting LoRA training..."
python3 training/fast_train.py \
    --manifest data/synthetic/m4_output/manifest.jsonl \
    --data-root data/synthetic \
    --steps $STEPS \
    --batch-size 2 \
    --grad-accum 4 \
    --lr 2e-4 \
    --warmup 200 \
    --max-samples 50000 \
    --lora-rank 16 \
    --output-dir checkpoints/lora_m4 \
    --device mps

echo ""
echo "Done! Checkpoint saved to checkpoints/lora_m4/final"
