#!/bin/bash
# Overnight training for TinyDoc-VLM on M4
# Generates 3K docs, trains ~17K steps (~16 hours)

cd "$(dirname "$0")/.."

echo "=== TinyDoc-VLM Overnight Training ==="
echo "Start: $(date)"
echo "Steps: 17000"
echo "Batch: 1 × 4 grad_accum = 4 effective"
echo "Data: 3000 synthetic docs"
echo "ETA: ~16 hours"
echo ""

# Generate 3K docs first (~3 min)
if [ ! -f data/synthetic/overnight_output/manifest.jsonl ]; then
    echo "Generating 3000 synthetic documents..."
    python3 data/synthetic/generator.py --num-docs 3000 --output-dir data/synthetic/overnight_output
    echo "Done. Data ready."
    echo ""
fi

# Count QA pairs
QA_COUNT=$(python3 -c "
import json
count = 0
with open('data/synthetic/overnight_output/manifest.jsonl') as f:
    for line in f:
        item = json.loads(line)
        count += len(item.get('qa_pairs', []))
print(count)
")
echo "Training data: $QA_COUNT QA pairs"
echo ""

# Launch training
echo "Starting training..."
PYTHONUNBUFFERED=1 python3 -u training/overnight_train.py \
    --steps 17000 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 2e-4 \
    --warmup 500 \
    --num-docs 3000 \
    --lora-rank 16 \
    --output-dir checkpoints/overnight \
    2>&1 | tee training/overnight_log.txt

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Checkpoints: checkpoints/overnight/"
echo "Best: checkpoints/overnight/best/"
echo "Final: checkpoints/overnight/final/"
echo "Log: training/overnight_log.txt"
