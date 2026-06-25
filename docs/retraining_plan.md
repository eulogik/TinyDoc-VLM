# Retraining Plan — From Scratch

> **Date**: 2026-06-25
> **Current state**: Model is non-functional (0% on OCRBench). Needs proper training at scale.

## What Went Wrong

1. **Training data too small**: 10K synthetic docs vs. 11.1M called for in project plan (1000× gap)
2. **Wrong prompt format for eval**: Model trained on extraction format, evaluated with QA format
3. **image_token_id bug**: Processor and model had different token IDs, preventing visual features from being used at all
4. **No real benchmark training**: Model never saw DocVQA, FUNSD, CORD, SROIE, or OCRBench during training

## What Needs to Happen

### Phase 1: Data (Week 1-2)

**Synthetic Data at Scale**
- Generate 1M+ synthetic documents (not 10K)
- Use existing `data/synthetic/generator.py` but run with `--num-docs 1000000`
- Distribute across 10 document types (invoice, receipt, form, table, etc.)
- Apply augmentation (rotation, noise, blur, grayscale, JPEG artifacts)
- Store on HF Hub as `eulogik/TinyDoc-VLM-Data`

**Public Dataset Integration**
- DocVQA (50K questions, 12K docs) — HF: `lmms-lab/docvqa`
- FUNSD (199 docs, 9.7K entities) — HF: `nielsr/funsd`
- CORD (1,000 receipts) — HF: `naver-clova-ix/cord-v2`
- SROIE (626 receipts) — HF: `rajistics/sroie`
- PubTabNet (568K tables) — needs custom loader
- RVL-CDIP (400K document images) — for layout pretraining

**QA Pair Generation**
- For each document, generate 5-10 question-answer pairs
- Questions should match benchmark style: "What is the total?", "Who is the vendor?", "Convert this table to JSON"
- Train with both extraction AND QA format

### Phase 2: Training (Week 3-6)

**Stage 1: Layout Pretraining (1M synthetic docs)**
- Task: Learn to "see" document structure
- Input: document image → output: structured JSON with all visible fields
- Loss: L_layout + L_ocr + L_region
- Estimated time: 2 days on 1× A100

**Stage 2: Document Understanding (public datasets)**
- Task: QA + extraction on real documents
- Data: DocVQA + FUNSD + CORD + SROIE + PubTabNet
- Loss: L_answer + L_json + L_kv + L_table
- Estimated time: 1 day on 1× A100

**Stage 3: Instruction Tuning (50K conversations)**
- Task: Multi-turn document conversations
- Data: Synthetic conversations from Stage 1 docs + public QA pairs
- Format: `[{"from": "human", "value": "<image>\nWhat is X?"}, {"from": "gpt", "value": "X is Y"}]`
- Loss: L_lm (standard language modeling)
- Estimated time: 4 hours on 1× A100

### Phase 3: Evaluation (Week 7)

**Run real eval on all benchmarks:**
```bash
python evaluation/evaluate.py --model-path checkpoints/stage3/best --benchmark ocrbench
python evaluation/evaluate.py --model-path checkpoints/stage3/best --benchmark funsd
python evaluation/evaluate.py --model-path checkpoints/stage3/best --benchmark cord
python evaluation/evaluate.py --model-path checkpoints/stage3/best --benchmark sroie
```

**Expected results after proper training:**
- DocVQA: 65-75% (competitive with SmolVLM2-500M)
- OCRBench: 55-65%
- FUNSD: 80-90% F1
- CORD: 85-92% F1
- SROIE: 82-90% F1

### Phase 4: Publication (Week 8-12)

- Update model card with real benchmarks
- Submit to OpenRouter
- Post marketing materials (HN, Reddit, Twitter)
- Submit paper to NeurIPS 2027 or CVPR 2027

## Infrastructure Needed

| Resource | Spec | Est. Cost |
|----------|------|-----------|
| GPU | 1× A100 80GB (or cloud equivalent) | $2-3/hr |
| Storage | 50GB for training data | $5/month |
| HF Hub | For model + data hosting | Free |
| **Total (4 weeks training)** | | **~$500-1000** |

## Immediate Next Steps

1. ✅ Fix image_token_id bug (done)
2. ✅ Generate 10K synthetic docs (done, but need 1M+)
3. ⬜ Scale generation to 1M+ docs
4. ⬜ Format and upload training data to HF Hub
5. ⬜ Set up training on Colab (free T4) or paid GPU
6. ⬜ Run Stage 1 training
7. ⬜ Run Stage 2 training on public data
8. ⬜ Run Stage 3 instruction tuning
9. ⬜ Evaluate on real benchmarks
10. ⬜ Publish results

## Colab Training Notebook

The existing `training/tinydoc_colab_training.ipynb` needs to be updated:
- Load data from HF Hub instead of generating locally
- Add public dataset loaders for Stage 2
- Support A100 if available (pro/colab+)
- Save checkpoints to HF Hub
- Add evaluation cells

## Key Files to Modify

| File | What to Change |
|------|---------------|
| `training/tinydoc_colab_training.ipynb` | Scale data loading, add public datasets |
| `data/datasets/unified.py` | Add loaders for DocVQA, FUNSD, CORD, SROIE |
| `evaluation/evaluate.py` | Fix prompt format to match training |
| `tinydoc_vlm/losses.py` | Add QA-specific loss terms |
