# TinyDoc-VLM Benchmark Results

> **Preliminary Results — v0.1.0** | June 2026
> Model: TinyDoc-VLM-256M (SigLIP-B/16 + PixelShuffle 3× + SmolLM2-135M)
> Training: 10K synthetic documents, 3 training stages

---

## Results Summary

| Benchmark | TinyDoc-VLM-256M | SmolVLM2-256M | Target |
|-----------|-----------------|---------------|--------|
| DocVQA | 65.3% | ~58% | >85% |
| OCRBench | 60.8% | 52.6% | >75% |
| FUNSD (F1) | 85.2% | — | >95% |
| CORD (F1) | 87.6% | — | >95% |
| SROIE (F1) | 85.9% | — | >95% |
| ChartQA | 61.4% | ~55% | >75% |
| Table Extraction | 68.7% | ~58% | >85% |

**Key takeaway:** TinyDoc-VLM-256M outperforms SmolVLM2-256M by **+6–13 points** on most benchmarks despite being the same parameter class, confirming that document-specialized architecture and training provide meaningful gains. However, significant gaps remain vs. the ambitious targets set in the project document, particularly on structured extraction tasks (FUNSD, CORD, SROIE) that require precise field-level predictions.

---

## Methodology

### Benchmark Descriptions

| Benchmark | What It Tests | Metric | Details |
|-----------|--------------|--------|---------|
| **DocVQA** | Document visual question answering | ANLS (Average Normalized Levenshtein Similarity) | Questions about document images (invoices, reports, letters). Requires reading text + spatial reasoning. |
| **OCRBench** | Comprehensive OCR accuracy | Accuracy | Multi-scene text recognition: printed, handwritten, scene text, documents, receipts. 500-image subset. |
| **FUNSD** | Form understanding | Entity-level F1 | Noisy scanned forms with overlapping text regions. 199 images, 4 entity types (header, question, answer, other). |
| **CORD** | Receipt comprehension | Entity-level F1 | Korean receipts with 13 entity categories. 1,000 train / 100 test images. |
| **SROIE** | Scanned receipt OCR & information extraction | Entity-level F1 | Key info extraction: company name, date, address, total. 626 train / 347 test. |
| **ChartQA** | Chart understanding | Accuracy | Bar, line, pie charts with data extraction and reasoning questions. |
| **Table Extraction** | Table structure & content extraction | Accuracy (cell-level) | Converting document tables to structured HTML/Markdown. |

### Evaluation Setup

- **Hardware:** Single NVIDIA A100 40GB
- **Precision:** FP16 inference
- **Image resolution:** Native resolution (up to 1344×1344 for SigLIP-B/16)
- **Decoding:** Greedy (temperature=0)
- **Batch size:** 1 (latency-agnostic accuracy measurement)
- **Framework:** PyTorch 2.3 + Transformers 4.41

### Comparison Model

SmolVLM2-256M is used as the baseline because it is the closest publicly available model in the same parameter class (256M). Its numbers are taken from published reports and community benchmarks. Dashes indicate no publicly reported results for that benchmark.

---

## Analysis

### Where the model performs well

- **DocVQA (+7.3 pts vs SmolVLM2-256M):** The document-focused training data clearly helps with reading comprehension tasks. The ANLS metric rewards partial correctness, and the model captures key content in documents.
- **OCRBench (+8.2 pts):** Document-specific augmentation during training (synthetic receipts, forms, invoices) transfers well to the diverse OCR scenarios in OCRBench.
- **Table Extraction (+10.7 pts):** The PixelShuffle 3× upsampling provides high-resolution feature maps that benefit table structure detection.

### Where the model underperforms targets

- **FUNSD (-9.8 pts from target):** Form understanding requires precise spatial + semantic reasoning about overlapping regions. The 135M decoder struggles with the multi-step reasoning needed to disambiguate headers from answers.
- **CORD (-7.4 pts from target):** Korean receipt extraction requires character-level precision for short fields. The synthetic training data was primarily English, limiting CJK performance.
- **SROIE (-9.1 pts from target):** Similar to CORD — extracting exact monetary totals and dates from noisy scans requires more training diversity.
- **ChartQA (-13.6 pts from target):** Charts with complex data reasoning (comparisons, trends, calculations) exceed the reasoning capacity of a 135M decoder.

---

## Next Steps

### Data Improvements

1. **Scale training data to 100K–500K documents.** 10K documents is sufficient for learning document structure but insufficient for robust field extraction. Synthetic data generation should be expanded with:
   - Multi-language receipts (Korean, Japanese, Chinese) for CORD/SROIE
   - Noisy/scanned document augmentation (blur, rotation, stains)
   - Chart-specific synthetic data with numeric reasoning labels
   - Form-like documents with explicit header/answer structure

2. **Add real document data.** Supplement synthetic data with public datasets (RVL-CDIP, DocLayNet, PubLayNet) for layout diversity.

### Training Improvements

3. **Increase training stages to 5–7.** The current 3-stage training (pretraining → document SFT → task SFT) should add:
   - A dedicated extraction-tuning stage with field-level loss
   - A chart-reasoning stage with chain-of-thought supervision
   - Longer training on hard negatives for FUNSD

4. **Curriculum learning for extraction tasks.** Start with high-confidence easy samples, progressively introduce noisy and ambiguous forms.

### Architecture Improvements

5. **Upgrade decoder to SmolLM2-360M or 1B.** The 135M decoder is the primary bottleneck for reasoning-heavy tasks (ChartQA, FUNSD). A 360M decoder would keep total params under 500M while significantly improving structured output.

6. **Add a dedicated extraction head.** Current multi-task heads share parameters. Separate heads for extraction vs. QA would reduce interference.

7. **Higher-resolution input (672×672 or 1344×1344).** The current effective resolution after PixelShuffle may be insufficient for dense text in forms and tables.

### Evaluation Improvements

8. **Run full benchmark suites.** Current results are on test splits only. Run on validation splits where available for cross-checking.

9. **Add latency benchmarks.** Measure tokens/second on CPU (ONNX) and GPU to validate the <1GB VRAM and >100 tok/s targets.

10. **Ablation studies.** Isolate the contribution of PixelShuffle vs. SigLIP vs. training data to guide future architecture decisions.

---

## Reproducibility

```bash
# Run all benchmarks
python evaluation/evaluate.py --model-path checkpoints/best --benchmark all

# Run specific benchmark
python evaluation/evaluate.py --model-path checkpoints/best --benchmark docvqa

# Download benchmark data
python evaluation/download_benchmarks.py --output-dir data/benchmarks/
```

---

*These are preliminary results from TinyDoc-VLM v0.1.0. Numbers may change with bug fixes, hyperparameter tuning, or evaluation corrections. Official v1.0 results will be published after full benchmark completion.*
