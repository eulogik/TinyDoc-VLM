
# TinyDoc-VLM: The World's Smallest Document Understanding Model
## Project Document — v1.0 | June 2026

---

## 1. Executive Summary

**TinyDoc-VLM** is a 150M–300M parameter vision-language model trained **exclusively** for document understanding — invoices, receipts, forms, tables, charts, ID cards, and business documents. It is not a generalist VLM. It is a document specialist.

**Why this matters:**
- No competitor exists in the <500M parameter document-specialized VLM space
- SmolVLM2-256M (generalist) gets only 52.6% on OCRBench — a doc-specialist could hit 80%+
- Document AI is a $10B+ market with massive enterprise demand
- Runs on Raspberry Pi (<1GB VRAM target)
- Natural fit for OpenRouter's rising tool-calling ecosystem

**Target:** Dominate HuggingFace downloads and OpenRouter usage in the document AI category within 6 months of release.

---

## 2. Market Analysis & Competitive Landscape

### 2.1 Current Tiny VLM Leaders (Generalist)

| Model | Params | DocVQA | OCRBench | VRAM | Type |
|-------|--------|--------|----------|------|------|
| **SmolVLM2-256M** | 256M | ~58% | 52.6% | 0.8GB | Generalist |
| **SmolVLM2-500M** | 500M | 70.5% | ~65% | 1.2GB | Generalist |
| **SmolVLM2-2.2B** | 2.2B | 80.0% | ~75% | 4.9GB | Generalist |
| **Qwen2.5-VL-3B** | 3B | ~85% | ~70% | 6GB | Generalist |
| **InternVL-3.5-1B** | 1B | ~75% | ~60% | 3GB | Generalist |
| **Moondream2** | ~1.6B | ~55% | ~50% | 2GB | Generalist |

**Key Insight:** All existing tiny VLMs are **generalists**. They do image captioning, VQA, OCR, and document understanding with the same model. This is inefficient — a document specialist can achieve 2-3x better document performance at 1/2 the size.

### 2.2 Document AI Market

- **Rossum** (AI document processing): $100M+ ARR
- **Docsumo** (document extraction): $50M+ ARR
- **Nanonets** (OCR + extraction): $30M+ ARR
- **Hyperscience** (document AI): $100M+ ARR, claims 99% accuracy with human-in-the-loop
- **Traditional OCR** (Tesseract, AWS Textract, Google Document AI): Expensive per-page pricing

**Gap:** All these solutions are either:
1. Cloud-based APIs (expensive, privacy concerns)
2. Large models requiring significant compute
3. Rule-based systems requiring templates

**TinyDoc-VLM fills the gap:** Local, tiny, accurate, no templates needed.

### 2.3 Benchmark Targets

| Benchmark | Current SOTA (tiny) | TinyDoc-VLM Target | Notes |
|-----------|---------------------|-------------------|-------|
| **DocVQA** | 70.5% (SmolVLM2-500M) | **>85%** | Document visual QA |
| **OCRBench** | 52.6% (SmolVLM2-256M) | **>75%** | Comprehensive OCR |
| **FUNSD** | 93.76% (LayoutLMv3-Base) | **>95% F1** | Form understanding |
| **CORD** | 97.23% (LayoutLMv3-Base) | **>95% F1** | Receipt extraction |
| **SROIE** | 96.5% (LayoutLMv3) | **>95% F1** | Receipt key info |
| **DocVQA-Code** | 54.0% (Qwen 3.5 Omni) | **>60%** | Code screenshots |
| **ChartQA** | ~65% (SmolVLM2-500M) | **>75%** | Chart understanding |
| **Table Extraction** | ~70% (SmolVLM2-500M) | **>85%** | HTML/Markdown tables |
| **VRAM Usage** | 0.8GB (SmolVLM2-256M) | **<1GB** | Edge deployment |
| **Inference Speed** | ~50 tok/s | **>100 tok/s** | CPU with ONNX |

---

## 3. Architecture Design

### 3.1 Core Philosophy

**"Balance, not brute force."**

SmolVLM research proved that smaller LMs get little benefit from huge encoders. For a 135M LM, pairing with a 93M SigLIP-B/16 is optimal — a 400M encoder only improves performance by 11.6% but increases parameters by 66%.

TinyDoc-VLM pushes this further: **document-specific architecture optimizations** that generalist VLMs cannot afford.

### 3.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TinyDoc-VLM Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  Input: Document Image (up to 1024x1024, variable aspect)   │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────┐     │
│  │ Layout-Aware    │    │ Document-Specific           │     │
│  │ Vision Encoder  │───▶│ Token Compressor            │     │
│  │ (~50M params)   │    │ (Pixel Shuffle + MLP)       │     │
│  └─────────────────┘    └─────────────────────────────┘     │
│         │                           │                       │
│         │    ┌──────────────────────┘                       │
│         │    │                                              │
│         ▼    ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Document Layout Tokens (learned positional)        │   │
│  │  + Visual Tokens + Text Prompt Tokens               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tiny Decoder with Document-Aware Attention         │   │
│  │  (~150M params, 8 layers, GQA, RoPE, SwiGLU)        │   │
│  │                                                      │   │
│  │  • Layout-preserving cross-attention                 │   │
│  │  • Region-of-interest masking                        │   │
│  │  • Structured output heads (JSON, KV, Table)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  Output: Structured JSON / Key-Value / Table / Text          │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Component Details

#### A. Layout-Aware Vision Encoder (~50M parameters)

**Base:** Custom tiny ViT inspired by SigLIP-B/16 but optimized for documents.

| Feature | Specification | Rationale |
|---------|--------------|-----------|
| **Architecture** | 12-layer ViT, 384 hidden dim, 6 heads | Balance of capacity and speed |
| **Patch Size** | 16×16 (default) + 8×8 (high-res mode) | Documents need fine-grained text |
| **Input Resolution** | 512×512 (default), 1024×1024 (high-res) | A4 documents at 150-300 DPI |
| **Aspect Ratio** | Variable (padding, not stretching) | Preserve document proportions |
| **Special Tokens** | [PAGE_START], [PAGE_END], [REGION] | Document structure awareness |
| **Position Encoding** | 2D sinusoidal + learned layout embeddings | Preserve spatial relationships |
| **Pretraining** | Document image-text contrastive learning | Unlike general CLIP/SigLIP |

**Key Innovation:** **Region-aware attention** — the encoder attends differently to text regions, table regions, image regions, and whitespace. This is learned during pretraining on document layout detection tasks.

#### B. Document Token Compressor

**Problem:** A 1024×1024 image at 16×16 patches = 4096 tokens. Too many for a tiny decoder.

**Solution:** Multi-stage compression:

1. **Pixel Shuffle (space-to-depth):** 4×4 shuffle → 16× reduction (4096 → 256 tokens)
2. **Document-aware pooling:** Merge tokens from same text line/table cell
3. **MLP projection:** 256 → 128 dim per token
4. **Layout token injection:** Add [TABLE], [HEADER], [FOOTER], etc. tokens

**Result:** 1024×1024 document → ~200-300 visual tokens (vs 4096 for naive approach)

#### C. Tiny Decoder (~150M parameters)

**Base:** Custom transformer, not a repurposed LLM.

| Feature | Specification | Rationale |
|---------|--------------|-----------|
| **Layers** | 8 | Deep enough for document reasoning |
| **Hidden Dim** | 768 | Matches visual token dim |
| **Heads** | 12 (GQA: 4 KV groups) | Efficient attention |
| **Context Length** | 8K tokens | Fits multiple document pages |
| **Position Encoding** | RoPE (base 273k, like SmolVLM) | Long context support |
| **Activation** | SwiGLU | Modern, efficient |
| **Norm** | RMSNorm | Stable training |
| **Vocab** | 32K (document-specialized) | JSON tokens, currency, dates, etc. |

**Key Innovation:** **Structured Output Heads**

Instead of generating free-form text, the decoder has multiple output heads:
- **JSON Head:** Generates valid JSON with schema constraints
- **KV Head:** Key-value pair extraction with confidence scores
- **Table Head:** HTML/Markdown table generation
- **OCR Head:** Plain text with reading order
- **QA Head:** Natural language answers

Each head is a small MLP on top of the final hidden states, trained with task-specific objectives.

#### D. Document-Aware Attention Mechanisms

1. **Layout-Preserving Cross-Attention:**
   - Visual tokens attend to each other with spatial bias (closer tokens attend more)
   - Implemented via 2D relative position bias in attention scores

2. **Region-of-Interest Masking:**
   - During training, mask random regions and ask model to reconstruct
   - Forces model to understand document structure, not just memorize

3. **Hierarchical Attention:**
   - First attend to page-level structure (headers, footers, columns)
   - Then attend to region-level (tables, paragraphs, lists)
   - Finally attend to token-level (words, numbers, symbols)

---

## 4. Training Strategy

### 4.1 Data Pipeline

#### Stage 1: Pretraining Data (10M+ synthetic document images)

**Synthetic Document Generation Pipeline:**

Inspired by "SynthID" approach (OCR + LLM + inpainting), but optimized for scale:

```
Real Document Templates (1000+)
    ↓
Layout Analysis (bounding boxes, regions, fonts)
    ↓
LLM-Generated Content (contextually realistic text)
    ↓
Render Engine (HTML/CSS → PDF → Image)
    ↓
Augmentation (rotation, noise, distortion, compression)
    ↓
Perfect Ground Truth JSON (structured data)
```

**Document Types:**
| Type | Count | Key Features |
|------|-------|-------------|
| Invoices | 2M | Line items, totals, tax, vendor info |
| Receipts | 2M | Store info, items, prices, dates |
| Forms | 1.5M | Fields, checkboxes, signatures |
| Tables | 1.5M | Financial, scientific, inventory |
| ID Cards | 1M | Passports, licenses, business cards |
| Charts/Graphs | 1M | Bar, line, pie, financial |
| Contracts | 0.5M | Clauses, parties, dates, amounts |
| Medical Records | 0.5M | Patient info, diagnoses, prescriptions |
| Handwritten | 0.5M | Notes, forms, letters |
| Mixed Layouts | 0.5M | Multi-column, magazines, brochures |

**Data Augmentation:**
- Rotation (±5°)
- Perspective distortion
- Gaussian noise
- JPEG compression artifacts
- Blur (motion, defocus)
- Color shifts (grayscale, sepia, faded)
- Overlapping stamps/signatures
- Watermarks
- Handwritten annotations
- Cropping/partial visibility

#### Stage 2: Fine-tuning Data (1M+ real + synthetic)

**Public Datasets:**
- DocVQA (50K questions, 12K documents)
- FUNSD (199 documents, 9,707 entities)
- CORD (1,000 receipts, 30K words)
- SROIE (626 receipts, 4K key-value pairs)
- ICDAR 2019 (table extraction)
- PubTabNet (568K tables)
- DeepForm (100K forms)
- Kleister Charity (2.7K documents)
- RVL-CDIP (400K document images, 16 classes)

**Synthetic QA Pairs:**
For each document, generate 10-20 question-answer pairs:
- "What is the total amount?"
- "Who is the vendor?"
- "Extract all line items as JSON"
- "What is the due date?"
- "Summarize the table"

#### Stage 3: Instruction Tuning (100K+ conversations)

**Conversation Format:**
```json
{
  "system": "You are TinyDoc-VLM, a document understanding assistant. Extract structured data from documents accurately.",
  "conversations": [
    {
      "from": "human",
      "value": "<image>
Extract all information from this invoice as JSON."
    },
    {
      "from": "gpt",
      "value": "{"vendor": "...", "items": [...], "total": "..."}"
    }
  ]
}
```

**Instruction Types:**
1. **Extraction:** "Extract the following fields: ..."
2. **QA:** "What is the ...?"
3. **Summarization:** "Summarize this document in 3 bullet points"
4. **Comparison:** "Compare the totals in these two invoices"
5. **Validation:** "Is this receipt valid? Check for required fields"
6. **Transformation:** "Convert this table to Markdown"
7. **Translation:** "Extract and translate the text to English"

### 4.2 Training Curriculum

**3-Stage Training:**

```
Stage 1: Layout Pretraining (2 weeks)
├── Data: 10M synthetic documents
├── Task: Layout detection + OCR + region classification
├── Loss: Combined (L_layout + L_ocr + L_region)
├── Objective: Learn to "see" document structure
└── Checkpoints: Every 100K steps

Stage 2: Document Understanding (3 weeks)
├── Data: 1M fine-tuning pairs (DocVQA, FUNSD, CORD, etc.)
├── Task: QA + extraction + table parsing
├── Loss: L_answer + L_json + L_kv + L_table
├── Objective: Learn to "understand" document content
└── Checkpoints: Every 50K steps

Stage 3: Instruction Tuning (1 week)
├── Data: 100K instruction conversations
├── Task: Multi-turn document conversations
├── Loss: L_lm (standard language modeling)
├── Objective: Learn to "converse" about documents
└── Checkpoints: Every 10K steps
```

### 4.3 Training Infrastructure

| Specification | Detail |
|--------------|--------|
| **Compute** | 8× A100 80GB (or 16× A6000 48GB) |
| **Framework** | PyTorch 2.6 + FSDP + FlashAttention-2 |
| **Optimizer** | AdamW (β1=0.9, β2=0.95, eps=1e-8) |
| **Learning Rate** | Stage 1: 1e-4 → 1e-5 (cosine decay) |
| **Batch Size** | 512 (global) = 64 per GPU × 8 GPUs |
| **Precision** | BF16 mixed precision |
| **Gradient Clipping** | 1.0 |
| **Warmup** | 5% of total steps |
| **Regularization** | Dropout 0.1, Weight decay 0.01 |
| **Total Training Time** | ~6 weeks |
| **Cost Estimate** | ~$15,000 (cloud) or ~$5,000 (own hardware) |

---

## 5. Evaluation & Benchmarking

### 5.1 Standard Benchmarks

| Benchmark | Metric | Target | Validation Strategy |
|-----------|--------|--------|---------------------|
| **DocVQA** | ANLS | >85% | Official eval script |
| **OCRBench** | Average | >75% | Official eval (private test set) |
| **FUNSD** | Entity F1 | >95% | Standard split |
| **CORD** | Entity F1 | >95% | Standard split |
| **SROIE** | F1 | >95% | Standard split |
| **ChartQA** | Accuracy | >75% | Human-evaluated |
| **TableExtraction** | TreeEditDist | <0.1 | Custom eval |
| **DocVQA-Code** | Accuracy | >60% | Screenshot reasoning |

### 5.2 Custom Benchmarks

**TinyDoc-Bench (proprietary):**
- 10,000 real-world documents (invoices, receipts, forms, IDs)
- 50,000 extraction tasks
- Multi-language: English, Spanish, French, German, Chinese, Japanese, Arabic
- Multi-format: scanned, photographed, digital PDF, screenshot
- Grading: Exact match + fuzzy match + human evaluation

**Edge Performance Benchmarks:**
| Device | Target Latency | Memory |
|--------|---------------|--------|
| Raspberry Pi 5 | <5s per page | <2GB RAM |
| iPhone 15 | <2s per page | <1GB RAM |
| MacBook Air M2 | <1s per page | <2GB RAM |
| NVIDIA Jetson Nano | <3s per page | <4GB RAM |
| CPU (Intel i5) | <2s per page | <2GB RAM |

### 5.3 Ablation Studies

Critical experiments to run:

1. **Encoder size impact:** 30M vs 50M vs 93M vs 400M encoder with 150M decoder
2. **Pixel shuffle ratio:** 2× vs 4× vs 8× compression
3. **Context length:** 2K vs 4K vs 8K vs 16K tokens
4. **Structured heads vs single head:** JSON+KV+Table+OCR vs unified generation
5. **Synthetic vs real data ratio:** 100% synthetic → 50/50 → 100% real
6. **Multilingual impact:** English-only vs 10 languages vs 50 languages

---

## 6. Distribution & Go-to-Market

### 6.1 HuggingFace Strategy

**Model Card (Critical for Adoption):**
- Extensive benchmark results with comparison tables
- Training details: data sources, mixture proportions, hyperparameters
- Architecture diagram and parameter breakdown
- Memory usage charts (VRAM vs batch size vs sequence length)
- Inference code examples (Python, JavaScript, cURL)
- Fine-tuning guide with LoRA/QLoRA configs
- Limitations and bias analysis

**Spaces Demo:**
- Interactive Gradio demo: upload document → get structured extraction
- Pre-loaded examples: invoice, receipt, form, table, chart, ID
- Side-by-side comparison with SmolVLM2-256M
- Real-time processing (show latency)
- Export results (JSON, CSV, Markdown)

**Integration:**
- `transformers` native support (PR to HF if needed)
- `optimum` ONNX export for CPU inference
- `mlx` support for Apple Silicon
- `llama.cpp` / `ollama` integration for local deployment
- `langchain` / `llamaindex` document loader integration

### 6.2 OpenRouter Strategy

**Model Listing:**
- Name: `tinydoc-vlm-256m` / `tinydoc-vlm-500m`
- Description: "The world's smallest document understanding model. Extract structured data from invoices, receipts, forms, and tables in <1GB VRAM."
- Pricing: **Free tier** (rate limited) + **$0.10/M tokens** (cheapest document VLM)
- Endpoints: `/chat/completions` with image input + `/extract` for structured output

**Tool Calling:**
- Native function calling for document extraction workflows
- Pre-built tools: `extract_invoice`, `extract_receipt`, `parse_table`, `validate_document`
- JSON schema output with validation

### 6.3 Community Building

**Week 1-2:**
- Release model + demo + Python SDK
- Post on Hacker News, Reddit (r/MachineLearning, r/LocalLLaMA)
- Tweet thread with benchmark comparisons
- YouTube demo video (5 minutes)

**Week 3-4:**
- Fine-tuning competition on HF with prizes
- Partner with LangChain/LlamaIndex for integration
- Release industry-specific fine-tunes (medical, legal, finance)
- Blog post: "How we built a 256M document VLM that beats 2B models"

**Month 2-3:**
- Enterprise case studies (with permission)
- Conference talks (CVPR, ICCV, NeurIPS workshop)
- Academic paper submission
- Open-source the training pipeline (not just model)

---

## 7. Moat & Defensibility

### 7.1 Data Moat

- **10M+ synthetic documents** with perfect ground truth — took 3 months to build pipeline
- **Proprietary document templates** from industry partnerships
- **Curated real-world dataset** (10K documents) with human-verified annotations
- **Continuous data collection** from community feedback (opt-in)

### 7.2 Architecture Moat

- **Document-specific attention mechanisms** (layout-preserving, region-aware)
- **Structured output heads** (JSON, KV, Table) — generalist VLMs don't have these
- **Optimized token compression** for documents (not images)
- **Multilingual document understanding** — most competitors are English-only

### 7.3 Ecosystem Moat

- **Python SDK** (`pip install tinydoc-vlm`) with one-liner extraction
- **API playground** for testing without code
- **Zapier/Make.com integration** for no-code users
- **Browser extension** (Chrome/Firefox) for instant document extraction
- **Mobile SDK** (iOS/Android) for on-device processing

### 7.4 Community Moat

- **Apache 2.0 license** for maximum adoption
- **Active Discord/Slack community** for support
- **Fine-tuning guides** for domain-specific variants
- **Bounty program** for bug reports and improvements
- **Regular model updates** (monthly) with community feedback

---

## 8. Risk Analysis & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Synthetic data quality issues** | Medium | High | Human validation pipeline + adversarial testing + real data fallback |
| **Benchmark overfitting** | Medium | High | Hold-out test set + external evaluation + real-world testing |
| **Competitor releases similar model** | High | Medium | Speed to market + ecosystem lock-in + continuous improvement |
| **Training compute costs** | Low | Medium | Gradient checkpointing + smaller batch sizes + cloud credits |
| **Multilingual performance gaps** | Medium | Medium | Language-specific adapters + community fine-tunes |
| **Structured output reliability** | Medium | High | JSON schema validation + constrained decoding + fallback parsing |
| **Adoption slower than expected** | Medium | High | Free tier + aggressive marketing + integration partnerships |

---

## 9. Timeline & Milestones

```
Month 1: Architecture & Data Pipeline
├── Week 1-2: Finalize architecture, build vision encoder prototype
├── Week 3-4: Build synthetic document generation pipeline (target: 1M docs)
└── Milestone: Generate 1M synthetic documents, validate quality

Month 2: Pretraining
├── Week 1-2: Stage 1 training (layout pretraining)
├── Week 3-4: Continue Stage 1, start Stage 2 data curation
└── Milestone: Layout detection accuracy >90% on internal benchmark

Month 3: Fine-tuning & Evaluation
├── Week 1-2: Stage 2 training (document understanding)
├── Week 3-4: Stage 3 training (instruction tuning), evaluation
└── Milestone: DocVQA >80%, OCRBench >70% on validation set

Month 4: Optimization & Packaging
├── Week 1-2: Quantization (INT8, INT4), ONNX export, edge optimization
├── Week 3-4: Build SDK, API, demo, documentation
└── Milestone: Model runs on Raspberry Pi in <5s per page

Month 5: Launch & Iteration
├── Week 1-2: HuggingFace release, OpenRouter integration, marketing
├── Week 3-4: Community feedback, bug fixes, first fine-tunes
└── Milestone: 10K+ downloads on HF, 100+ API users

Month 6: Scale & Dominate
├── Week 1-2: Industry-specific fine-tunes (medical, legal, finance)
├── Week 3-4: Academic paper, conference talks, enterprise pilots
└── Milestone: #1 trending document model on HF, 100K+ downloads
```

---

## 10. Resource Requirements

### 10.1 Compute

| Phase | Duration | Hardware | Cost |
|-------|----------|----------|------|
| Data generation | 4 weeks | 4× A100 (synthetic rendering) | $3,000 |
| Pretraining | 2 weeks | 8× A100 80GB | $5,000 |
| Fine-tuning | 3 weeks | 8× A100 80GB | $7,500 |
| Optimization | 1 week | 4× A100 + CPU servers | $1,500 |
| **Total** | **10 weeks** | | **~$17,000** |

**Alternative:** Own hardware (4× A6000 48GB) = ~$12,000 upfront, reusable for future models.

### 10.2 Team

| Role | Time | Skills |
|------|------|--------|
| **ML Engineer (Lead)** | Full-time | Vision transformers, multimodal models, PyTorch |
| **ML Engineer (Data)** | Full-time | Synthetic data generation, data pipelines, augmentation |
| **ML Engineer (Training)** | Full-time | Distributed training, optimization, benchmarking |
| **Software Engineer** | Half-time | SDK, API, demo, integrations |
| **Designer/UX** | Contract | Demo UI, marketing materials |
| **Technical Writer** | Contract | Documentation, model card, blog posts |

### 10.3 Budget Summary

| Category | Cost |
|----------|------|
| Compute (cloud) | $17,000 |
| Hardware (if buying) | $12,000 |
| Team salaries (3 months, 3.5 FTE) | $60,000 |
| Marketing & community | $5,000 |
| Infrastructure (API, hosting) | $2,000 |
| **Total (3 months)** | **~$84,000** |
| **Total (if own hardware)** | **~$79,000** |

---

## 11. Success Metrics

### 11.1 Technical Metrics

| Metric | 1 Month | 3 Months | 6 Months |
|--------|---------|----------|----------|
| DocVQA | >80% | >85% | >90% |
| OCRBench | >70% | >75% | >80% |
| FUNSD F1 | >90% | >95% | >97% |
| CORD F1 | >90% | >95% | >97% |
| Inference speed (CPU) | <5s | <2s | <1s |
| VRAM usage | <2GB | <1GB | <0.5GB |

### 11.2 Adoption Metrics

| Metric | 1 Month | 3 Months | 6 Months |
|--------|---------|----------|----------|
| HF Downloads | 10K | 100K | 500K |
| HF Likes | 500 | 2K | 10K |
| OpenRouter API calls | 1K/day | 10K/day | 100K/day |
| GitHub stars (SDK) | 500 | 2K | 10K |
| Community Discord members | 200 | 1K | 5K |
| Enterprise pilots | 2 | 10 | 30 |

### 11.3 Business Metrics

| Metric | 6 Months | 12 Months |
|--------|----------|-----------|
| API revenue | $1K/month | $10K/month |
| Enterprise licenses | 0 | 5 |
| Sponsorship/grants | $10K | $50K |
| Consulting/fine-tuning | $5K | $20K/month |

---

## 12. Conclusion

TinyDoc-VLM represents a **once-in-a-generation opportunity** in the tiny model space:

1. **Zero direct competition** in the <500M document-specialized VLM category
2. **Massive market demand** — every company needs document extraction
3. **Proven technical feasibility** — SmolVLM2 and LayoutLM3 prove the components work
4. **Clear differentiation** — structured output, multilingual, edge-first
5. **Strong moat** — data + architecture + ecosystem + community

**The bet:** A 256M parameter model, trained exclusively on documents, with structured output heads and layout-aware attention, can achieve **>85% DocVQA** and **>75% OCRBench** while running on a Raspberry Pi.

**If successful:** TinyDoc-VLM becomes the "all-MiniLM of document AI" — the default choice for any developer building document processing into their application.

**The time to build is now.**

---

*Document prepared June 2026. For questions: contact@tinydoc-vlm.org*
