# AGENTS.md

## Heavy compute: ALWAYS ASK FIRST

Ask the user (use the question tool) and get explicit approval before running:
- Ollama / VLM inference over multiple documents (evals, pilots, benchmarks)
- OCR over 10+ documents (RapidOCR/Tesseract batch runs)
- Model downloads (`ollama pull`, HF weight downloads)
- Anything training-related, or jobs expected to take > ~2 minutes of CPU/GPU

Context: this is a 16 GB M4 MacBook. Sustained 7B-on-MPS generation already
crashed the machine once (2026-09-25). Approved runs should still ship with a
memory watchdog when they are model inference.

## Environment

- `export PATH="/opt/homebrew/bin:$PATH"` first in every shell (homebrew tools).
- System python: `/opt/homebrew/bin/python3` (3.14, PEP 668 — use
  `pip install --break-system-packages` only for pure-python deps, or a venv).
- RapidOCR venv (python3.11): `/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/ocr_venv`
  — has rapidocr-onnxruntime, pytesseract, pydantic, jsonschema.
- Scratch dir: `/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/`
  (repo `/tmp` gets wiped by macOS). NOTE 2026-09-29: a macOS REBOOT also
  wipes this scratch dir — venv, model weights, datasets, and training
  checkpoints do NOT survive reboots. Everything is rebuildable from the repo
  (builder + supervisor scripts); committed artifacts in `evaluation/phase0/results/`
  are the source of truth. Rebuild order: venv → SmolVLM-500M weights →
  rajistics/sroie → build_stage1_manifest.py → hf_train/hf_val parquets →
  supervise_training.sh.
- Ollama: never send `format:"json"`; prompt-v1 lives in
  `engines.EXTRACT_PROMPT` ↔ `pipeline.OllamaEngine` (must stay in sync).
- Eval images live behind the untracked `data/training` symlink — never
  `resolve()` the `evaluation` symlink when constructing paths.
- `TINYDOC_OLLAMA_TIMEOUT` (default 180) controls Ollama client timeout; use
  600 for slow local runs — client disconnects mid-generation wedge the
  single-slot server.
- `TINYDOC_OLLAMA_NUM_CTX` (default 8192) sets Ollama's context window. The
  old implicit 4096 rejected 2481×3508 scans with HTTP 400, which phase-0
  harnesses counted as empty predictions. Do not lower it for full-page scans.

## Commands

- Full test suite: `python3 -m pytest tests/ -q` (expect 137 passed)
- CI-scope lint: `python3 -m ruff check tinydoc_vlm/ tests/`
- Claim audit (must exit 0 before publishing numbers):
  `python3 evaluation/phase0/recompute_scores.py`
- Product gates: `node scripts/verify-product.mjs <baseline|engine|evidence_ab|claims|tests|gitignore|ci|audit|all>`
- OCRBench gates: `node scripts/verify-ocrbench.mjs all`

## Honesty rules (repo policy)

- Only numbers that recompute from `evaluation/phase0/results/` artifacts may
  appear as results. Estimates/drafts must carry banners.
- FUNSD is a transfer probe with nonstandard gold mapping — never cited as a
  leaderboard number.
- The 256M checkpoint is retired from performance claims (OCRBench 0.0%, n=1000).
- VLM-predicted boxes failed the grounding probe (hit@0.5 0.04–0.20 vs 0.70
  gate). Ship OCR-span evidence only; never claim VLM grounding.
- Confidence is only partly calibrated. Evidence-off eval runs are constant
  (0.4025; field ECE 0.52–0.55); the evidence-on default is field ECE 0.115–0.135
  raw and 0.005–0.014 after document-grouped isotonic OOF. Document-level ECE is
  0.09–0.10 and does not reliably improve. Never present doc confidence as a
  probability; field-level scores are for review routing only.
- Invoices are measured only on the synthetic, vendor-made Invoice Extraction
  Benchmark v1 (page 1): 3B 91.9% key-header / 65.7% fully correct; 2B 93.5% /
  71.8%. The paired tier diff is not significant (p=0.092) and does not transfer
  to receipts. Flags are 0/7. Never cite as real-invoice performance; real
  invoices are unmeasured.
