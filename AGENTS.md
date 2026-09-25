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
  (repo `/tmp` gets wiped by macOS).
- Ollama: never send `format:"json"`; prompt-v1 lives in
  `engines.EXTRACT_PROMPT` ↔ `pipeline.OllamaEngine` (must stay in sync).
- Eval images live behind the untracked `data/training` symlink — never
  `resolve()` the `evaluation` symlink when constructing paths.
- `TINYDOC_OLLAMA_TIMEOUT` (default 180) controls Ollama client timeout; use
  600 for slow local runs — client disconnects mid-generation wedge the
  single-slot server.

## Commands

- Full test suite: `python3 -m pytest tests/ -q` (expect 69 passed)
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
