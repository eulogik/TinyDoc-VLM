# TinyDoc release checklist

**Current target:** `tinydoc` 0.3.0 (page-wise invoice SDK + local UI), published.
**Last published:** 0.3.0 on 2026-11-12 (multipage + UI).

## 1. Preflight (must all pass)

```bash
export PATH="/opt/homebrew/bin:$PATH"
python3 -m pytest tests/ -q                 # expect 228 passed
python3 -m ruff check tinydoc_vlm/ tests/   # CI scope
python3 evaluation/phase0/recompute_scores.py
node scripts/verify-product.mjs audit
```

Claim audit must exit 0. Do not publish a number that does not recompute from
`evaluation/phase0/results/`.

## 2. Build

```bash
cd sdk
rm -rf build dist tinydoc.egg-info
python3 -m build --wheel --sdist
unzip -l dist/tinydoc-0.3.0-py3-none-any.whl | grep schemas
```

The wheel must contain `tinydoc/schemas/sroie_receipt.schema.json` and
`tinydoc/schemas/invoice.schema.json`.

## 3. Clean-venv install test

```bash
VENV=/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/tinydoc_venv_release
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q sdk/dist/tinydoc-0.3.0-py3-none-any.whl
"$VENV/bin/python" -c "import importlib.metadata as m; assert m.version('tinydoc')=='0.3.0'"
"$VENV/bin/tinydoc" --help
"$VENV/bin/python" -c "from tinydoc.invoice import invoice_schema_is_valid; assert invoice_schema_is_valid({'vendor_name':'A','invoice_number':'1','invoice_date':'2026-01-01','due_date':None,'currency':'USD','subtotal':1.0,'tax':0.0,'total_amount':1.0,'document_type':'invoice','line_items':[]})"
TINYDOC_OLLAMA_TIMEOUT=600 TINYDOC_OLLAMA_NUM_CTX=8192 \
  "$VENV/bin/tinydoc" extract <one-receipt.jpg> \
  --engine ollama:qwen3-vl:2b-instruct --no-evidence
```

The extraction must return `schema=True` and non-empty fields.

## 4. Publish to PyPI

Requires a PyPI API token with upload scope. Never commit the token.

```bash
python3 -m twine upload sdk/dist/tinydoc-0.3.0-py3-none-any.whl sdk/dist/tinydoc-0.3.0.tar.gz
# username: __token__
# password: pypi-...
```

If `twine` is missing: `python3 -m pip install --user twine`.

## 5. Verify the published package

```bash
python3 -m venv /tmp/tinydoc_pypi_check
/tmp/tinydoc_pypi_check/bin/pip install tinydoc==0.3.0
/tmp/tinydoc_pypi_check/bin/python -c "import tinydoc; print(tinydoc.__all__)"
```

Confirm <https://pypi.org/project/tinydoc/0.3.0/> shows the new README. The
0.2.0 upload token was rotated after a leak; use a fresh token for any future
release and never commit it.

## 6. Hugging Face Space

The Space is a **static card** (`demo/hf_space/`). It must not load the retired
256M checkpoint. Deploy the three files:

```bash
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
tok = Path("training/.hf_token").read_text().strip()
api = HfApi(token=tok)
for name in ("README.md", "app.py", "requirements.txt"):
    api.upload_file(
        path_or_fileobj=str(Path("demo/hf_space") / name),
        path_in_repo=name,
        repo_id="eulogik/TinyDoc-VLM",
        repo_type="space",
        commit_message="TinyDoc static evidence card",
    )
PY
```

Check the Space reaches `RUNNING` and renders the card.

## 7. Post-release

- Tag **`sdk-v0.3.0`** at the 0.3.0 release commit and create a GitHub release
  with the wheel/sdist. Do **not** reuse `v0.2.0` or `sdk-v0.2.0`: those tags
  point at older commits and published tags should not be moved.
- Keep the README install line pinned to the latest PyPI release.
- Keep the model card retirement notice; do not re-point it at the 256M.
- Record the release in `docs/BENCHMARKS.md` if any number changes.

## Rollback

- PyPI cannot reuse a version. If 0.2.0 is broken, yank it and release 0.2.1.
- The HF Space can be reverted with `git revert` on the Space repo (or re-upload
  the previous `app.py`).

## Current status (2026-11-12)

- **0.3.0 is published on PyPI** with the wheel and sdist: `InvoicePipeline`,
  `ocr_totals.py`, `invoice.schema.json`, `ui.py`, and the `invoice`/`ui` CLI
  commands. A clean venv installs 0.3.0 and passes the schema/CLI smoke tests.
- The clean 10-document multipage probe (VLM + OCR totals) is measured and
  committed: 10/10 fully correct, 90/90 key-header, 0.848 line F1.
- The degraded multipage probe is measured and committed: 8/10 documents
  completed, 7/8 fully correct, 71/72 key-header, 0.795 line F1, and two
  explicitly allowed page failures (malformed JSON and an Ollama HTTP 500).
- Space card and repository docs describe the 0.3.0 release as preview-quality.
- Tag **`sdk-v0.3.0`** at the release commit; do not move `v0.2.0` or
  `sdk-v0.2.0`.

## 0.2.0 status (2026-11-07)

- 0.2.0 wheel and sdist built in `sdk/dist/`; `twine check` passed.
- **0.2.0 is published on PyPI**; `https://pypi.org/pypi/tinydoc/json` reports
  0.2.0 with both files.
- Clean-venv install + Ollama extraction passed against the built wheel before
  upload. On 2026-11-08, a fresh venv installed `tinydoc==0.2.0` from
  `https://pypi.org/simple`, imported version 0.2.0, and the `tinydoc` CLI
  started. (An earlier attempt failed on transient index network errors, not on
  the package.)
- The upload token appeared in a chat transcript: **rotate it** in PyPI account
  settings and delete any local copy.
- Space updated to the static card and verified **RUNNING**.
- Use tag `sdk-v0.2.0` for this release; the existing `v0.2.0` tag is older.
