# TinyDoc release checklist

**Current target:** `tinydoc` 0.2.0 (SDK pipeline, schemas, evidence, CLI).

## 1. Preflight (must all pass)

```bash
export PATH="/opt/homebrew/bin:$PATH"
python3 -m pytest tests/ -q                 # expect 106 passed
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
unzip -l dist/tinydoc-0.2.0-py3-none-any.whl | grep schemas
```

The wheel must contain `tinydoc/schemas/sroie_receipt.schema.json`.

## 3. Clean-venv install test

```bash
VENV=/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/tinydoc_venv_release
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q sdk/dist/tinydoc-0.2.0-py3-none-any.whl
"$VENV/bin/python" -c "import importlib.metadata as m; assert m.version('tinydoc')=='0.2.0'"
"$VENV/bin/tinydoc" --help
TINYDOC_OLLAMA_TIMEOUT=600 TINYDOC_OLLAMA_NUM_CTX=8192 \
  "$VENV/bin/tinydoc" extract <one-receipt.jpg> \
  --engine ollama:qwen3-vl:2b-instruct --no-evidence
```

The extraction must return `schema=True` and non-empty fields.

## 4. Publish to PyPI

Requires a PyPI API token with upload scope. Never commit the token.

```bash
python3 -m twine upload sdk/dist/tinydoc-0.2.0-py3-none-any.whl sdk/dist/tinydoc-0.2.0.tar.gz
# username: __token__
# password: pypi-...
```

If `twine` is missing: `python3 -m pip install --user twine`.

## 5. Verify the published package

```bash
python3 -m venv /tmp/tinydoc_pypi_check
/tmp/tinydoc_pypi_check/bin/pip install tinydoc==0.2.0
/tmp/tinydoc_pypi_check/bin/python -c "import tinydoc; print(tinydoc.__all__)"
```

Confirm <https://pypi.org/project/tinydoc/0.2.0/> shows the new README.

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

- Tag `v0.2.0` and create a GitHub release with the wheel/sdist.
- Update the README install line from source to PyPI once 0.2.0 is live.
- Keep the model card retirement notice; do not re-point it at the 256M.
- Record the release in `docs/BENCHMARKS.md` if any number changes.

## Rollback

- PyPI cannot reuse a version. If 0.2.0 is broken, yank it and release 0.2.1.
- The HF Space can be reverted with `git revert` on the Space repo (or re-upload
  the previous `app.py`).

## Current status (2026-11-06)

- 0.2.0 wheel and sdist build successfully.
- Clean-venv install + Ollama extraction verified.
- PyPI upload **not done** — no token in the environment.
- Space updated to the static card and verified **RUNNING** (`/config` serves the new card).
