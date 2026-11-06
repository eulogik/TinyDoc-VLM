"""Ollama request-option tests (no network).

Regression guard for the 4096-token context bug: full-page scans are ~4.2k
visual tokens, so Ollama's default context returned HTTP 400 and the harness
counted the document as four empty fields.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from tinydoc.pipeline import FIELDS, OllamaEngine


@pytest.fixture
def receipt_image(tmp_path: Path) -> Path:
    p = tmp_path / "receipt.png"
    Image.new("RGB", (200, 300), (255, 255, 255)).save(p)
    return p


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_requests(monkeypatch, payloads: list[dict]) -> None:
    def fake_urlopen(req, timeout=None):
        payloads.append(json.loads(req.data.decode()))
        return _FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "company": "A",
                            "date": "d",
                            "address": "a",
                            "total": "1.00",
                        }
                    )
                }
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_ollama_default_context_is_8192(monkeypatch, receipt_image):
    monkeypatch.delenv("TINYDOC_OLLAMA_NUM_CTX", raising=False)
    payloads: list[dict] = []
    _capture_requests(monkeypatch, payloads)

    fields = OllamaEngine(model="test-model").extract_fields(str(receipt_image))

    assert set(fields) == set(FIELDS)
    assert len(payloads) == 1
    opts = payloads[0]["options"]
    assert opts["num_ctx"] == 8192
    assert opts["num_predict"] == 512
    assert opts["temperature"] == 0


def test_ollama_context_env_override(monkeypatch, receipt_image):
    monkeypatch.setenv("TINYDOC_OLLAMA_NUM_CTX", "16384")
    payloads: list[dict] = []
    _capture_requests(monkeypatch, payloads)

    OllamaEngine(model="test-model").extract_fields(str(receipt_image))

    assert payloads[0]["options"]["num_ctx"] == 16384
