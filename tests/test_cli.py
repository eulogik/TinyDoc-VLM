"""Unit tests for the tinydoc CLI (pipeline stubbed — no network/OCR)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from tinydoc import cli
from tinydoc import pipeline as pl
from tinydoc.pipeline import DocumentResult, FieldResult


def _result(image_path: str, value: str = "34.21") -> DocumentResult:
    fr = FieldResult(
        name="total",
        value=value,
        present=True,
        evidence={
            "kind": "ocr_span",
            "quote": value,
            "bbox": [10.0, 10.0, 60.0, 30.0],
            "page": 1,
            "score": 0.9,
        },
        confidence=0.8,
    )
    return DocumentResult(
        image_path=image_path,
        engine="stub",
        fields={"company": "ACME", "date": "2024-01-15", "address": "1 Main St", "total": value},
        field_results=[fr],
        schema_valid=True,
        confidence=0.8,
        raw="{}",
        latency_ms=12.0,
        evidence_coverage=1.0,
    )


class StubPipeline:
    instances: ClassVar[list] = []

    def __init__(self, engine_name: str = "auto", **kwargs):
        self.init_args = (engine_name, kwargs)
        self.engine = type("E", (), {"name": "stub"})()
        self.calls: list = []
        StubPipeline.instances.append(self)

    def extract(self, path, **kwargs):
        self.calls.append(("extract", str(path), kwargs))
        return _result(str(path))

    def extract_folder(self, path, **kwargs):
        self.calls.append(("folder", str(path), kwargs))
        imgs = sorted(Path(path).glob("*.png")) + sorted(Path(path).glob("*.jpg"))
        return [_result(str(p)) for p in imgs[: kwargs.get("limit") or len(imgs)]]


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch):
    StubPipeline.instances.clear()
    monkeypatch.setattr(pl, "ReceiptPipeline", StubPipeline)
    return StubPipeline


def _images(folder: Path, n: int = 2) -> list[Path]:
    from PIL import Image

    paths = []
    for i in range(n):
        p = folder / f"doc_{i}.png"
        Image.new("RGB", (40, 20), (255, 255, 255)).save(p)
        paths.append(p)
    return paths


def test_path_not_found_returns_2(tmp_path: Path):
    assert cli.main(["extract", str(tmp_path / "nope.png")]) == 2


def test_empty_folder_returns_1(tmp_path: Path):
    assert cli.main(["extract", str(tmp_path)]) == 1


def test_extract_writes_jsonl(tmp_path: Path, stub_pipeline, capsys):
    _images(tmp_path, 2)
    out = tmp_path / "out" / "results.jsonl"
    rc = cli.main(["extract", str(tmp_path), "--engine", "ocr_regex", "--out", str(out)])
    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["fields"]["total"] == "34.21"
    assert lines[0]["schema_valid"] is True
    captured = capsys.readouterr().out
    assert "engine=stub n=2" in captured
    assert "schema=1.000" in captured


def test_no_evidence_flag_reaches_pipeline(tmp_path: Path, stub_pipeline):
    _images(tmp_path, 1)
    cli.main(["extract", str(tmp_path), "--no-evidence"])
    kind, _, kwargs = stub_pipeline.instances[0].calls[0]
    assert kind == "folder"
    assert kwargs["with_evidence"] is False


def test_limit_reaches_pipeline(tmp_path: Path, stub_pipeline):
    _images(tmp_path, 3)
    cli.main(["extract", str(tmp_path), "--limit", "1"])
    _, _, kwargs = stub_pipeline.instances[0].calls[0]
    assert kwargs["limit"] == 1


def test_model_kwarg_passthrough(tmp_path: Path, stub_pipeline):
    _images(tmp_path, 1)
    cli.main(["extract", str(tmp_path), "--model", "qwen2.5vl:7b"])
    _engine_name, kwargs = stub_pipeline.instances[0].init_args
    assert kwargs.get("model") == "qwen2.5vl:7b"


def test_overlay_dir_writes_pngs(tmp_path: Path, stub_pipeline):
    _images(tmp_path, 2)
    odir = tmp_path / "overlays"
    rc = cli.main(["extract", str(tmp_path), "--overlay", str(odir)])
    assert rc == 0
    pngs = sorted(odir.glob("*_overlay.png"))
    assert len(pngs) == 2
    assert all(p.stat().st_size > 0 for p in pngs)


def test_single_image_uses_extract(tmp_path: Path, stub_pipeline):
    img = _images(tmp_path, 1)[0]
    rc = cli.main(["extract", str(img)])
    assert rc == 0
    kind, _path, _kwargs = stub_pipeline.instances[0].calls[0]
    assert kind == "extract"
