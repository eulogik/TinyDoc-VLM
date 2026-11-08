#!/usr/bin/env python3
"""Run a DocVQA reading probe through Ollama (no cloud).

DocVQA asks short questions about document images. This probe measures whether a
base reader can resolve document text; it is not field extraction and not a
product claim. The set is public and may be in the reader's pretraining.

Writes per-question JSONL and a compact scores JSON with DocVQA-style ANLS and
normalized exact match. Inference over multiple documents needs explicit
approval (AGENTS.md).

Usage:
    scripts/run_docvqa_probe.sh qwen3-vl:2b-instruct qwen3-vl_2b_instruct
    # or directly
    python evaluation/docvqa/run_docvqa_probe.py \
        --manifest evaluation/docvqa/manifest_docvqa_200.jsonl \
        --model qwen3-vl:2b-instruct \
        --out evaluation/docvqa/results/rows_docvqa_qwen3-vl_2b_instruct.jsonl \
        --scores evaluation/docvqa/results/scores_docvqa_qwen3-vl_2b_instruct.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import string
import sys
import time
import urllib.request
from pathlib import Path

PROMPT = """Read the document image and answer the question.
Reply with the exact answer only, no explanation, no markdown.

Question: {question}

Answer:"""

ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = text.translate(PUNCT)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, 1):
        current = [i]
        for j, ch_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ch_a != ch_b),
                )
            )
        previous = current
    return previous[-1]


def anls_score(prediction: str, target: str, threshold: float = 0.5) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if not pred or not gold:
        return 0.0
    similarity = 1.0 - levenshtein(pred, gold) / max(len(pred), len(gold), 1)
    return similarity if similarity >= threshold else 0.0


def extract_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for key in ("answer", "response", "text"):
                    if key in obj:
                        return str(obj[key]).strip()
        except json.JSONDecodeError:
            pass
    for prefix in ("Answer:", "answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text.splitlines()[0].strip().strip('"').strip("'")


def _read_image(path: Path, attempts: int = 3, delay: float = 2.0) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return path.read_bytes()
        except OSError as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
    assert last is not None
    raise last


def _check_host(host: str, timeout: float = 10.0) -> None:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            resp.read(1024)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Ollama not reachable at {host}: {type(exc).__name__}: {exc}")


def ask_ollama(host: str, model: str, prompt: str, b64: str, num_ctx: int, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 128, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return (body.get("message") or {}).get("content", "") or ""


def _load_resume_rows(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0
    loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    done = [row for row in loaded if not row.get("error")]
    return done, len(loaded) - len(done)


def summarize(rows: list[dict]) -> dict:
    scores = [row["anls"] for row in rows]
    exact = [1.0 if row["exact"] else 0.0 for row in rows]
    latencies = [row["latency_ms"] for row in rows]
    errors = sum(1 for row in rows if row.get("error"))
    return {
        "n": len(rows),
        "anls_mean": statistics.mean(scores) if scores else 0.0,
        "exact_rate": statistics.mean(exact) if exact else 0.0,
        "errors": errors,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-dir", default="data/datasets/docvqa")
    ap.add_argument("--model", default="qwen3-vl:2b-instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument(
        "--num-ctx", type=int, default=int(os.environ.get("TINYDOC_OLLAMA_NUM_CTX", "8192"))
    )
    ap.add_argument(
        "--timeout", type=float, default=float(os.environ.get("TINYDOC_OLLAMA_TIMEOUT", "600"))
    )
    ap.add_argument("--out", required=True, help="per-question JSONL")
    ap.add_argument("--scores", required=True, help="compact scores JSON")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"no rows in {manifest_path}", file=sys.stderr)
        return 2

    host = args.host.rstrip("/")
    _check_host(host)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_rows: list[dict] = []
    failed = 0
    if args.resume and out_path.exists():
        done_rows, failed = _load_resume_rows(out_path)
        with out_path.open("w") as handle:
            for row in done_rows:
                handle.write(json.dumps(row) + "\n")
        done_ids = {row["id"] for row in done_rows}
        rows = [row for row in rows if row["id"] not in done_ids]
        print(f"resume: {len(done_ids)} done, {failed} failed will retry, {len(rows)} remaining")

    total = len(done_rows) + len(rows)
    with out_path.open("a") as sink:
        for row in rows:
            image_path = data_dir / "images" / row["image"]
            raw = ""
            answer = ""
            error = None
            started = time.time()
            try:
                b64 = base64.b64encode(_read_image(image_path)).decode()
                raw = ask_ollama(
                    host,
                    args.model,
                    PROMPT.format(question=row["question"]),
                    b64,
                    args.num_ctx,
                    args.timeout,
                )
                answer = extract_answer(raw)
                if not answer:
                    error = "empty answer"
            except Exception as exc:  # noqa: BLE001 - keep the row, count the failure
                error = f"{type(exc).__name__}: {exc}"
            latency = (time.time() - started) * 1000
            anls = anls_score(answer, row["target"])
            exact = normalize_answer(answer) == normalize_answer(row["target"])
            done = {
                "id": row["id"],
                "image": row["image"],
                "question": row["question"],
                "target": row["target"],
                "raw": raw,
                "answer": answer,
                "anls": anls,
                "exact": exact,
                "latency_ms": latency,
                "error": error,
            }
            done_rows.append(done)
            sink.write(json.dumps(done) + "\n")
            sink.flush()
            if len(done_rows) % 20 == 0:
                print(
                    f"[{len(done_rows)}/{total}] anls={statistics.mean(r['anls'] for r in done_rows):.4f} "
                    f"errors={sum(1 for r in done_rows if r['error'])}",
                    flush=True,
                )

    summary = summarize(done_rows)
    summary.update(
        {
            "model": args.model,
            "manifest": str(manifest_path),
            "set": "docvqa_public_mirror",
            "single_reference": True,
            "sampling": "one question per image, seed 42",
            "caveat": (
                "public DocVQA questions; may be in pretraining; reading probe, not extraction; "
                "single-reference ANLS, not comparable to leaderboard numbers"
            ),
            "rows": str(out_path),
        }
    )
    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
