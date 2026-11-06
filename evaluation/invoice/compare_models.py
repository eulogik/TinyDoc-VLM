#!/usr/bin/env python3
"""Paired comparison of two invoice-benchmark runs (no model calls).

Reads two `scores_invoice_<tag>.json` files from the benchmark scorer, pairs
documents by id, and reports key-header accuracy, fully-correct rate, per-field
and per-category differences, a document-level bootstrap CI, and a sign test.

`diff` is always **B - A**.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from summarize_scores import KEY_FIELDS


def _doc_stats(doc: dict) -> tuple[int, int, bool, dict[str, bool]]:
    correct = scored = 0
    fields: dict[str, bool] = {}
    for name in KEY_FIELDS:
        result = (doc.get("fields") or {}).get(name)
        if result is None:
            continue
        scored += 1
        hit = bool(result.get("correct"))
        correct += int(hit)
        fields[name] = hit
    return correct, scored, bool(doc.get("fully_correct")), fields


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(max(round(q * (len(ordered) - 1)), 0), len(ordered) - 1)
    return ordered[index]


def _sign_p(a_better: int, b_better: int) -> float:
    n = a_better + b_better
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_better, b_better) + 1))
    return min(1.0, 2 * tail / 2**n)


def compare(
    a_scores: dict,
    b_scores: dict,
    a_label: str = "A",
    b_label: str = "B",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    a_docs = {d["doc_id"]: d for d in a_scores.get("documents") or []}
    b_docs = {d["doc_id"]: d for d in b_scores.get("documents") or []}
    ids = sorted(set(a_docs) & set(b_docs))
    if not ids:
        raise ValueError("no paired documents")

    a_rows = {doc_id: _doc_stats(a_docs[doc_id]) for doc_id in ids}
    b_rows = {doc_id: _doc_stats(b_docs[doc_id]) for doc_id in ids}

    a_correct = sum(a_rows[i][0] for i in ids)
    a_scored = sum(a_rows[i][1] for i in ids)
    b_correct = sum(b_rows[i][0] for i in ids)
    b_scored = sum(b_rows[i][1] for i in ids)
    a_full = sum(int(a_rows[i][2]) for i in ids)
    b_full = sum(int(b_rows[i][2]) for i in ids)

    a_better = sum(1 for i in ids if a_rows[i][0] > b_rows[i][0])
    b_better = sum(1 for i in ids if b_rows[i][0] > a_rows[i][0])
    ties = len(ids) - a_better - b_better

    rng = random.Random(seed)
    key_diffs: list[float] = []
    full_diffs: list[float] = []
    for _ in range(n_boot):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        ac = sum(a_rows[i][0] for i in sample)
        asc = sum(a_rows[i][1] for i in sample)
        bc = sum(b_rows[i][0] for i in sample)
        bsc = sum(b_rows[i][1] for i in sample)
        key_diffs.append((bc / bsc if bsc else 0.0) - (ac / asc if asc else 0.0))
        full_diffs.append(
            sum(int(b_rows[i][2]) for i in sample) / len(sample)
            - sum(int(a_rows[i][2]) for i in sample) / len(sample)
        )

    per_field: dict[str, dict] = {}
    for name in KEY_FIELDS:
        a_hits = [a_rows[i][3].get(name) for i in ids if name in a_rows[i][3]]
        b_hits = [b_rows[i][3].get(name) for i in ids if name in b_rows[i][3]]
        if not a_hits:
            continue
        per_field[name] = {
            "a": sum(bool(x) for x in a_hits) / len(a_hits),
            "b": sum(bool(x) for x in b_hits) / len(b_hits),
            "n": len(a_hits),
            "diff": sum(bool(x) for x in b_hits) / len(b_hits)
            - sum(bool(x) for x in a_hits) / len(a_hits),
        }

    per_category: dict[str, dict] = defaultdict(lambda: {"n": 0, "a_key": 0, "a_scored": 0, "b_key": 0, "b_scored": 0})
    for doc_id in ids:
        category = a_docs[doc_id].get("category") or "unknown"
        cat = per_category[category]
        cat["n"] += 1
        cat["a_key"] += a_rows[doc_id][0]
        cat["a_scored"] += a_rows[doc_id][1]
        cat["b_key"] += b_rows[doc_id][0]
        cat["b_scored"] += b_rows[doc_id][1]
    categories = {}
    for name, cat in sorted(per_category.items()):
        a_acc = cat["a_key"] / cat["a_scored"] if cat["a_scored"] else 0.0
        b_acc = cat["b_key"] / cat["b_scored"] if cat["b_scored"] else 0.0
        categories[name] = {
            "n": cat["n"],
            "a": a_acc,
            "b": b_acc,
            "diff": b_acc - a_acc,
        }

    return {
        "a": a_label,
        "b": b_label,
        "n_docs": len(ids),
        "key": {
            "a": a_correct / a_scored if a_scored else 0.0,
            "b": b_correct / b_scored if b_scored else 0.0,
            "diff": (b_correct / b_scored if b_scored else 0.0)
            - (a_correct / a_scored if a_scored else 0.0),
            "ci95": [_pct(key_diffs, 0.025), _pct(key_diffs, 0.975)],
            "a_better_docs": a_better,
            "b_better_docs": b_better,
            "ties": ties,
            "sign_p": _sign_p(a_better, b_better),
        },
        "fully_correct": {
            "a": a_full / len(ids),
            "b": b_full / len(ids),
            "diff": (b_full - a_full) / len(ids),
            "ci95": [_pct(full_diffs, 0.025), _pct(full_diffs, 0.975)],
            "a_only": sum(1 for i in ids if a_rows[i][2] and not b_rows[i][2]),
            "b_only": sum(1 for i in ids if b_rows[i][2] and not a_rows[i][2]),
            "both": sum(1 for i in ids if a_rows[i][2] and b_rows[i][2]),
            "neither": sum(1 for i in ids if not a_rows[i][2] and not b_rows[i][2]),
        },
        "per_field": per_field,
        "per_category": categories,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-scores", required=True)
    ap.add_argument("--b-scores", required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    report = compare(
        json.loads(Path(args.a_scores).read_text()),
        json.loads(Path(args.b_scores).read_text()),
        args.a_label,
        args.b_label,
        args.boot,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
