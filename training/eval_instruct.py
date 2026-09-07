#!/usr/bin/env python3
"""
Eval: instruction-format prompts vs original question-format prompts.

Tests whether switching from "Answer the question: What is X?"
to "Extract X from the document." recovers QA performance on step 9000
without retraining.

Usage:
    python training/eval_instruct.py --ckpt-dir checkpoints/step9000
    python training/eval_instruct.py --ckpt-dir checkpoints/step9000 --device cpu
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

logger = logging.getLogger(__name__)


def normalize(s: str) -> str:
    s = re.sub(r"[\$£€]|\s+", "", s.lower())
    return s.strip(" .,:;\"'`!?")


def loose_match(pred: str, truth: str) -> bool:
    p, t = normalize(pred), normalize(truth)
    if not t:
        return False
    return p == t or t in p or p in t


def load_model(ckpt_dir: Path, device: str):
    from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor
    model = TinyDocVLMForConditionalGeneration.from_pretrained(
        str(ckpt_dir), trust_remote_code=True)
    processor = TinyDocVLMProcessor()
    processor.image_processor.image_size = model.config.image_size
    model = model.to(device).eval()
    return model, processor


def generate(model, processor, img, prompt: str, device: str,
             max_new_tokens: int = 128) -> str:
    inputs = processor(text=prompt, images=[img], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


INSTRUCT_TEMPLATES = {
    "What is the total amount?": "Extract the total amount from the document.",
    "Who is the vendor?": "Identify the vendor name from the document.",
    "What store is this receipt from?": "Identify the store name from the receipt.",
    "What is the form title?": "Extract the form title.",
    "What fields are on this form?": "List all fields on this form.",
    "What is the table title?": "Extract the table title.",
    "Convert table to JSON.": "Convert the table to JSON format.",
    "What is the cardholder's name?": "Extract the cardholder's name.",
    "What type of card is this?": "Identify the card type.",
    "What is the chart title?": "Extract the chart title.",
    "What is the maximum value?": "Extract the maximum value from the chart.",
    "What is the contract title?": "Extract the contract title.",
    "Who are the parties?": "Identify the parties involved in the contract.",
    "Who is the sender?": "Identify the sender of the letter.",
    "What is the subject?": "Extract the subject of the letter.",
    "What is the patient's name?": "Extract the patient's name.",
    "What is the diagnosis?": "Extract the diagnosis.",
    "What is the report title?": "Extract the report title.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--manifest", default="data/eval_768/manifest.json")
    ap.add_argument("--output", default="eval_instruct_results.json")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--skip-markdown", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    model, processor = load_model(Path(args.ckpt_dir), device)
    manifest = json.loads(Path(args.manifest).read_text())

    results = []
    orig_correct = orig_total = 0
    instruct_correct = instruct_total = 0
    md_orig_ok = md_instruct_ok = md_total = 0

    for item in manifest:
        from PIL import Image
        img = Image.open(item["page"]).convert("RGB")
        doc_type = item["doc_type"]
        page_name = Path(item["page"]).name
        entry = {"page": item["page"], "doc_type": doc_type, "qa": []}

        # Markdown comparison
        if not args.skip_markdown:
            for fmt, label in [("Convert the document to markdown:", "md_orig"),
                               ("Convert this document to markdown.", "md_instruct")]:
                t0 = time.time()
                out = generate(model, processor, img, fmt, device, args.max_new_tokens)
                entry[label] = out
                logger.info("[%s] %s (%.1fs) -> %d chars", doc_type, label, time.time()-t0, len(out))

            # Simple heuristic: does it contain a heading or table syntax?
            for label in ["md_orig", "md_instruct"]:
                text = entry.get(label, "")
                has_structure = bool(re.search(r'^#{1,3}\s|^\|.*\||^\*\*|\n#{1,3}\s', text, re.M))
                if label == "md_orig":
                    md_orig_ok += int(has_structure)
                else:
                    md_instruct_ok += int(has_structure)
                md_total += 1

        # QA comparison: original vs instruct
        for qa in item["qa"]:
            q = qa["question"]
            a = qa["answer"]
            instruct_q = INSTRUCT_TEMPLATES.get(q, f"Extract the answer for: {q}")

            # Original prompt
            t0 = time.time()
            pred_orig = generate(model, processor, img, f"Answer the question: {q}", device,
                                 min(args.max_new_tokens, 64))
            match_orig = loose_match(pred_orig, a)
            orig_total += 1
            orig_correct += int(match_orig)
            t_orig = time.time() - t0

            # Instruction prompt
            t0 = time.time()
            pred_instruct = generate(model, processor, img, instruct_q, device,
                                     min(args.max_new_tokens, 64))
            match_instruct = loose_match(pred_instruct, a)
            instruct_total += 1
            instruct_correct += int(match_instruct)
            t_instruct = time.time() - t0

            mark_o = "OK " if match_orig else "XX "
            mark_i = "OK " if match_instruct else "XX "
            logger.info("[%s] Q: %s", doc_type, q)
            logger.info("  %s orig:    %s (expect: %s, %.1fs)", mark_o, pred_orig[:60], a[:40], t_orig)
            logger.info("  %s instruct: %s (expect: %s, %.1fs)", mark_i, pred_instruct[:60], a[:40], t_instruct)

            entry["qa"].append({
                "question": q, "answer": a,
                "pred_orig": pred_orig, "match_orig": match_orig,
                "pred_instruct": pred_instruct, "match_instruct": match_instruct,
                "instruct_prompt": instruct_q,
            })

        results.append(entry)

    orig_pct = 100 * orig_correct / orig_total if orig_total else 0
    instruct_pct = 100 * instruct_correct / instruct_total if instruct_total else 0

    logger.info("=" * 70)
    logger.info("QA ORIGINAL:    %d/%d = %.1f%%", orig_correct, orig_total, orig_pct)
    logger.info("QA INSTRUCT:    %d/%d = %.1f%%", instruct_correct, instruct_total, instruct_pct)
    logger.info("DELTA:          %+.1f%%", instruct_pct - orig_pct)
    if md_total:
        logger.info("MD orig structure:    %d/%d", md_orig_ok, md_total)
        logger.info("MD instruct structure: %d/%d", md_instruct_ok, md_total)
    logger.info("=" * 70)

    out = {
        "device": device,
        "ckpt": str(args.ckpt_dir),
        "pages": len(results),
        "qa_orig": {"correct": orig_correct, "total": orig_total, "pct": orig_pct},
        "qa_instruct": {"correct": instruct_correct, "total": instruct_total, "pct": instruct_pct},
        "results": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results -> %s", args.output)


if __name__ == "__main__":
    main()
