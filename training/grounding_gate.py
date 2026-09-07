#!/usr/bin/env python3
"""
Grounding gate: does the model actually use the image?

Runs three checks on the 12-page eval set and reports PASS/FAIL:
  1. IMAGE-DEPENDENCE: same question on different pages must give
     different answers (>70% of page pairs differ). Catches blind priors.
  2. ENTROPY: mean first-answer-token entropy must be >1.0 nat.
     Catches mode collapse (P=0.9999 single-token priors).
  3. QA ACCURACY: loose-match on eval manifest (informational; >15% to pass).

Works with SmolVLM2 (AutoModelForImageTextToText) and TinyDoc-VLM
(--model-type tinydoc).

Usage:
    python training/grounding_gate.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
    python training/grounding_gate.py --model eulogik/SmolVLM2-TinyDoc-real --adapter-only
    python training/grounding_gate.py --model checkpoints/step9000 --model-type tinydoc
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
PASS_DEP, PASS_ENT, PASS_QA = 0.70, 1.0, 0.15


def normalize(s: str) -> str:
    s = re.sub(r"[\$£€]|\s+", "", s.lower())
    return s.strip(" .,:;\"'`!?")


def loose_match(pred: str, truth: str) -> bool:
    p, t = normalize(pred), normalize(truth)
    return bool(t) and (p == t or t in p or p in t)


def build_prompt(processor, model_type: str, img, question: str):
    if model_type == "tinydoc":
        return processor(text=f"<image>\n{question}", images=[img],
                         return_tensors="pt")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=[text], images=[[img]], return_tensors="pt")


def generate(model, processor, model_type: str, img, question: str,
             device: str, max_new_tokens: int = 64) -> str:
    inp = build_prompt(processor, model_type, img, question)
    kwargs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inp.items() if k != "image_token_id"}
    with torch.no_grad():
        out = model.generate(**kwargs, max_new_tokens=max_new_tokens,
                             do_sample=False, use_cache=True)
    # strip prompt tokens (batch dim may be nested for SmolVLM2)
    seq = out[0]
    n_in = kwargs["input_ids"].shape[-1]
    gen_ids = seq[n_in:] if seq.shape[0] >= n_in else seq
    tok = getattr(processor, "tokenizer", processor)
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def first_token_entropy(model, processor, model_type: str, img,
                        question: str, device: str) -> float:
    inp = build_prompt(processor, model_type, img, question)
    kwargs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inp.items() if k != "image_token_id"}
    kwargs.pop("pixel_values", None) if False else None
    with torch.no_grad():
        logits = model(**{k: v for k, v in kwargs.items()
                           if k in ("input_ids", "attention_mask",
                                    "pixel_values")}).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="smolvlm2", choices=["smolvlm2", "tinydoc"])
    ap.add_argument("--adapter-only", action="store_true",
                    help="--model is a LoRA adapter repo; load over SmolVLM2 base")
    ap.add_argument("--manifest", default="data/eval_768/manifest.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="grounding_gate.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.model_type == "tinydoc":
        from tinydoc_vlm import TinyDocVLMForConditionalGeneration as Cls, TinyDocVLMProcessor
        model = Cls.from_pretrained(args.model, trust_remote_code=True)
        processor = TinyDocVLMProcessor()
        processor.image_processor.image_size = model.config.image_size
    else:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel
        base = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        if args.adapter_only:
            m0 = AutoModelForImageTextToText.from_pretrained(
                base, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)
            model = PeftModel.from_pretrained(m0, args.model)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)
    model.eval()

    from PIL import Image
    manifest = json.loads(Path(args.manifest).read_text())
    pages = [(m["doc_type"], Image.open(m["page"]).convert("RGB")) for m in manifest]
    qas = [(m["page"], q["question"], q["answer"])
           for m in manifest for q in m["qa"]]
    logger.info("%d pages, %d QA pairs", len(pages), len(qas))

    # 1. image-dependence: fixed question, all page pairs
    probe_q = "What is the total amount?"
    outs = {}
    for dt, img in pages:
        t0 = time.time()
        outs[dt + "|" + str(len(outs))] = generate(
            model, processor, args.model_type, img, probe_q, device, 32)
        logger.info("dep probe %s %.0fs", dt, time.time() - t0)
    vals = list(outs.values())
    diff = sum(1 for i in range(len(vals)) for j in range(i + 1, len(vals))
               if normalize(vals[i]) != normalize(vals[j]))
    total_pairs = len(vals) * (len(vals) - 1) // 2
    dep = diff / total_pairs if total_pairs else 0.0

    # 2. entropy over a few (page, question) combos
    ents = []
    for (page, q, a) in qas[:6]:
        img = Image.open(page).convert("RGB")
        try:
            e = first_token_entropy(model, processor, args.model_type, img, q, device)
        except Exception as ex:  # pragma: no cover
            logger.warning("entropy probe failed: %s", ex)
            continue
        ents.append(e)
        logger.info("entropy %.2f for %s", e, q[:40])
    mean_ent = sum(ents) / len(ents) if ents else 0.0

    # 3. QA accuracy
    correct = 0
    for page, q, a in qas:
        img = Image.open(page).convert("RGB")
        pred = generate(model, processor, args.model_type, img, q, device, 64)
        hit = loose_match(pred, a)
        correct += int(hit)
        logger.info("[%s] %s -> %s | %s", "OK " if hit else "XX ", q[:45],
                    pred[:60], a[:40])
    acc = correct / len(qas) if qas else 0.0

    verdict = {"image_dependence": dep, "mean_entropy": mean_ent, "qa_acc": acc,
               "pass_dep": dep >= PASS_DEP, "pass_ent": mean_ent >= PASS_ENT,
               "pass_qa": acc >= PASS_QA}
    verdict["ALL_PASS"] = verdict["pass_dep"] and verdict["pass_ent"] and verdict["pass_qa"]
    logger.info("GATE: dep=%.2f ent=%.2f qa=%.2f -> %s",
                dep, mean_ent, acc, "PASS" if verdict["ALL_PASS"] else "FAIL")
    Path(args.output).write_text(json.dumps(verdict, indent=2))
    sys.exit(0 if verdict["ALL_PASS"] else 1)


if __name__ == "__main__":
    main()
