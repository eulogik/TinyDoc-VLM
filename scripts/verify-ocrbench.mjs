#!/usr/bin/env node
// Gate checker for the OCRBench 256M re-evaluation.
// Usage: node scripts/verify-ocrbench.mjs <controls|vision|tokenid|pilot|full|recompute>
// Requires: node, python3 with torch/transformers/PIL, and the repo on disk.

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const RESULTS = join(ROOT, "evaluation", "phase0", "results");

const mode = process.argv[2];
if (!mode) {
  console.error("usage: verify-ocrbench.mjs <controls|vision|tokenid|pilot|full|recompute>");
  process.exit(2);
}

function fail(msg) {
  console.error(`FAIL: ${msg}`);
  process.exit(1);
}

function py(code) {
  return execFileSync("python3", ["-c", code], {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function findSummary(tag) {
  const suffix = tag ? `_${tag}` : "";
  const path = join(RESULTS, `ocrbench_256m${suffix}.json`);
  return existsSync(path) ? path : null;
}

if (mode === "controls") {
  // Positive control: exact match and a genuinely close OCR answer must score.
  // Negative control: unrelated garbage must not score.
  const out = py(`
import sys
sys.path.insert(0, "evaluation")
from run_ocrbench_eval import score_sample, edit_distance
fails = []
r = score_sample("CENTRE", ["CENTRE"], "Regular Text Recognition")
if r["strict"] != 1.0: fails.append("exact match must be strict 1.0, got %r" % r)
r = score_sample("  centre  ", ["CENTRE"], "Regular Text Recognition")
if r["strict"] != 1.0: fails.append("casefold/space-normalised match must score, got %r" % r)
r = score_sample("CENTR", ["CENTRE"], "Regular Text Recognition")
if r["lenient"] != 1.0: fails.append("one-edit-off must be lenient 1.0, got %r" % r)
if r["strict"] != 0.0: fails.append("one-edit-off must be strict 0.0, got %r" % r)
r = score_sample("xdxdxd", ["CENTRE"], "Regular Text Recognition")
if r["lenient"] != 0.0 or r["strict"] != 0.0:
    fails.append("garbage must score 0, got %r" % r)
r = score_sample("", ["CENTRE"], "Regular Text Recognition")
if r["strict"] != 0.0: fails.append("empty prediction must score 0, got %r" % r)
r = score_sample("y_2=-1", ["y _ { 2 } = - 1\\n", "y_2 = - 1\\n"], "Handwritten Mathematical Expression Recognition")
if r["lenient"] != 1.0: fails.append("whitespace-insensitive math must score, got %r" % r)
r = score_sample("Chatswood Epping", ["Chatswood Epping", "Chatswood, Epping"], "Scene Text-centric VQA")
if r["strict"] != 1.0: fails.append("any-of-answers must score, got %r" % r)
r = score_sample("dips dips machining", ["CENTRE"], "Regular Text Recognition")
if r["lenient"] != 0.0: fails.append("prompt-echo garbage must score 0, got %r" % r)
if fails:
    print("\\n".join(fails))
    raise SystemExit(1)
print("CONTROLS_OK")
`).trim();
  if (!out.includes("CONTROLS_OK")) fail(`controls output missing marker: ${out}`);
  console.log("OCRBENCH_CONTROLS_OK");
}

if (mode === "vision") {
  // Positive control: two different images must give different features.
  // If features are constant, the model is blind regardless of downstream code.
  const out = py(`
import sys, torch
sys.path.insert(0, ".")
import tinydoc_vlm
from transformers import AutoModelForCausalLM, SiglipImageProcessor
from PIL import Image
from pathlib import Path
m = AutoModelForCausalLM.from_pretrained("eulogik/TinyDoc-VLM-256M", dtype=torch.float32).eval()
p = SiglipImageProcessor.from_pretrained("eulogik/TinyDoc-VLM-256M")
base = Path("evaluation/data/ocrbench/images")
paths = sorted(base.glob("*.png"))[:2]
if len(paths) < 2: raise SystemExit("need 2 images")
feats = []
for path in paths:
    img = Image.open(path).convert("RGB")
    pv = p.preprocess(img, return_tensors="pt")["pixel_values"].unsqueeze(1)
    with torch.no_grad():
        f = m.vision_encoder(pv)
    feats.append(f.float())
a, b = feats[0].flatten(), feats[1].flatten()
if a.shape != b.shape: raise SystemExit("shape mismatch %s %s" % (a.shape, b.shape))
const_a = float(a.std())
cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
if const_a < 1e-6: raise SystemExit("DEAD_TOWER constant features std=%g" % const_a)
if cos > 0.9999: raise SystemExit("DEAD_TOWER near-identical features cosine=%g std=%g" % (cos, const_a))
print("VISION_OK std=%g cosine=%g" % (const_a, cos))
`).trim();
  if (!out.includes("VISION_OK")) fail(`vision probe failed: ${out}`);
  console.log(`VISION_TOWER_ALIVE ${out}`);
}

if (mode === "tokenid") {
  const path = findSummary("") || findSummary("pilot") || findSummary("full");
  if (!path) fail("no ocrbench summary found in results/");
  const summary = JSON.parse(readFileSync(path, "utf8"));
  const info = summary.image_token_id_reconciliation;
  if (!info) fail("summary missing image_token_id_reconciliation");
  if (info.forced_image_token_id == null) fail("forced id not recorded");
  if (!info.aligned) fail(`processor id ${info.processor_image_token_id} != forced ${info.forced_image_token_id}`);
  console.log(`TOKENID_RECONCILED processor=${info.processor_image_token_id} forced=${info.forced_image_token_id} model_before=${info.model_image_token_id_before}`);
}

if (mode === "pilot") {
  const candidates = ["pilot", "align", "mismatch"].map(findSummary).filter(Boolean);
  if (!candidates.length) fail("no pilot summary found");
  const s = JSON.parse(readFileSync(candidates[0], "utf8"));
  if (!(s.total_samples > 0)) fail("pilot scored 0 samples");
  if (!(s.elapsed_seconds > 0)) fail("pilot missing elapsed time");
  console.log(`PILOT_COMPLETE samples=${s.total_samples} strict=${s.ocrbench_score_strict} lenient=${s.ocrbench_score_lenient} ${s.elapsed_seconds}s`);
}

if (mode === "full") {
  const path = findSummary("full") || findSummary("");
  if (!path) fail("no full summary found");
  const s = JSON.parse(readFileSync(path, "utf8"));
  if (s.total_samples !== 1000) fail(`expected 1000 samples, got ${s.total_samples}`);
  if (!(s.elapsed_seconds > 0)) fail("missing elapsed time");
  console.log(`FULL_1000_COMPLETE strict=${s.ocrbench_score_strict} lenient=${s.ocrbench_score_lenient} errors=${s.error_count} ${s.elapsed_seconds}s`);
}

if (mode === "recompute") {
  const jsonlName = ["full", ""].map((t) => (t ? `ocrbench_256m_${t}.jsonl` : "ocrbench_256m.jsonl")).find((n) => existsSync(join(RESULTS, n)));
  if (!jsonlName) fail("no predictions jsonl found");
  const jsonName = jsonlName.replace(".jsonl", ".json");
  const stored = JSON.parse(readFileSync(join(RESULTS, jsonName), "utf8"));
  const lines = readFileSync(join(RESULTS, jsonlName), "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
  if (lines.length !== stored.total_samples) fail(`jsonl has ${lines.length} rows, summary says ${stored.total_samples}`);
  const scored = lines.filter((r) => !r.error);
  const strict = (scored.reduce((a, r) => a + r.strict, 0) / scored.length) * 100;
  const lenient = (scored.reduce((a, r) => a + r.lenient, 0) / scored.length) * 100;
  if (Math.abs(strict - stored.ocrbench_score_strict) > 1e-6) fail(`strict mismatch recomputed=${strict} stored=${stored.ocrbench_score_strict}`);
  if (Math.abs(lenient - stored.ocrbench_score_lenient) > 1e-6) fail(`lenient mismatch recomputed=${lenient} stored=${stored.ocrbench_score_lenient}`);
  console.log(`RECOMPUTE_EXACT strict=${strict.toFixed(4)} lenient=${lenient.toFixed(4)} n=${scored.length}`);
}

console.log(`gate:${mode} ok`);
