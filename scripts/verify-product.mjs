#!/usr/bin/env node
// Product-ship gate checks. Each mode corresponds to one GATES.md entry:
//   node scripts/verify-product.mjs baseline|engine|evidence_ab|claims|tests|gitignore|ci|audit|all
import { execSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RESULTS = path.join(ROOT, "evaluation/phase0/results");
const ok = (msg) => { console.log(`OK: ${msg}`); return true; };
const fail = (msg) => { console.error(`FAIL: ${msg}`); return false; };
const readJson = (p) => JSON.parse(readFileSync(p, "utf8"));

function gateBaseline() {
  const p = path.join(RESULTS, "scores_pipeline_ollama_qwen2.5vl_3b.json");
  const pAlt = path.join(RESULTS, "scores_ollama.json");
  const c = path.join(RESULTS, "scores_ppocr_heuristics.json");
  const oursPath = existsSync(p) ? p : pAlt;
  if (!existsSync(oursPath)) return fail("pipeline scores artifact missing");
  if (!existsSync(c)) return fail("ppocr baseline artifact missing");
  const ours = readJson(oursPath);
  const comp = readJson(c);
  if (comp.n !== 100 && comp.n_examples !== 100) return fail("baseline not n=100");
  if (!(comp.field_f1 > 0)) return fail("baseline field_f1 not measured");
  if (!(ours.field_f1 > comp.field_f1)) {
    return fail(`pipeline ${ours.field_f1} does not beat baseline ${comp.field_f1}`);
  }
  return ok(`BASELINE_MEASURED_PIPELINE_WINS (pipeline ${ours.field_f1} > baseline ${comp.field_f1})`);
}

function gateEngine() {
  const p = path.join(RESULTS, "engine_ladder.json");
  if (!existsSync(p)) return fail("engine_ladder.json missing (7b pilot not concluded)");
  const e = readJson(p);
  if (!e.decision || !["adopt_qwen2.5vl:7b", "keep_qwen2.5vl:3b"].includes(e.decision)) {
    return fail(`engine decision invalid: ${e.decision}`);
  }
  if (!e.n || e.n < 10) return fail(`engine ladder evaluated on n=${e.n}, expected >=10 paired`);
  if (!e.latency_ms_3b || !e.latency_ms_7b) return fail("latency for both engines required");
  if (typeof e.f1_7b !== "number" || typeof e.f1_3b !== "number") fail("f1 for both engines required");
  if (!e.hardware?.full_n100_attempt && e.n < 100) {
    return fail("partial-n decision requires hardware.attempt record of the full run");
  }
  if (e.decision === "adopt_qwen2.5vl:7b" && !(e.f1_7b > e.f1_3b)) {
    return fail("adopting 7b without a paired f1 win");
  }
  return ok(`ENGINE_LADDER_DECIDED (${e.decision}, paired n=${e.n}: f1 3b=${e.f1_3b} / 7b=${e.f1_7b}, latency ${e.latency_ms_3b}ms vs ${e.latency_ms_7b}ms)`);
}

function gateEvidenceAb() {
  const p = path.join(RESULTS, "evidence_ab.json");
  if (!existsSync(p)) return fail("evidence_ab.json missing");
  const e = readJson(p);
  if (!["adopt_rapidocr", "keep_tesseract"].includes(e.decision)) {
    return fail(`evidence decision invalid: ${e.decision}`);
  }
  const t = e.summary?.tesseract, r = e.summary?.rapidocr;
  if (!t || !r) return fail("both engine arms must be summarized");
  if (e.decision === "adopt_rapidocr" && !(r.gold_in_quote_rate > t.gold_in_quote_rate)) {
    return fail("adopting rapidocr without a gold_in_quote win");
  }
  return ok(`EVIDENCE_AB_DECIDED (${e.decision}, gold_in_quote ${t.gold_in_quote_rate} -> ${r.gold_in_quote_rate})`);
}

function gateClaims() {
  let bad = [];
  const br = path.join(ROOT, "docs/benchmark_results.json");
  if (!existsSync(br)) bad.push("docs/benchmark_results.json missing");
  else {
    const j = readJson(br);
    if (!Array.isArray(j.measured) || j.measured.length < 5) bad.push("measured block missing/thin");
    if (!j.legacy_unverified?.warning) bad.push("legacy_unverified warning missing");
  }

  const index = existsSync(path.join(ROOT, "docs/index.html"))
    ? readFileSync(path.join(ROOT, "docs/index.html"), "utf8") : "";
  if (index.includes("⏳ Running")) bad.push("index.html still shows ⏳ Running");
  if (!index.includes("0.870")) bad.push("index.html missing measured 0.870");
  if (!index.includes("0.376")) bad.push("index.html missing baseline 0.376");

  const bench = existsSync(path.join(ROOT, "docs/BENCHMARKS.md"))
    ? readFileSync(path.join(ROOT, "docs/BENCHMARKS.md"), "utf8") : "";
  if (!/0\.0%/.test(bench) || !/1,?000/.test(bench)) bad.push("BENCHMARKS.md missing n=1000 0.0% result");

  const drafts = ["docs/paper.md", "docs/pitch_deck.md", "docs/reddit_post.md",
                  "docs/twitter_thread.md", "docs/launch_announcement.md"];
  const banner = /NOT MEASURED|DO NOT|fabricated|WITHDRAWN|unmeasured/i;
  for (const d of drafts) {
    const p = path.join(ROOT, d);
    if (!existsSync(p)) { bad.push(`${d} missing`); continue; }
    if (!banner.test(readFileSync(p, "utf8"))) bad.push(`${d} lacks a do-not-publish banner`);
  }

  const readme = existsSync(path.join(ROOT, "README.md"))
    ? readFileSync(path.join(ROOT, "README.md"), "utf8") : "";
  for (const fabricated of ["65.3", "60.8", "85.2", "87.6", "85.9"]) {
    if (readme.includes(fabricated)) bad.push(`README cites withdrawn number ${fabricated}`);
  }

  // measured SROIE contamination must stay disclosed in public docs
  if (!/58%|0\.58/.test(readme) || !/leak|caveat/i.test(readme)) {
    bad.push("README missing the SROIE train-overlap caveat (58% of eval receipts are public-train)");
  }
  if (!/not a clean held-out|not a clean/i.test(bench)) {
    bad.push("BENCHMARKS.md missing the SROIE leakage caveat");
  }

  if (bad.length) return fail(`PUBLIC_CLAIMS_CLEAN: ${bad.join("; ")}`);
  return ok("PUBLIC_CLAIMS_CLEAN");
}

function gateTests() {
  for (const f of ["tests/test_evidence.py", "tests/test_overlay.py", "tests/test_cli.py"]) {
    if (!existsSync(path.join(ROOT, f))) return fail(`${f} missing`);
  }
  const r = spawnSync("python3", ["-m", "pytest", "tests/test_evidence.py",
    "tests/test_overlay.py", "tests/test_cli.py", "-q"], { cwd: ROOT, encoding: "utf8" });
  if (r.status !== 0) return fail(`new tests failed:\n${r.stdout}${r.stderr}`);
  return ok("EVIDENCE_TESTS_PRESENT (evidence+overlay+cli tests pass)");
}

function gateGitignore() {
  let out = "";
  try { out = execSync("git add -An .", { cwd: ROOT, encoding: "utf8" }); } catch (e) { return fail(`git add -An errored: ${e.message}`); }
  const bad = out.split("\n").filter((l) =>
    /\.unlazy|data\/training|data\/eval_|data\/datasets\/docmatix|data\/datasets\/docvqa|data\/datasets\/sroie|overnight_output|evaluation\/data/.test(l));
  if (bad.length) return fail(`git add -An would stage ignored paths:\n${bad.join("\n")}`);
  return ok("GITADD_DRYRUN_CLEAN");
}

function gateCi() {
  const lint = spawnSync("python3", ["-m", "ruff", "check", "tinydoc_vlm/", "tests/"],
    { cwd: ROOT, encoding: "utf8" });
  if (lint.status !== 0) return fail(`ruff failed:\n${lint.stdout}${lint.stderr}`);
  const t = spawnSync("python3", ["-m", "pytest", "tests/", "-q"],
    { cwd: ROOT, encoding: "utf8", timeout: 300000 });
  if (t.status !== 0) return fail(`pytest failed:\n${t.stdout}${t.stderr}`);
  const n = (t.stdout.match(/(\d+) passed/) || [])[1] || "?";
  let remote = "";
  const gh = spawnSync("gh", ["run", "list", "--limit", "1", "--json", "conclusion,status",
    "--jq", ".[0]"], { cwd: ROOT, encoding: "utf8" });
  if (gh.status === 0 && gh.stdout.trim()) {
    try {
      const run = JSON.parse(gh.stdout);
      if (run.status === "completed" && run.conclusion !== "success") {
        return fail(`remote CI latest run: ${run.conclusion}`);
      }
      remote = `, remote latest: ${run.status}/${run.conclusion || "running"}`;
    } catch { /* ignore parse issues */ }
  }
  return ok(`CI_GREEN (local ruff clean, ${n} tests passed${remote})`);
}

function gateAudit() {
  const r = spawnSync("python3", ["evaluation/phase0/recompute_scores.py"],
    { cwd: ROOT, encoding: "utf8" });
  if (r.status !== 0) return fail(`claim audit failed:\n${r.stdout}${r.stderr}`);
  return ok("CLAIM_AUDIT_PASSED");
}

const GATES = {
  baseline: gateBaseline, engine: gateEngine, evidence_ab: gateEvidenceAb,
  claims: gateClaims, tests: gateTests, gitignore: gateGitignore,
  ci: gateCi, audit: gateAudit,
};

const mode = process.argv.slice(2);
const modes = mode.length === 0 || mode.includes("all")
  ? Object.keys(GATES)
  : mode.flatMap((m) => m.split(",")).filter(Boolean);
let allOk = true;
for (const m of modes) {
  if (!GATES[m]) { console.error(`unknown mode: ${m}`); allOk = false; continue; }
  if (!GATES[m]()) allOk = false;
}
process.exit(allOk ? 0 : 1);
