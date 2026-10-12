"""Offline workflow on the built-in demo, plus independent re-computation of every number."""
import hashlib
import json
import math
import re
import shutil
import unicodedata
from collections import Counter

import numpy as np

from common import *

c = Checker("2 Offline workflow, reports and statistics")
W = WORK / "offline"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)

r = sh("atlasforge demo", cwd=W, log="t02_demo")
D = W / "atlasforge-demo"
files = sorted(str(p.relative_to(D)) for p in D.rglob("*") if p.is_file())
c("2.01", "demo writes a dataset and two finished runs with no model, token or network",
  r.rc == 0 and files == ["runs/base/results.jsonl", "runs/base/run.json", "runs/tuned/results.jsonl", "runs/tuned/run.json", "toy_qa.jsonl"],
  f"files={files}", "t02_demo")
n = len(read_jsonl(D / "toy_qa.jsonl"))
c("2.02", "The demo dataset has 92 questions (docs claim)", n == 92, f"n={n}")
mf = json.loads((D / "runs/base/run.json").read_text())
c("2.03", "Demo runs are labelled synthetic in their manifest so they cannot pass for real results",
  "synthetic" in json.dumps(mf).lower(), f"model={mf.get('model')}")

r = sh("atlasforge dataset validate atlasforge-demo/toy_qa.jsonl", cwd=W, log="t02_validate")
c("2.04", "dataset validate accepts the demo dataset (exit 0, OK)", r.rc == 0 and "OK" in r.out, r.out.strip()[-120:], "t02_validate")

DS = "atlasforge-demo/toy_qa.jsonl"
r = sh(f"atlasforge report atlasforge-demo/runs/base --dataset {DS}", cwd=W, log="t02_report")
rep = D / "runs/base"
c("2.05", "report writes report.md, report.json and report.html with no model",
  r.rc == 0 and all((rep / f).exists() for f in ("report.md", "report.json", "report.html")), r.out.strip()[-160:], "t02_report")
rj = json.loads((rep / "report.json").read_text())
views = {m["view"] for m in rj["metrics"]}
c("2.06", "Every text metric is reported under both tone-aware and tone-insensitive views",
  views == {"tone_aware", "tone_insensitive"}, f"views={sorted(views)}")
c("2.07", "report states failed examples (they count as wrong, never dropped)",
  rj["n_failed"] > 0 and rj["n_ok"] + rj["n_failed"] + rj["n_missing"] == rj["n_total"] and "failed" in r.out.lower(),
  f"ok={rj['n_ok']} failed={rj['n_failed']} missing={rj['n_missing']} total={rj['n_total']}")

r = sh(f"atlasforge compare {DS} --base atlasforge-demo/runs/base --candidate atlasforge-demo/runs/tuned --slice domain --out cmp1", cwd=W, log="t02_compare")
cmp1 = W / "cmp1"
c("2.08", "compare writes comparison.md, comparison.json and comparison.html",
  r.rc == 0 and all((cmp1 / f).exists() for f in ("comparison.md", "comparison.json", "comparison.html")), r.out.strip()[-160:], "t02_compare")
cj = json.loads((cmp1 / "comparison.json").read_text())
has_num = [s for s in cj["slices"] if s["field"] == "has_number" and s["value"] == "yes"]
overall = [m for m in cj["metrics"] if m["key"] == "exact_match@tone_aware"] or cj["metrics"][:1]
c("2.09", "Demo behaves as documented: tuned is better overall but REGRESSES on has_number (a regression is surfaced, not hidden)",
  bool(overall) and overall[0]["verdict"] == "improved" and has_num and has_num[0]["status"] == "regressed",
  f"overall={overall[0]['key']}:{overall[0]['verdict']} delta={overall[0]['delta']:.3f}; has_number=yes status={has_num[0]['status'] if has_num else None}")

# ----------------------------------------------- HTML claims: one self-contained page
for name, path in (("report.html", rep / "report.html"), ("comparison.html", cmp1 / "comparison.html")):
    html = path.read_text(encoding="utf-8")
    ext = re.findall(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//|@import|url\(\s*["']?(?:https?:)?//""", html, re.I)
    scripts = re.findall(r"<script", html, re.I)
    c(f"2.10{'a' if name.startswith('report') else 'b'}", f"{name} is one self-contained page (no <script>, no external reference)",
      not scripts and not ext, f"size={len(html)}B scripts={len(scripts)} external_refs={len(ext)}")

# ----------------------------------------------- determinism / byte-stable reports
r = sh(f"atlasforge compare {DS} --base atlasforge-demo/runs/base --candidate atlasforge-demo/runs/tuned --slice domain --out cmp2", cwd=W, log="t02_compare2")
same = (cmp1 / "comparison.json").read_bytes() == (W / "cmp2" / "comparison.json").read_bytes()
c("2.11", "compare is deterministic: the same inputs and seed give byte-identical comparison.json", same, "identical" if same else "DIFFERENT")
r = sh(f"atlasforge compare {DS} --base atlasforge-demo/runs/base --candidate atlasforge-demo/runs/tuned --out cmp3 --seed 7", cwd=W, log="t02_compare3")
cj3 = json.loads((W / "cmp3" / "comparison.json").read_text())
c("2.12", "A different --seed changes only the bootstrap interval, never the means or counts",
  cj3["metrics"][0]["base_mean"] == cj["metrics"][0]["base_mean"] and cj3["metrics"][0]["wins"] == cj["metrics"][0]["wins"],
  f"low {cj['metrics'][0]['low']:.4f} vs {cj3['metrics'][0]['low']:.4f}")

# ----------------------------------------------- independent recomputation of the numbers
def norm(t: str, strip_tones: bool) -> str:
    t = t.lower()
    t = unicodedata.normalize("NFD", t)
    if strip_tones:
        t = re.sub("[̀́̂̄̌]", "", t)
    t = unicodedata.normalize("NFC", t)
    t = "".join(" " if unicodedata.category(ch)[0] in "PS" and ch != "'" else ch for ch in t)
    return " ".join(t.split())

ds = read_jsonl(D / "toy_qa.jsonl")
def results(dirname):
    m = {r["id"]: r for r in read_jsonl(D / "runs" / dirname / "results.jsonl")}
    return m
rb, rt = results("base"), results("tuned")

def em(res, strip):
    out = []
    for ex in ds:
        rec = res.get(ex["id"])
        pred = "" if rec is None or rec["error"] else rec["prediction"] or ""
        out.append(1.0 if norm(pred, strip) == norm(ex["reference"], strip) else 0.0)
    return out

for view, strip in (("tone_aware", False), ("tone_insensitive", True)):
    b, t = em(rb, strip), em(rt, strip)
    m = next(x for x in cj["metrics"] if x["key"] == f"exact_match@{view}")
    ok = (abs(m["base_mean"] - np.mean(b)) < 1e-12 and abs(m["candidate_mean"] - np.mean(t)) < 1e-12
          and m["wins"] == sum(1 for x, y in zip(b, t) if y > x) and m["losses"] == sum(1 for x, y in zip(b, t) if y < x)
          and m["ties"] == sum(1 for x, y in zip(b, t) if y == x))
    c(f"2.13{'a' if not strip else 'b'}", f"Exact-match means and win/tie/loss counts match an independent recomputation ({view})", ok,
      f"base={np.mean(b):.4f} cand={np.mean(t):.4f} W/T/L={m['wins']}/{m['ties']}/{m['losses']}")

# exact McNemar (own implementation)
b, t = em(rb, False), em(rt, False)
b01 = sum(1 for x, y in zip(b, t) if x == 1 and y == 0); b10 = sum(1 for x, y in zip(b, t) if x == 0 and y == 1)
d = b01 + b10; k = min(b01, b10)
p = min(1.0, 2 * sum(math.comb(d, i) for i in range(k + 1)) / 2 ** d) if d else 1.0
m = next(x for x in cj["metrics"] if x["key"] == "exact_match@tone_aware")["mcnemar"]
c("2.14", "Exact McNemar p-value matches an independent exact binomial computation",
  m["base_only"] == b01 and m["candidate_only"] == b10 and math.isclose(m["p_value"], p, rel_tol=1e-9), f"discordant={b01}/{b10} p_independent={p:.3e} p_tool={m['p_value']:.3e}")

# bootstrap CI close to an independent bootstrap
diffs = np.array(t) - np.array(b)
rng = np.random.default_rng(12345)
means = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(20000)])
lo, hi = np.quantile(means, [0.025, 0.975])
mm = next(x for x in cj["metrics"] if x["key"] == "exact_match@tone_aware")
c("2.15", "The 95% bootstrap interval agrees with an independent 20,000-resample bootstrap (within 0.03)",
  abs(mm["low"] - lo) < 0.03 and abs(mm["high"] - hi) < 0.03 and math.isclose(mm["delta"], diffs.mean(), abs_tol=1e-12),
  f"tool=[{mm['low']:.3f},{mm['high']:.3f}] independent=[{lo:.3f},{hi:.3f}] delta={mm['delta']:.3f}")

# slice rule
dom = [s for s in cj["slices"] if s["field"] == "domain"]
counts = Counter(ex["meta"]["domain"] for ex in ds)
c("2.16", "Slices under 30 examples are reported as insufficient data, larger ones get a verdict",
  all((s["status"] == "insufficient data") == (counts[s["value"]] < 30) for s in dom),
  "; ".join(f"{s['value']} n={s['n']} {s['status']}" for s in dom))

# identical runs -> zero-width interval, no clear change
r = sh(f"atlasforge compare {DS} --base atlasforge-demo/runs/base --candidate atlasforge-demo/runs/base --out cmp_same", cwd=W, log="t02_compare_same")
cs = json.loads((W / "cmp_same" / "comparison.json").read_text())
c("2.17", "Comparing a run with itself gives delta 0, a zero-width interval and 'no clear change'",
  all(x["delta"] == 0 and x["low"] == 0 and x["high"] == 0 and x["verdict"] == "no clear change" for x in cs["metrics"] if x["delta"] is not None),
  f"verdicts={sorted({x['verdict'] for x in cs['metrics']})}")

# compare refuses runs from a different dataset
shutil.copy(D / "toy_qa.jsonl", W / "edited.jsonl")
with open(W / "edited.jsonl", "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"id": "extra-1", "input": "x", "reference": "y", "meta": {"domain": "agri"}}) + "\n")
r = sh(f"atlasforge compare edited.jsonl --base atlasforge-demo/runs/base --candidate atlasforge-demo/runs/tuned --out cmp_bad", cwd=W, log="t02_compare_bad")
c("2.18", "compare refuses to pair runs made from a different dataset (hash check), with a hint",
  r.rc == 2 and "not produced from this dataset" in r.both and "Traceback" not in r.both, r.err.strip()[:200], "t02_compare_bad")

# report is re-runnable with different metrics and does not touch results.jsonl
before = hashlib.sha256((rep / "results.jsonl").read_bytes()).hexdigest()
r = sh(f"atlasforge report atlasforge-demo/runs/base --dataset {DS} -m exact_match -m chrf++", cwd=W, log="t02_report_metrics")
after = hashlib.sha256((rep / "results.jsonl").read_bytes()).hexdigest()
rj2 = json.loads((rep / "report.json").read_text())
c("2.19", "report re-scores with different metrics (chrf++) and never modifies results.jsonl",
  r.rc == 0 and before == after and any(m["name"] == "chrf++" for m in rj2["metrics"]), f"results.jsonl unchanged={before == after}; metrics={sorted({m['name'] for m in rj2['metrics']})}", "t02_report_metrics")

r = sh(f"atlasforge report atlasforge-demo/runs/base --dataset {DS} -m accuracy", cwd=W, log="t02_report_badmetric")
c("2.20", "Asking for a classification-only metric on a generation dataset fails with a clear hint",
  r.rc == 2 and "classification" in r.both and "Traceback" not in r.both, r.err.strip()[:200], "t02_report_badmetric")

r = sh("atlasforge report /nonexistent --dataset atlasforge-demo/toy_qa.jsonl", cwd=W, log="t02_report_nodir")
c("2.21", "Pointing report at a non-run directory gives a clean error, not a traceback",
  r.rc == 2 and "Traceback" not in r.both and "not a run directory" in r.both, r.err.strip()[:200], "t02_report_nodir")
