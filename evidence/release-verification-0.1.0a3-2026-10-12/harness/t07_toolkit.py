"""Fine-tune dry-run, model cards, benchmark wrapper, Python API, and what the package ships."""
import json
import re
import shutil
import zipfile
from pathlib import Path

from common import *

c = Checker("7 Fine-tune plan, model cards, Python API and packaging")
W = WORK / f"toolkit_{LABEL}"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)

# ------------------------------------------------------------------ fine-tune (dry run only; training needs an NVIDIA GPU)
train = [{"id": f"t{i}", "input": f"Train question {i}?", "reference": f"Train answer {i}", "lang": "en"} for i in range(30)]
test = [{"id": f"x{i}", "input": f"Held out question {i}?", "reference": f"Held answer {i}", "lang": "en"} for i in range(10)]
write_jsonl(W / "train.jsonl", train)
write_jsonl(W / "test.jsonl", test)
(W / "ft.json").write_text(json.dumps({"train_file": "train.jsonl", "eval_file": "test.jsonl", "output_dir": "adapters/a"}))
r = sh("atlasforge finetune ft.json --dry-run", cwd=W, log="t07_dry")
c("7.01", "finetune --dry-run validates data and prints the plan with no GPU, writes nothing, exit 0",
  r.rc == 0 and "dry run" in r.out.lower() and not (W / "adapters").exists(), " ".join(r.out.split())[:200], "t07_dry")

leak = test + [{"id": "L1", "input": "TRAIN QUESTION 3?", "reference": "Train answer 3"}]
write_jsonl(W / "test_leak.jsonl", leak)
(W / "ft_leak.json").write_text(json.dumps({"train_file": "train.jsonl", "eval_file": "test_leak.jsonl", "output_dir": "adapters/b"}))
r = sh("atlasforge finetune ft_leak.json --dry-run", cwd=W, log="t07_leak")
c("7.02", "Train/test leakage is refused before any training (even when it differs only in case/punctuation)",
  r.rc != 0 and "appear in" in r.both and "Traceback" not in r.both, (r.err or r.out).strip()[:200], "t07_leak")

write_jsonl(W / "tiny.jsonl", train[:5])
(W / "ft_tiny.json").write_text(json.dumps({"train_file": "tiny.jsonl", "output_dir": "adapters/c"}))
r = sh("atlasforge finetune ft_tiny.json --dry-run", cwd=W, log="t07_tiny")
c("7.03", "A tiny training set (5 examples) is refused: 'at least 20 are required'", r.rc != 0 and "20" in r.both and "Traceback" not in r.both, (r.err or r.out).strip()[:200], "t07_tiny")

(W / "ft_typo.json").write_text(json.dumps({"train_file": "train.jsonl", "output_dir": "o", "lora_rank": 8, "learning_rate": -1, "quantize": "9bit"}))
r = sh("atlasforge finetune ft_typo.json --dry-run", cwd=W, log="t07_typo")
c("7.04", "Config typos and every invalid value are reported together (unknown key, bad learning_rate, bad quantize)",
  r.rc != 0 and "lora_rank" in r.both and "learning_rate" in r.both and "quantize" in r.both, " ".join((r.err or r.out).split())[:260], "t07_typo")

r = sh("atlasforge finetune ft.json", cwd=W, log="t07_train_nogpu", timeout=300)
has_torch = sh([PY, "-c", "import torch"]).rc == 0
want = ("NVIDIA" in r.both) if has_torch else ("torch" in r.both and "finetune" in r.both)
c("7.05", "Real training without the needed hardware/extras fails early with a clear message and no partial output "
  "(torch missing -> install hint; torch present but no NVIDIA GPU -> says it needs an NVIDIA GPU)",
  r.rc != 0 and "Traceback" not in r.both and not (W / "adapters").exists() and want, f"torch_installed={has_torch}; " + (r.err or r.out).strip()[:200], "t07_train_nogpu")
c.na("7.06", "An actual QLoRA training run and evaluating the resulting adapter (--adapter)", "Needs an NVIDIA GPU with CUDA; this Mac has none. The documentation says the same: the training run has never been executed.")

# ------------------------------------------------------------------ model card
cmpj = WORK / "llm_venv-pypi" / "cmp" / "comparison.json"
if not cmpj.exists():
    cmpj = WORK / "offline" / "cmp1" / "comparison.json"
(W / "terms.txt").write_text("Own data, licence CC BY 4.0.")
r = sh(f"atlasforge card --name 'Hausa Agri Helper' --description 'Answers farming questions.' --training-data @terms.txt --lang ha --domain agriculture --author 'Brainers Labs' --comparison {cmpj} --out card.md", cwd=W, log="t07_card")
card = (W / "card.md").read_text(encoding="utf-8") if (W / "card.md").exists() else ""
h1 = next((l for l in card.splitlines() if l.startswith("# ")), "")
c("7.07", "card adds the required 'Powered by Awarri' suffix to the model name when missing", r.rc == 0 and h1.strip() == "# Hausa Agri Helper - Powered by Awarri", f"title line: {h1!r}", "t07_card")
low = card.lower()
c("7.08", "The card carries the licence obligations: attribution to Awarri Technologies, 1,000 user cap, no redistribution of base weights",
  "awarri" in low and "1,000" in card and "attribution" in low and "base" in low, f"has Awarri={'awarri' in low} cap={'1,000' in card}")
c("7.09", "The card includes the evaluation numbers and states regressions from comparison.json (does not hide them)",
  ("regress" in low) and ("exact_match" in low or "chrf" in low), f"mentions regression={'regress' in low}")
c("7.10", "The card includes the stated training-data licence from the file given with @", "cc by 4.0" in low, "found 'CC BY 4.0'" if "cc by 4.0" in low else "missing")
r = sh("atlasforge card --name x --description y", cwd=W, log="t07_card_missing")
c("7.11", "card refuses to run without a stated training-data source and licence (required option)", r.rc == 2 and "training-data" in r.both, r.both.strip()[-120:], "t07_card_missing")

# ------------------------------------------------------------------ benchmark wrapper
r = sh("atlasforge bench afrobench --list", cwd=W, log="t07_bench_list")
c("7.12", "bench afrobench without the harness fails with a clear instruction, not a traceback", r.rc != 0 and "Traceback" not in r.both and ("lm-evaluation-harness" in r.both or "bench" in r.both), " ".join((r.err or r.out).split())[:200], "t07_bench_list")
c.na("7.13", "Running AfroBench-LITE against N-ATLaS through lm-evaluation-harness", "Needs the bench extra, a GPU and the gated weights loaded in-process; not possible on this machine. Docs mark it unverified.")

# ------------------------------------------------------------------ Python API
api = r'''
import json, sys, tempfile
sys.path.insert(0, %r)
from common import StubServer
import atlasforge
from atlasforge.types import BackendInfo, Generation
from pathlib import Path

class Fake:
    def generate(self, messages, params=None):
        q = messages[-1]["content"]
        return Generation(text=q.replace("q", "ans"), latency_ms=1.0)
    def transcribe(self, audio, lang): raise NotImplementedError
    def info(self): return BackendInfo(backend="fake", model="fake-1", capabilities=frozenset({"generate"}))
    def close(self): print("CLOSED-BY-ATLASFORGE")

tmp = Path(tempfile.mkdtemp())
rows = [{"id": f"e{i}", "input": f"q{i}", "reference": f"ans{i}"} for i in range(12)]
(tmp / "d.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
backend = Fake()
ev = atlasforge.evaluate(tmp / "d.jsonl", out_dir=tmp / "run", backend=backend, metrics=["exact_match", "chrf"])
print("EM", ev.metric("exact_match@tone_aware").mean, "files", sorted(p.name for p in (tmp / "run").iterdir()))
print("API", sorted(n for n in atlasforge.__all__ if not n.startswith("_")))
rep = atlasforge.score_finished_run(tmp / "run", tmp / "d.jsonl")
print("RESCORED", rep.n_ok)
cmp = atlasforge.compare_runs(tmp / "d.jsonl", tmp / "run", tmp / "run")
print("CMP", [m.verdict for m in cmp.metrics][:2])
''' % str(ROOT)
(W / "api.py").write_text(api)
r = sh([PY, "api.py"], cwd=W, log="t07_api")
c("7.14", "Python API: atlasforge.evaluate() with your own Backend object runs, scores and writes run.json/results/report.md/json/html",
  r.rc == 0 and "EM 1.0" in r.out and "report.html" in r.out, " ".join(r.out.split())[:250] or r.err[-250:], "t07_api")
c("7.15", "A Backend object you pass in stays yours: AtlasForge does not close it", "CLOSED-BY-ATLASFORGE" not in r.out, "close() not called")
c("7.16", "score_finished_run and compare_runs work from the Python API", "RESCORED 12" in r.out and "CMP ['no clear change'" in r.out, " ".join(r.out.split())[-120:])

# ------------------------------------------------------------------ what the package ships
whl = next((Path(p) for p in sh(f"find {WORK} {ROOT} -name 'brainers_atlasforge*.whl' 2>/dev/null | head -1").out.split()), None)
if whl is None:
    sh(f"pip download brainers-atlasforge --no-deps -d {ROOT}/dl -q", log="t07_dl")
    whl = next((ROOT / "dl").glob("*.whl"), None)
if whl:
    z = zipfile.ZipFile(whl)
    names = z.namelist()
    bad = [n for n in names if re.search(r"\.(safetensors|bin|gguf|pt|pth|ckpt|onnx|npy|wav|mp3|ogg)$", n)]
    big = [(n, z.getinfo(n).file_size) for n in names if z.getinfo(n).file_size > 500_000]
    c("7.17", "The distribution contains no model weights or audio (licence: nothing is redistributed), and is small",
      not bad and not big and whl.stat().st_size < 1_000_000, f"{len(names)} files, {whl.stat().st_size // 1024} KB; weights/audio={bad}; files >500KB={big}")
    c("7.18", "The wheel ships type information (py.typed) and the licence", any(n.endswith("py.typed") for n in names) and any("LICENSE" in n for n in names), "py.typed and LICENSE present")
    ids = set()
    for n in names:
        if n.endswith(".py"):
            ids |= set(re.findall(r"\bNCAIR1/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", z.read(n).decode("utf-8", "replace")))
    official = {"NCAIR1/N-ATLaS", "NCAIR1/Hausa-ASR", "NCAIR1/Yoruba-ASR", "NCAIR1/Igbo-ASR", "NCAIR1/NigerianAccentedEnglish"}
    c("7.19", "Only the five official NCAIR1 model ids appear in the shipped code (no other foundation model)", ids <= official and len(ids) >= 5, f"ids={sorted(ids)}")
    other = [n for n in names if n.endswith(".py")]
    src = "".join(z.read(n).decode("utf-8", "replace") for n in other).lower()
    judge = [w for w in ("gpt-4", "gpt-3", "claude-", "gemini", "llm-as-judge", "openai.com", "anthropic.com") if w in src]
    c("7.20", "No LLM-as-judge and no third-party model API is referenced in the shipped code", not judge, f"found={judge}")
else:
    c("7.17", "Wheel inspected", False, "could not obtain wheel")
