"""Each fix from the 0.1.0a2 verification, proved on the published package."""
import json
import shutil
from pathlib import Path

from common import *

c = Checker("11 Fixes from the 0.1.0a2 verification, on the release")
W = WORK / f"fixes_{LABEL}"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)

# ---- run - : the shell's trailing newline is not sent to the model
s = StubServer(answer=lambda p: "ok").start()
sh(f"echo hello | atlasforge run - --base-url {s.url} --model m", cwd=W, log="t11_stdin")
sh(f"printf 'a\\nb\\n\\n' | atlasforge run - --base-url {s.url} --model m", cwd=W, log="t11_stdin2")
sent = [x["data"]["messages"][-1]["content"] for x in s.seen]
c("11.01", "`echo hello | atlasforge run -` sends 'hello', not 'hello' plus a newline (was a minor bug in 0.1.0a2); inner newlines are kept",
  sent == ["hello", "a\nb\n"], f"sent={sent}")
s.stop()

# ---- 401 abort names the API key
s = StubServer().start()
s.fail_always = 401
write_jsonl(W / "d.jsonl", [{"id": f"e{i}", "input": f"q{i}", "reference": "x"} for i in range(10)])
r = sh(f"atlasforge eval d.jsonl --out o --base-url {s.url} --model m --retries 0 --max-consecutive-failures 3", cwd=W, log="t11_401")
c("11.02", "When every request is refused with HTTP 401, the abort hint tells the user to set ATLASFORGE_API_KEY (generic advice before)",
  r.rc == 2 and "ATLASFORGE_API_KEY" in r.err and "401" in r.err and "Traceback" not in r.both, r.err.strip()[:230], "t11_401")
s.stop()

# ---- doctor on Apple Silicon
r = sh("atlasforge doctor --json", cwd=W, log="t11_doctor")
rows = {x["name"]: x for x in json.loads(r.out)}
import platform
if platform.system() == "Darwin" and platform.machine() == "arm64":
    g = rows["gpu"]
    c("11.03", "doctor recognises Apple Silicon: still warns about NVIDIA, but says the speech models can use the Apple GPU and the LLM needs a quantised server",
      g["status"] == "warn" and "Apple Silicon" in g["detail"] and "speech" in (g["hint"] or "") and "quantised" in (g["hint"] or ""), f"{g['detail']} | {g['hint'][:140]}", "t11_doctor")
else:
    c.na("11.03", "doctor on Apple Silicon", "not an Apple Silicon machine")

# ---- install on Python 3.14 (the scipy pin used to make this fail)
log314 = ROOT / "logs" / "rel314__install.log"
if log314.exists():
    t = log314.read_text(errors="replace")
    c("11.04", "`pip install \"brainers-atlasforge[finetune]\"` on macOS with Python 3.14 now succeeds (the scipy<1.15 pin used to force a failing source build)",
      "Successfully installed" in t and "brainers-atlasforge-0.1.0a3" in t and "metadata-generation-failed" not in t, "installed 0.1.0a3 with peft/trl/datasets on Python 3.14")
else:
    c.na("11.04", "Python 3.14 install", "install log missing")

# ---- multilingual speech run names both models (real models)
if sh([PY, "-c", "import torch, transformers"]).rc == 0:
    shutil.copytree(DATA / "audio", W / "audio")
    rows_ = [json.loads(l) for l in (DATA / "asr_ha.jsonl").read_text(encoding="utf-8").splitlines()[:2]] + \
            [json.loads(l) for l in (DATA / "asr_yo.jsonl").read_text(encoding="utf-8").splitlines()[:2]]
    write_jsonl(W / "mixed.jsonl", rows_)
    r = sh("atlasforge eval mixed.jsonl --task asr --out mixed_run --backend local", cwd=W, log="t11_mixed", timeout=1500)
    m = json.loads((W / "mixed_run/run.json").read_text()) if r.rc == 0 else {}
    md = (W / "mixed_run/report.md").read_text(encoding="utf-8") if r.rc == 0 else ""
    c("11.05", "A mixed Hausa and Yoruba speech run records BOTH speech models (and not the LLM) in run.json and the report header",
      r.rc == 0 and m.get("model") == "NCAIR1/Hausa-ASR, NCAIR1/Yoruba-ASR" and "NCAIR1/Yoruba-ASR" in md and "N-ATLaS" not in md.split("## Metrics")[0],
      f"model={m.get('model')!r}", "t11_mixed")
