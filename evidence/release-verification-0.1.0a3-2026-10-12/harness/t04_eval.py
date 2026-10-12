"""Evaluation engine robustness, black-box, against an independent stub server."""
import json
import os
import shutil
import signal
import subprocess
import time

from common import *

c = Checker("4 Evaluation engine robustness and secret handling")
W = WORK / "eval"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)

N = 40
rows = [{"id": f"e{i}", "input": f"q{i}", "reference": f"ans{i}", "meta": {"domain": "a" if i < 20 else "b"}} for i in range(N)]
write_jsonl(W / "d.jsonl", rows)
ANS = {f"q{i}": f"ans{i}" for i in range(N)}
KEY = "sk-" + "test-" + "SECRET-" + "KEY-" + "9f8e7d6c5b4a"  # a fake, built at runtime so no secret-shaped literal is committed


def stub(**kw):
    s = StubServer(answer=lambda p: ANS.get(p, "?"))
    for k, v in kw.items():
        setattr(s, k, v)
    return s.start()


def ev(out, s, extra="", env=None, log=None, model="m", timeout=300):
    return sh(f"atlasforge eval d.jsonl --out {out} --base-url {s.url} --model {model} --retries 0 {extra}", cwd=W, env=env, log=log or f"t04_{out}", timeout=timeout)


# --------------------------------------------------------------- basic run + secrets
s = stub(require_key=KEY)
r = ev("basic", s, "-m exact_match", env={"ATLASFORGE_API_KEY": KEY})
res = read_jsonl(W / "basic/results.jsonl")
c("4.01", "eval runs a dataset through an OpenAI-compatible server and writes run.json, results.jsonl, report.*",
  r.rc == 0 and len(res) == N and all(x["error"] is None for x in res) and all((W / "basic" / f).exists() for f in ("run.json", "report.md", "report.json", "report.html")),
  f"rc={r.rc} records={len(res)} request_id={res[0]['request_id']}", "t04_basic")
c("4.02", "The API key is sent as a Bearer token (server accepted it) ...", all(x["auth"] == f"Bearer {KEY}" for x in s.seen), f"{len(s.seen)} requests authenticated")
blob = r.both + "".join(p.read_text(errors="replace") for p in (W / "basic").iterdir() if p.is_file())
c("4.03", "... and the key never appears in stdout, stderr, run.json, results, or any report", KEY not in blob and KEY[:12] not in blob, "searched all outputs")
manifest = (W / "basic/run.json").read_text()
c("4.04", "run.json records model and settings but no URL, key or token", "127.0.0.1" not in manifest and KEY not in manifest and "http" not in manifest, "manifest clean")
hp = sh("atlasforge eval --help", log="t04_evalhelp")
c("4.05", "There is no --api-key flag (so a key cannot land in shell history or the process list)", "--api-key" not in hp.out, "flag absent")
r = ev("nokey", s, env={})
c("4.06", "A server demanding a key, given none: reported as HTTP 401 failures and the breaker stops the run (exit 2), never a traceback",
  r.rc == 2 and "Traceback" not in r.both and "401" in r.err and "401" in (W / "nokey/results.jsonl").read_text(), r.err.strip()[:140], "t04_nokey")
s.stop()

# --------------------------------------------------------------- resume after a hard kill
s = stub(delay=0.2)
cmd = f"{BIN}/atlasforge eval d.jsonl --out resume --base-url {s.url} --model m --retries 0 -m exact_match"
p = subprocess.Popen(cmd, shell=True, cwd=W, env=os.environ | {"PATH": f"{BIN}:{os.environ['PATH']}"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(3.5)
os.killpg(p.pid, signal.SIGKILL)  # no chance to clean up
p.wait()
done_before = len(read_jsonl(W / "resume/results.jsonl")) if (W / "resume/results.jsonl").exists() else 0
seen_before = len(s.seen)
s.delay = 0.0
r = ev("resume", s, "-m exact_match", log="t04_resume")
final = read_jsonl(W / "resume/results.jsonl")
ids = [x["id"] for x in final]
reqs = [x["data"]["messages"][-1]["content"] for x in s.seen]
reran = sum(1 for i in range(done_before) if reqs[seen_before:].count(f"q{i}") > 0)
c("4.07", "After kill -9 mid-run, re-running the same command resumes: finished examples are NOT run again, and all end up complete",
  0 < done_before < N and r.rc == 0 and len(set(ids)) == N and f"Resumed: {done_before}" in r.out and reran == 0,
  f"finished before kill={done_before}; resumed message={'Resumed' in r.out}; re-requested finished={reran}; final unique ids={len(set(ids))}/{N}", "t04_resume")
s.stop()

# --------------------------------------------------------------- manifest guard
s = stub()
ev("guard", s, log="t04_guard1")
r = ev("guard", s, model="other-model", log="t04_guard2")
c("4.08", "Re-using a run directory with a different model is refused (never mixes results)", r.rc == 2 and "changed: model" in r.err, r.err.strip()[:160], "t04_guard2")
r = sh(f"atlasforge eval d.jsonl --out guard --base-url {s.url} --model m --retries 0 --temperature 0.9", cwd=W, log="t04_guard3")
c("4.09", "... or with different generation settings", r.rc == 2 and "gen_params" in r.err, r.err.strip()[:160], "t04_guard3")
shutil.copy(W / "d.jsonl", W / "d2.jsonl")
with open(W / "d2.jsonl", "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"id": "new", "input": "q0", "reference": "ans0"}) + "\n")
r = sh(f"atlasforge eval d2.jsonl --out guard --base-url {s.url} --model m --retries 0", cwd=W, log="t04_guard4")
c("4.10", "... or with an edited dataset", r.rc == 2 and "dataset_sha256" in r.err, r.err.strip()[:160], "t04_guard4")
s.stop()

# --------------------------------------------------------------- circuit breaker
s = stub(fail_always=500)
r = ev("breaker", s, "--max-consecutive-failures 5")
n_req = len(s.seen)
c("4.11", "A dead server trips the circuit breaker: stops after 5 failures in a row (exit 2) instead of hammering all 40 examples",
  r.rc == 2 and "Stopped after 5 failures in a row" in r.err and n_req <= 8, f"rc={r.rc} requests sent={n_req} (of {N}); msg={r.err.strip()[:110]}", "t04_breaker")
s.fail_always = 0
r = ev("breaker", s, "--max-consecutive-failures 5", log="t04_breaker_resume")
fin = read_jsonl(W / "breaker/results.jsonl")
latest = {x["id"]: x for x in fin}
c("4.12", "After fixing the server, the same command resumes and completes everything that failed",
  r.rc == 0 and len(latest) == N and all(x["error"] is None for x in latest.values()), f"rc={r.rc} ok={sum(x['error'] is None for x in latest.values())}/{N}", "t04_breaker_resume")
s.stop()

s = stub(fail_always=500)
r = ev("allfail", s, "--max-consecutive-failures 0 -m exact_match")
rj = json.loads((W / "allfail/report.json").read_text())
c("4.13", "With the breaker off and everything failing: all 40 are tried, exit code 1, and the report scores them as wrong (0%), not as missing",
  r.rc == 1 and len(s.seen) == N and rj["n_failed"] == N and rj["metrics"][0]["mean"] == 0.0, f"rc={r.rc} requests={len(s.seen)} n_failed={rj['n_failed']} mean={rj['metrics'][0]['mean']}", "t04_allfail")
s.stop()

# --------------------------------------------------------------- failures count as wrong; retry-errors
s = stub(fail_first=3)
r = ev("partial", s, "-m exact_match")
rj = json.loads((W / "partial/report.json").read_text())
c("4.14", "3 transient failures: exit 0 with a warning; the report counts 3 failed and the score drops by exactly those (37/40)",
  r.rc == 0 and rj["n_failed"] == 3 and abs(rj["metrics"][0]["mean"] - 37 / 40) < 1e-9 and "failed" in r.err, f"n_failed={rj['n_failed']} mean={rj['metrics'][0]['mean']:.4f}; stderr={r.err.strip()[:80]}", "t04_partial")
r = ev("partial", s, "--no-retry-errors -m exact_match", log="t04_partial_noretry")
rj2 = json.loads((W / "partial/report.json").read_text())
c("4.15", "--no-retry-errors leaves failed examples as failures on resume", rj2["n_failed"] == 3, f"n_failed={rj2['n_failed']}", "t04_partial_noretry")
r = ev("partial", s, "-m exact_match", log="t04_partial_retry")
rj3 = json.loads((W / "partial/report.json").read_text())
c("4.16", "By default a resume retries failed examples and fixes them (40/40)", rj3["n_failed"] == 0 and abs(rj3["metrics"][0]["mean"] - 1.0) < 1e-9, f"n_failed={rj3['n_failed']} mean={rj3['metrics'][0]['mean']}", "t04_partial_retry")
s.stop()

# --------------------------------------------------------------- retries and timeouts
s = stub(fail_first=2)
r = sh(f"atlasforge eval d.jsonl --out retried --base-url {s.url} --model m --retries 2 -m exact_match", cwd=W, log="t04_retried")
rj = json.loads((W / "retried/report.json").read_text())
c("4.17", "--retries masks transient 503s: every example succeeds and the server saw the extra attempts",
  rj["n_failed"] == 0 and len(s.seen) == N + 2, f"n_failed={rj['n_failed']} requests={len(s.seen)} (expected {N + 2})", "t04_retried")
s.stop()

s = stub(delay=2.5)
sm = rows[:3]
write_jsonl(W / "small.jsonl", sm)
t0 = time.time()
r = sh(f"atlasforge eval small.jsonl --out timeout --base-url {s.url} --model m --retries 2 --timeout 1 --max-consecutive-failures 0", cwd=W, log="t04_timeout")
msgs = {x["error"] for x in read_jsonl(W / "timeout/results.jsonl")}
c("4.18", "A timeout is reported ('No response within 1s') and is NOT retried (3 examples = 3 requests)",
  r.rc == 1 and len(s.seen) == 3 and any("No response within 1s" in (m or "") for m in msgs), f"requests={len(s.seen)} error={list(msgs)[:1]}", "t04_timeout")
s.stop()

# --------------------------------------------------------------- concurrency
s = stub()
ev("seq", s, "-m exact_match -m chrf", log="t04_seq")
r = ev("conc", s, "-m exact_match -m chrf --concurrency 8", log="t04_conc")
a = json.loads((W / "seq/report.json").read_text())["per_example"]
b = json.loads((W / "conc/report.json").read_text())["per_example"]
c("4.19", "--concurrency 8 gives exactly the same scores as sequential", r.rc == 0 and a == b, f"identical per-example scores for {len(a)} examples", "t04_conc")
s.stop()

# --------------------------------------------------------------- errors never echo the server's response body
s = stub(fail_always=500, error_body="LEAK-CANARY echo of prompt q17 and key")
r = ev("noecho", s, "--max-consecutive-failures 0")
blob = r.both + "".join(p.read_text(errors="replace") for p in (W / "noecho").iterdir() if p.is_file())
c("4.20", "A server error body (which may echo the prompt) never reaches output, results or reports", "LEAK-CANARY" not in blob, "canary absent everywhere", "t04_noecho")
s.stop()

# --------------------------------------------------------------- plain http guard
r = sh("atlasforge run hi --base-url http://192.0.2.10:8000/v1 --timeout 2 --retries 0", cwd=W, log="t04_http_guard")
c("4.21", "Plain http:// to a non-local host is refused before any request is made", r.rc == 2 and "Refusing plain http" in r.err, r.err.strip()[:160], "t04_http_guard")
r = sh("atlasforge run hi --base-url http://192.0.2.10:8000/v1 --timeout 2 --retries 0 --allow-insecure-http", cwd=W, log="t04_http_guard2")
c("4.22", "--allow-insecure-http lets the user opt in knowingly (fails later on connection, not on policy)", "Refusing" not in r.err, r.err.strip()[:120], "t04_http_guard2")

# --------------------------------------------------------------- run command
s = stub()
r = sh(f"atlasforge run q3 --base-url {s.url} --model m", cwd=W, log="t04_run")
c("4.23", "run prints the answer", r.rc == 0 and r.out.strip() == "ans3", repr(r.out.strip()), "t04_run")
r = sh(f"atlasforge run q4 --base-url {s.url} --model m --json --system 'Be brief.' --temperature 0 --max-new-tokens 77 --seed 5", cwd=W, log="t04_run_json")
j = json.loads(r.out)
body = s.seen[-1]["data"]
c("4.24", "run --json returns text, model, latency, finish_reason, request_id, usage",
  {"text", "model", "latency_ms", "finish_reason", "request_id", "usage"} <= set(j) and j["text"] == "ans4", f"keys={sorted(j)}", "t04_run_json")
c("4.25", "run forwards --system, --temperature, --max-new-tokens, --seed and the model-card repetition penalty (1.12) to the server",
  body["messages"][0] == {"role": "system", "content": "Be brief."} and body["temperature"] == 0 and body["max_tokens"] == 77 and body.get("seed") == 5 and body.get("repetition_penalty") == 1.12,
  json.dumps({k: body[k] for k in body if k != "messages"}))
r = sh(f"atlasforge run q5 --base-url {s.url} --model m --no-send-repetition-penalty", cwd=W)
c("4.26", "--no-send-repetition-penalty omits the field for servers that reject it", "repetition_penalty" not in s.seen[-1]["data"], "field absent")
r = sh(f"printf q6 | atlasforge run - --base-url {s.url} --model m", cwd=W, log="t04_run_stdin")
c("4.27", "run - reads the prompt from standard input", r.out.strip() == "ans6", repr(r.out.strip()), "t04_run_stdin")
r = sh("atlasforge run hi", cwd=W, log="t04_run_nourl")
c("4.28", "Without a server URL the error says exactly what to do", r.rc == 2 and "--base-url" in r.err and "Traceback" not in r.both, r.err.strip()[:160], "t04_run_nourl")
s.stop()

# --------------------------------------------------------------- config file precedence and custom metric
sa, sb, sc = stub(), stub(), stub()
(W / "proj").mkdir()
(W / "proj/atlasforge.toml").write_text(f'[atlasforge]\nbase_url = "{sa.url}"\nmodel = "from-file"\n')
sh("atlasforge run q1", cwd=W / "proj", log="t04_cfg_file")
a_hit = len(sa.seen)
sh("atlasforge run q1", cwd=W / "proj", env={"ATLASFORGE_BASE_URL": sb.url}, log="t04_cfg_env")
b_hit = len(sb.seen)
sh(f"atlasforge run q1 --base-url {sc.url} --model from-flag", cwd=W / "proj", env={"ATLASFORGE_BASE_URL": sb.url}, log="t04_cfg_flag")
c("4.29", "Precedence works as documented: flag beats environment beats atlasforge.toml",
  a_hit == 1 and b_hit == 1 and len(sc.seen) == 1 and len(sb.seen) == 1 and sa.seen[0]["data"]["model"] == "from-file" and sc.seen[0]["data"]["model"] == "from-flag",
  f"file-server hits={a_hit}, env-server hits={len(sb.seen)}, flag-server hits={len(sc.seen)}")
(W / "proj/atlasforge.toml").write_text('[atlasforge]\nbase-url = "x"\n')
r = sh("atlasforge run q1", cwd=W / "proj", log="t04_cfg_bad")
c("4.30", "A misspelled setting in atlasforge.toml is an error naming the valid keys (never a silent no-op)", r.rc == 2 and "unknown" in r.err.lower() and "base_url" in r.err, r.err.strip()[:200], "t04_cfg_bad")
r = sh("atlasforge doctor", cwd=W / "proj", log="t04_cfg_doctor")
c("4.31", "doctor shows the config file it found", "config" in r.out and "atlasforge.toml" in r.out, [l for l in r.out.splitlines() if "config" in l][:1].__str__()[:160], "t04_cfg_doctor")
for x in (sa, sb, sc):
    x.stop()

(W / "mymetric.py").write_text('def long_enough(prediction, reference, example):\n    return float(len(prediction) >= 4)\n')
s = stub()
r = sh(f"atlasforge eval d.jsonl --out custom --base-url {s.url} --model m --retries 0 -m exact_match -m mymetric:long_enough", cwd=W, env={"PYTHONPATH": str(W)}, log="t04_custom")
if r.rc == 0:
    rj = json.loads((W / "custom/report.json").read_text())
    m = next((x for x in rj["metrics"] if x["name"] == "long_enough"), None)
    c("4.32", "A custom metric named module:function runs through eval and is reported under both views", m is not None and abs(m["mean"] - 1.0) < 1e-9, f"mean={m and m['mean']}; views={sorted({x['view'] for x in rj['metrics'] if x['name']=='long_enough'})}", "t04_custom")
else:
    c("4.32", "A custom metric named module:function runs through eval", False, (r.err or r.out).strip()[:200], "t04_custom")
s.stop()
