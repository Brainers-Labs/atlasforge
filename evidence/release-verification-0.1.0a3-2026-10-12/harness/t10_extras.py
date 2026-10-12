"""Remaining features: HTTP speech endpoint, hostile output in HTML, partial runs, config discovery, doctor exit code."""
import json
import os
import shutil
import signal
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from common import *

c = Checker("10 Extra features and edge cases")
W = WORK / f"extras_{LABEL}"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)

# ---------------------------------------------------------------- speech over an OpenAI-style /audio/transcriptions endpoint
calls = []


class ASRHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): return

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        calls.append({"path": self.path, "len": n, "ctype": self.headers.get("content-type", ""), "has_riff": b"RIFF" in body[:2000],
                      "lang": b'name="language"\r\n\r\nha' in body, "model": b'name="model"\r\n\r\nwhisper-test' in body})
        out = json.dumps({"text": f"stub transcript {len(calls)}"}).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(out))); self.end_headers(); self.wfile.write(out)


srv = ThreadingHTTPServer(("127.0.0.1", 0), ASRHandler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
shutil.copy(DATA / "audio/ha_00.wav", W / "short.wav")
shutil.copy(DATA / "audio/ha_long.wav", W / "long.wav")
r = sh(f"atlasforge transcribe short.wav --lang ha --base-url {url} --model whisper-test", cwd=W, log="t10_http_asr")
c("10.01", "transcribe over HTTP (--backend openai): posts the audio as WAV with language and model to /audio/transcriptions and prints the text",
  r.rc == 0 and r.out.strip() == "stub transcript 1" and calls[0]["path"].endswith("/audio/transcriptions") and calls[0]["has_riff"] and calls[0]["lang"] and calls[0]["model"],
  f"out={r.out.strip()!r} request={calls[0] if calls else None}", "t10_http_asr")
calls.clear()
r = sh(f"atlasforge transcribe long.wav --lang ha --base-url {url} --model whisper-test --json", cwd=W, log="t10_http_asr_long")
rec = json.loads(r.out.strip().splitlines()[-1]) if r.rc == 0 else {}
c("10.02", "A 79 s recording over HTTP is split client-side into 3 windows (3 requests, each a WAV under 30 s) and merged",
  r.rc == 0 and len(calls) == 3 and len(rec.get("chunks", [])) == 3 and all(x["len"] < 1_100_000 for x in calls), f"requests={len(calls)} sizes={[x['len'] for x in calls]} chunks={len(rec.get('chunks', []))}", "t10_http_asr_long")
srv.shutdown()

# ---------------------------------------------------------------- hostile model output cannot break out of the HTML reports
evil = '<script>alert(1)</script></td><img src=x onerror=alert(2)> & "quotes"'
s = StubServer(answer=lambda p: evil).start()
write_jsonl(W / "d.jsonl", [{"id": f"e{i}", "input": f"q{i}", "reference": "ok", "meta": {"domain": "<b>x</b>"}} for i in range(5)])
r = sh(f"atlasforge eval d.jsonl --out evil --base-url {s.url} --model '<img src=x onerror=alert(3)>' --retries 0", cwd=W, log="t10_evil")
r2 = sh(f"atlasforge eval d.jsonl --out evil2 --base-url {s.url} --model 'safe' --retries 0", cwd=W)
sh(f"atlasforge compare d.jsonl --base evil --candidate evil2 --out evil_cmp --slice domain --task generation", cwd=W, log="t10_evil_cmp")
bad = []
for p in (W / "evil/report.html", W / "evil_cmp/comparison.html"):
    h = p.read_text(encoding="utf-8")
    if "<script>alert" in h or "<img src=x" in h or "onerror=alert" in h.replace("&quot;", '"') and "&lt;img" not in h:
        bad.append(p.name)
c("10.03", "Hostile text in a model name, a model answer, or a slice value is escaped in report.html and comparison.html (no injected tags)",
  not bad and r.rc == 0, f"unescaped in: {bad}; escaped marker present={'&lt;script&gt;' in (W / 'evil/report.html').read_text() or '&lt;img' in (W / 'evil/report.html').read_text()}", "t10_evil")
s.stop()

# ---------------------------------------------------------------- partial runs: killed mid-way, then scored
s = StubServer(answer=lambda p: "ans" + p[1:]).start()
s.delay = 0.25
write_jsonl(W / "d40.jsonl", [{"id": f"e{i}", "input": f"q{i}", "reference": f"ans{i}"} for i in range(40)])
p = subprocess.Popen(f"{BIN}/atlasforge eval d40.jsonl --out partial --base-url {s.url} --model m --retries 0", shell=True, cwd=W,
                     env=os.environ | {"PATH": f"{BIN}:{os.environ['PATH']}"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(3.5)
os.killpg(p.pid, signal.SIGKILL); p.wait()
done = len(read_jsonl(W / "partial/results.jsonl"))
r = sh("atlasforge report partial --dataset d40.jsonl -m exact_match", cwd=W, log="t10_partial_report")
rj = json.loads((W / "partial/report.json").read_text())
mean = next(m for m in rj["metrics"] if m["key"] == "exact_match@tone_aware")["mean"]
c("10.04", "Scoring a run that was killed halfway counts the unfinished examples as MISSING and wrong (not silently ignored): score = finished/40",
  r.rc == 0 and rj["n_missing"] == 40 - done and abs(mean - done / 40) < 1e-9 and "missing" in r.out, f"finished={done} n_missing={rj['n_missing']} exact_match={mean:.3f} (expected {done / 40:.3f})", "t10_partial_report")
s.stop()

# ---------------------------------------------------------------- config discovery from a subdirectory
s = StubServer().start()
(W / "root").mkdir()
(W / "root/atlasforge.toml").write_text(f'[atlasforge]\nbase_url = "{s.url}"\nmodel = "from-parent-file"\n')
(W / "root/a/b/c").mkdir(parents=True)
r = sh("atlasforge run hello", cwd=W / "root/a/b/c", log="t10_cfg_nested")
c("10.05", "atlasforge.toml is found from any subdirectory below it (nearest file at or above the working directory)",
  r.rc == 0 and len(s.seen) == 1 and s.seen[0]["data"]["model"] == "from-parent-file", f"server hits={len(s.seen)} model={s.seen[0]['data']['model'] if s.seen else None}", "t10_cfg_nested")
r = sh("atlasforge run hello", cwd=W, log="t10_cfg_outside")
c("10.06", "A config file in a SIBLING/other directory is not picked up (nothing read from elsewhere or from the home directory)", r.rc == 2 and "server URL" in r.err, r.err.strip()[:120], "t10_cfg_outside")
s.stop()

# ---------------------------------------------------------------- doctor exit codes
(W / "badcfg").mkdir()
(W / "badcfg/atlasforge.toml").write_text("[atlasforge]\nthis is = not valid toml [[[\n")
r = sh("atlasforge doctor", cwd=W / "badcfg", log="t10_doctor_bad")
c("10.07", "doctor exits 1 when a check FAILS (a broken atlasforge.toml), and says which line/why, rather than crashing", r.rc == 1 and "config" in r.out and "Traceback" not in r.both, " ".join([l for l in r.out.splitlines() if "config" in l][:1]).__str__()[:200], "t10_doctor_bad")
r = sh("atlasforge -V")
c("10.08", "-V is the short form of --version", r.rc == 0 and "atlasforge" in r.out, r.out.strip())

# ---------------------------------------------------------------- demo twice, dataset validate on stdin-less misuse
r1 = sh("atlasforge demo", cwd=W / "root", log="t10_demo1")
r2 = sh("atlasforge demo", cwd=W / "root", log="t10_demo2")
c("10.09", "Running `demo` twice in the same place is safe (no traceback, data still valid afterwards)",
  r1.rc == 0 and r2.rc in (0, 2) and "Traceback" not in r2.both and sh("atlasforge dataset validate atlasforge-demo/toy_qa.jsonl", cwd=W / "root").rc == 0,
  f"second run rc={r2.rc}: {(r2.out + r2.err).strip()[:120]}", "t10_demo2")
r = sh("atlasforge dataset validate /does/not/exist.jsonl")
c("10.10", "A missing dataset path is a clean error, not a traceback", r.rc == 2 and "Traceback" not in r.both, r.err.strip()[:140])
