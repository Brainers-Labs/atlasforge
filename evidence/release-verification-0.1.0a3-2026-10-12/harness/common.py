"""Black-box test helpers. Everything here drives the installed `atlasforge` as a user would.

Nothing imports atlasforge's own test code. The stub server below is an independent
re-implementation of just enough of the OpenAI protocol to test failure handling.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
DATA = ROOT / "data"
WORK = ROOT / "work"
RESULTS = ROOT / "results.jsonl"
VENV = Path(os.environ.get("AF_VENV", ROOT / "venv-pypi"))
LABEL = os.environ.get("AF_LABEL", VENV.name)
BIN = VENV / "bin"
PY = str(BIN / "python")
for d in (LOGS, DATA, WORK):
    d.mkdir(exist_ok=True)

PASS, FAIL, NOT_TESTABLE = "PASS", "FAIL", "NOT TESTABLE"


@dataclass
class Run:
    rc: int
    out: str
    err: str
    secs: float
    cmd: str

    @property
    def both(self) -> str:
        return self.out + self.err


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ATLASFORGE_", "HF_TOKEN", "HUGGING_FACE"))}
    env["PATH"] = f"{BIN}:{env['PATH']}"
    env["NO_COLOR"] = "1"
    env["COLUMNS"] = "200"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def sh(cmd: str | list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None,
       timeout: int = 600, inp: str | None = None, log: str | None = None) -> Run:
    """Run a command with the venv first on PATH. Logs the full transcript."""
    shell = isinstance(cmd, str)
    start = time.perf_counter()
    try:
        p = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout,
                           env=_clean_env(env), cwd=cwd or WORK, input=inp, encoding="utf-8", errors="replace")
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        rc, out, err = 124, (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), "TIMEOUT"
    secs = time.perf_counter() - start
    text = cmd if shell else " ".join(cmd)
    if log:
        (LOGS / f"{LABEL}__{log}.log").write_text(
            f"$ {text}\n[exit {rc}] [{secs:.1f}s]\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n", encoding="utf-8")
    return Run(rc, out, err, secs, text)


def record(group: str, cid: str, claim: str, status: str, evidence: str, log: str = "") -> None:
    row = {"label": LABEL, "group": group, "id": cid, "claim": claim, "status": status,
           "evidence": evidence.strip().replace("\n", " | ")[:600], "log": log}
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    mark = {"PASS": "[ ok ]", "FAIL": "[FAIL]", "NOT TESTABLE": "[ n/a]"}[status]
    print(f"{mark} {cid}: {claim}\n         {row['evidence'][:200]}", flush=True)


def clear_group(prefix: str) -> None:
    """Drop earlier rows of this label whose id starts with prefix, so re-runs do not duplicate."""
    if not RESULTS.exists():
        return
    keep = [l for l in RESULTS.read_text(encoding="utf-8").splitlines()
            if not (json.loads(l)["label"] == LABEL and json.loads(l)["id"].split(".")[0] == prefix)]
    RESULTS.write_text("".join(l + "\n" for l in keep), encoding="utf-8")


class Checker:
    def __init__(self, group: str) -> None:
        self.group = group
        clear_group(group.split()[0])

    def __call__(self, cid: str, claim: str, ok: bool, evidence: str, log: str = "") -> bool:
        record(self.group, cid, claim, PASS if ok else FAIL, evidence, log)
        return ok

    def na(self, cid: str, claim: str, reason: str) -> None:
        record(self.group, cid, claim, NOT_TESTABLE, reason)


# ---------------------------------------------------------------- stub OpenAI server


class StubServer(ThreadingHTTPServer):
    """Independent minimal OpenAI-compatible server with controllable misbehaviour."""

    daemon_threads = True
    request_queue_size = 128

    def __init__(self, answer: Callable[[str], str] | None = None) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.answer = answer or (lambda prompt: "stub:" + prompt)
        self.seen: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.fail_always = 0  # status to return for every request (0 = off)
        self.fail_first = 0  # fail the first N requests with 503 then recover
        self.require_key: str | None = None
        self.error_body = ""  # body sent with error statuses
        self.delay = 0.0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def start(self) -> StubServer:
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    server: StubServer

    def log_message(self, *a: object) -> None:  # silence
        return

    def _send(self, status: int, payload: dict[str, Any] | str) -> None:
        body = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        s = self.server
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        try:
            data = json.loads(raw)
        except ValueError:
            data = {}
        prompt = (data.get("messages") or [{}])[-1].get("content", "")
        with s.lock:
            s.seen.append({"auth": self.headers.get("authorization"), "path": self.path, "data": data})
            n = len(s.seen)
        if s.delay:
            time.sleep(s.delay)
        if s.require_key and self.headers.get("authorization") != f"Bearer {s.require_key}":
            self._send(401, s.error_body or '{"error":"bad key"}')
            return
        if s.fail_always:
            self._send(s.fail_always, s.error_body or '{"error":"boom"}')
            return
        if n <= s.fail_first:
            self._send(503, s.error_body or '{"error":"warming up"}')
            return
        self._send(200, {"id": "stub-1", "choices": [{"message": {"content": s.answer(prompt)}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
