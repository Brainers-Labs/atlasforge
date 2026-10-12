"""Build REPORT.md from results.jsonl. Every status in the tables is a recorded result, not hand-written."""
import json
import platform
import subprocess
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

PYPI = ("rel", "rel-asr", "rel314")          # what a public user gets from `pip install brainers-atlasforge`
HEAD = ()               # the repo at HEAD (unreleased)
EXTRA = ()                              # PyPI [asr] exactly as installed, before the scipy workaround


def build_of(label):
    return "pypi" if label in PYPI else "head" if label in HEAD else "pypi-asr-asis" if label in EXTRA else label


by_id = OrderedDict()
for r in rows:
    key = r["id"]
    e = by_id.setdefault(key, {"group": r["group"], "claim": r["claim"], "res": {}})
    e["res"][build_of(r["label"])] = (r["status"], r["evidence"], r["log"], r["label"])


def sortkey(k):
    a, _, b = k.partition(".")
    num = "".join(ch for ch in b if ch.isdigit()) or "0"
    return (int(a) if a.isdigit() else 99, int(num), b)


prev_rows = [json.loads(l) for l in (ROOT / "archive_0.1.0a2" / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
prev = {}
for r in prev_rows:
    if r["label"] in ("venv-pypi", "pypi-asr", "pypi-asr-scipyfix"):
        old = prev.get(r["id"])
        # worst result wins, so "5.01 broke as installed" is not hidden by the later workaround run
        rank = {"FAIL": 2, "NOT TESTABLE": 1, "PASS": 0}
        if old is None or rank[r["status"]] > rank[old["status"]]:
            prev[r["id"]] = r
ids = sorted(by_id, key=sortkey)
mark = {"PASS": "PASS", "FAIL": "**FAIL**", "NOT TESTABLE": "n/a"}


def prevcell(i):
    r = prev.get(i)
    return mark[r["status"]] if r else "-"


def cell(e, b):
    r = e["res"].get(b)
    return mark[r[0]] if r else "-"


def counts(build):
    c = Counter()
    for e in by_id.values():
        r = e["res"].get(build)
        if r:
            c[r[0]] += 1
    return c


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


env = {
    "date": datetime.now().strftime("%d %B %Y, %H:%M WAT"),
    "mac": sh("sw_vers -productVersion") + " / " + platform.machine() + " / " + sh("sysctl -n hw.memsize | awk '{print $1/1073741824\" GB\"}'"),
    "pythons": "3.10.12 (core and speech), 3.14.7 (fine-tune install and adapter checks)",
    "pypi": sh(f"{ROOT}/venv-pypi/bin/pip list 2>/dev/null | grep -i '^brainers'"),
    "head": sh(f"git -C {ROOT.parents[2]} log --oneline -1"),
    "asr_pkgs": sh(f"{ROOT}/venv-pypi-asr/bin/pip list 2>/dev/null | grep -E '^(torch|transformers|scipy|librosa) ' | tr -s ' ' '=' | tr '\\n' ' '"),
    "ollama": sh("ollama --version 2>&1 | tail -1"),
    "ffmpeg": sh("ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3"),
}

out = []
w = out.append
w("# AtlasForge 0.1.0a3 release verification report\n")
w(f"*Generated {env['date']} by the same black-box harness used for 0.1.0a2. This run installs **what PyPI serves now** in fresh virtual environments with no workarounds. Every status is a recorded result; the evidence is in `logs/` and `results.jsonl`.*\n")
w("## Verdict\n")
cp = counts("pypi")
w("| Build | Checks run | PASS | FAIL | Not testable |")
w("|---|---|---|---|---|")
w(f"| **PyPI `brainers-atlasforge` 0.1.0a3** | {sum(cp.values())} | {cp['PASS']} | {cp['FAIL']} | {cp['NOT TESTABLE']} |")
w("")
FINDINGS = (ROOT / "FINDINGS_REL.md")
if FINDINGS.exists():
    w(FINDINGS.read_text(encoding="utf-8").strip() + "\n")

w("## Test environment\n")
w(f"- Machine: MacBook, macOS {env['mac']}, Apple MPS or CPU only (no NVIDIA GPU)")
w(f"- Python: {env['pythons']}")
pkg_core = sh(f"{ROOT}/venv-rel/bin/pip list 2>/dev/null | grep -i '^brainers'")
pkg_asr = sh(f"{ROOT}/venv-rel-asr/bin/pip list 2>/dev/null | grep -E '^(torch|transformers|scipy|librosa) ' | tr -s ' ' '=' | tr '\\n' ' '")
w(f"- Published package under test: `{pkg_core}` installed from PyPI with a plain `pip install` in a fresh environment; speech stack: {pkg_asr}")
w(f"- Model server: Ollama {env['ollama']} serving the official `NCAIR1/N-ATLaS` weights imported as int4 (chat template restored); ffmpeg: {env['ffmpeg']}")
w("- Speech models: the four official `NCAIR1` ASR repositories, run locally with the Hugging Face cache and a real token")
w("- Test audio: real clips from Google FLEURS (CC BY 4.0), 10 per language")
w("- All checks drive the installed `atlasforge` as a user would (command line, and the documented Python API). The stub server in `common.py` is an independent implementation, not AtlasForge's own test code.\n")

# real-model numbers
summ = ROOT / "asr_summary_rel-asr.json"
if summ.exists():
    s = json.loads(summ.read_text())
    w("## Formal speech results (real models, real audio)\n")
    w("Pooled word error rate (WER) and character error rate (CER) through `atlasforge eval --task asr`, 10 FLEURS test clips per language. **Small samples: indicative only, not a benchmark.** Lower is better.\n")
    w("| Language | Model | WER (tone-aware) | WER (tone-insensitive) | CER | Mean latency per clip | Failed |")
    w("|---|---|---|---|---|---|---|")
    names = {"ha": "NCAIR1/Hausa-ASR", "yo": "NCAIR1/Yoruba-ASR", "ig": "NCAIR1/Igbo-ASR", "en": "NCAIR1/NigerianAccentedEnglish"}
    for lang in ("ha", "yo", "ig", "en"):
        if lang in s:
            a, b, c_, lat, fl = s[lang]
            note = " (US-accent audio, not Nigerian-accented)" if lang == "en" else ""
            w(f"| {lang} | {names[lang]}{note} | {a:.3f} | {b:.3f} | {c_:.3f} | {lat:.0f} ms | {fl} |")
    w("")

# per-group tables
w("## Every check, by group\n")
w("`0.1.0a2` = the result of the same check on the previous release (worst result across its runs, so a failure that a workaround later hid is still shown). `0.1.0a3` = this run, on a fresh install from PyPI with no workaround. `-` = not run.\n")
groups = OrderedDict()
for i in ids:
    groups.setdefault(by_id[i]["group"], []).append(i)
for g, gids in groups.items():
    w(f"### {g}\n")
    w("| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |")
    w("|---|---|---|---|---|")
    for i in gids:
        e = by_id[i]
        ev = (e["res"].get("pypi") or e["res"].get("head") or next(iter(e["res"].values())))[1]
        # prefer evidence from a failing build so the problem is visible
        for b in ("pypi", "head", "pypi-asr-asis"):
            if b in e["res"] and e["res"][b][0] == "FAIL":
                ev = e["res"][b][1]
                break
        ev = ev.replace("|", "/").replace("\n", " ")[:230]
        w(f"| {i} | {e['claim'].replace('|', '/')} | {prevcell(i)} | {cell(e, 'pypi')} | {ev} |")
    w("")

pa = []
if pa:
    w("### PyPI `[asr]` extra exactly as installed (before the scipy workaround)\n")
    for i in pa:
        e = by_id[i]
        w(f"- **{i}** {e['claim']}: {mark[e['res']['pypi-asr-asis'][0]]}. {e['res']['pypi-asr-asis'][1][:300]}")
    w("")

w("## How to reproduce\n")
w("```bash\ncd ~/Desktop/test-atlasforge  # then: ./run_release.sh  (installs from PyPI in fresh venvs and runs everything)\n#\n# core: PyPI build\nvenv-pypi/bin/python t01_core.py; venv-pypi/bin/python t02_offline.py   # and t03, t04, t07, t08, t10\n# speech + real LLM\nAF_VENV=$PWD/venv-pypi-asr AF_LABEL=pypi-asr-scipyfix venv-pypi-asr/bin/python t05_asr.py\nvenv-pypi/bin/python t06_llm.py\n# unreleased HEAD: same scripts with AF_VENV=$PWD/venv-head AF_LABEL=head\npython report.py\n```\n")
(ROOT / "REPORT_0.1.0a3.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote REPORT.md", len("\n".join(out)), "chars;", len(ids), "distinct checks")
