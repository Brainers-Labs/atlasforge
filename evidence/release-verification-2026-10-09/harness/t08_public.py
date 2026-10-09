"""Public-facing claims: live docs site, PyPI page, repository, CI, and the Colab notebook executed from a clean environment."""
import json
import re
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import *

c = Checker("8 Public site, PyPI, CI and notebook")
W = WORK / "public"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)


def get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "atlasforge-release-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # noqa: BLE001
        return 0, str(e).encode()


SITE = "https://brainers-labs.github.io/atlasforge/"
st, body = get(SITE + "sitemap.xml")
urls = [u.text for u in ET.fromstring(body).iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")] if st == 200 else []
with ThreadPoolExecutor(8) as ex:
    codes = list(ex.map(lambda u: (u, get(u)[0]), urls))
badpages = [(u, s) for u, s in codes if s != 200]
c("8.01", "The live documentation site is up and every page in its sitemap loads (HTTP 200), logged out", st == 200 and len(urls) > 30 and not badpages, f"{len(urls)} pages; failing={badpages[:5]}")

# internal link check on the live site
pages = {}
with ThreadPoolExecutor(8) as ex:
    for u, (s, b) in zip(urls, ex.map(get, urls)):
        pages[u] = b.decode("utf-8", "replace")
broken = []
checked = 0
known = set(urls) | {u.rstrip("/") for u in urls}
for u, html in pages.items():
    for href in re.findall(r'href="([^"#]+)(?:#[^"]*)?"', html):
        if href.startswith(("http", "mailto", "javascript", "data:")) or href.endswith((".css", ".js", ".png", ".svg", ".ico", ".json", ".xml", ".woff2")):
            continue
        target = urllib.parse.urljoin(u, href)
        if not target.startswith(SITE):
            continue
        checked += 1
        if target not in known and target.rstrip("/") not in known and target + "/" not in known:
            broken.append((u.replace(SITE, ""), href))
c("8.02", "No broken internal links on the live site", not broken, f"{checked} internal links checked; broken={broken[:6]}")

home = pages.get(SITE, "")
need = ["Quickstart", "Troubleshooting", "Reference"]
c("8.03", "The live home page and navigation are intact (Quickstart, Reference, Troubleshooting present)", all(n in home for n in need), f"{len(home)} bytes")

# Pages content vs reality: the rows that WERE stale must now say what we proved (other rows legitimately stay "not run")
status = re.sub(r"<[^>]+>", " ", pages.get(SITE + "help/status/", ""))
status = " ".join(status.split())
def row(name):
    i = status.find(name)
    return status[i:i + 200] if i >= 0 else ""
asr_row, spk_row, ollama_row = row("Official ASR models"), row("`local` backend, speech models") or row("local backend, speech models"), row("Serving N-ATLaS with Ollama")
integ = re.sub(r"<[^>]+>", " ", pages.get(SITE + "concepts/n-atlas-integration/", ""))
ok = ("Run on real weights" in asr_row) and ("Run on real weights" in ollama_row) and ("131072" in integ)
c("8.04", "The live status page and integration page reflect what was proved: ASR models and Ollama serving 'Run on real weights', context length 131072 recorded",
  ok, f"ASR row: {asr_row[:90]!r} | Ollama row: {ollama_row[:80]!r} | 131072 on integration page: {'131072' in integ}")

# ---------------------------------------------------------------- PyPI page
st, body = get("https://pypi.org/pypi/brainers-atlasforge/json")
meta = json.loads(body) if st == 200 else {}
info = meta.get("info", {})
c("8.05", "PyPI: name, version, Python range, licence and project URLs are correct",
  info.get("version") == "0.1.0a2" and info.get("requires_python") == ">=3.10" and (info.get("license_expression") == "Apache-2.0") and all("Brainers-Labs" in v for v in info.get("project_urls", {}).values()),
  f"version={info.get('version')} python={info.get('requires_python')} license={info.get('license_expression')} urls={info.get('project_urls')}")
desc = info.get("description", "")
stale = [t for t in ("never been run", "Not yet run against real weights", "have not been loaded", "has not been run either") if t.lower() in desc.lower()]
c("8.06", "PyPI project description is accurate now that the speech models and LLM were run on real weights", not stale,
  f"PyPI page (frozen at release 0.1.0a2) still says: {stale}. It can only be corrected by publishing a new version.")
st, _ = get("https://pypi.org/project/brainers-atlasforge/")
c("8.07", "The PyPI project page loads", st == 200, f"HTTP {st}")

# ---------------------------------------------------------------- repository
st, _ = get("https://github.com/Brainers-Labs/atlasforge")
c("8.08", "The GitHub repository is public (loads logged out)", st == 200, f"HTTP {st}")
for path in ("LICENSE", "README.md", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md"):
    st, _ = get(f"https://raw.githubusercontent.com/Brainers-Labs/atlasforge/main/{path}")
    c(f"8.09-{path}", f"Repository has {path}", st == 200, f"HTTP {st}")
st, body = get("https://raw.githubusercontent.com/Brainers-Labs/atlasforge/main/SECURITY.md")
c("8.10", "SECURITY.md has a real private reporting route (advisory form or email) and no leftover TODO placeholder", st == 200 and "TODO" not in body.decode() and ("security/advisories/new" in body.decode() or "@" in body.decode()), re.sub(r"\s+", " ", body.decode())[:200])
st, body = get("https://raw.githubusercontent.com/Brainers-Labs/atlasforge/main/pyproject.toml")
c("8.11", "pyproject.toml has no placeholder organisation text", st == 200 and "NOTE: confirm" not in body.decode() and "brainerslabs/atlasforge" not in body.decode(), "no placeholder markers")

# ---------------------------------------------------------------- CI: three operating systems
r = sh("gh api 'repos/Brainers-Labs/atlasforge/actions/runs?branch=main&per_page=12' --jq '.workflow_runs[] | [.name, .conclusion, .head_sha[0:7], .created_at] | @tsv'", log="t08_ci")
runs = [l.split("\t") for l in r.out.strip().splitlines()]
latest = {}
for name, concl, sha, ts in runs:
    latest.setdefault(name, (concl, sha, ts))
c("8.12", "Latest CI runs on main are green (CI, docs, release workflows where present)", bool(latest) and all(v[0] in ("success", "skipped") for v in latest.values() if v[0]), f"{ {k: v[0] for k, v in latest.items()} }", "t08_ci")
ci = next((l for l in runs if l[0] == "CI"), None)
if ci:
    rid = sh("gh api 'repos/Brainers-Labs/atlasforge/actions/runs?branch=main&per_page=10' --jq '[.workflow_runs[] | select(.name==\"CI\")][0].id'").out.strip()
    jobs = sh(f"gh api repos/Brainers-Labs/atlasforge/actions/runs/{rid}/jobs --jq '.jobs[] | [.name, .conclusion] | @tsv'", log="t08_ci_jobs").out.strip().splitlines()
    names = " ".join(jobs)
    ok = all(o in names for o in ("ubuntu", "macos", "windows")) and all(j.split("\t")[1] == "success" for j in jobs)
    c("8.13", "CI passes on Linux, macOS and Windows with Python 3.10 and 3.13 (the support claim)", ok and "3.10" in names and "3.13" in names, f"{len(jobs)} jobs: " + "; ".join(j.replace("\t", "=") for j in jobs)[:260], "t08_ci_jobs")

# ---------------------------------------------------------------- the notebook, executed
import os
NB = Path(os.environ.get("ATLASFORGE_REPO", ROOT.parents[2])) / "notebooks" / "atlasforge-quickstart.ipynb"
st, _ = get("https://colab.research.google.com/github/Brainers-Labs/atlasforge/blob/main/notebooks/atlasforge-quickstart.ipynb")
c("8.14", "The Colab link resolves", st == 200, f"HTTP {st}")
st, nbbody = get("https://raw.githubusercontent.com/Brainers-Labs/atlasforge/main/notebooks/atlasforge-quickstart.ipynb")
c("8.15", "The notebook exists in the public repository (so the Colab link has something to open)", st == 200 and len(nbbody) > 2000, f"HTTP {st}, {len(nbbody)} bytes")
if st == 200:
    shutil.rmtree(ROOT / "venv-nb", ignore_errors=True)
    sh(f"python3 -m venv {ROOT}/venv-nb && {ROOT}/venv-nb/bin/pip install -q --upgrade pip && {ROOT}/venv-nb/bin/pip install -q brainers-atlasforge nbclient ipykernel nbformat", timeout=900, log="t08_nb_install")
    (W / "nb.ipynb").write_bytes(nbbody)
    runner = '''
import nbformat, sys
from nbclient import NotebookClient
nb = nbformat.read("nb.ipynb", as_version=4)
# Colab-only cells (marked) are Colab's job: the %pip line there does the same install already done in this venv
for cell in nb.cells:
    if cell.cell_type == "code" and cell.source.startswith("# notebook: colab-only"):
        cell.source = "pass"
NotebookClient(nb, timeout=300, kernel_name="python3").execute()
out = "".join(o.get("text", "") for cell in nb.cells if cell.cell_type == "code" for o in cell.get("outputs", []))
print("CELLS_OK", sum(1 for x in nb.cells if x.cell_type == "code"))
print("HAS_REGRESSION", "has_number" in out, "REGRESS" in out.upper())
'''
    (W / "run_nb.py").write_text(runner)
    r = sh([str(ROOT / "venv-nb/bin/python"), "run_nb.py"], cwd=W, timeout=600, log="t08_nb_run")
    c("8.16", "The quickstart notebook runs top to bottom in a clean environment using the PUBLISHED package, with no errors",
      r.rc == 0 and "CELLS_OK" in r.out, f"{r.out.strip()[-120:]}" if r.rc == 0 else r.err.strip()[-300:], "t08_nb_run")
