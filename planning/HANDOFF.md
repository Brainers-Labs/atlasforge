# AtlasForge - Handoff (updated 7 Oct 2026)

## What this is
Brainers Labs' NAIC 2026 Problem 01 entry. Open-source Python toolkit + CLI to **run, evaluate, compare and fine-tune the official N-ATLaS models** (`NCAIR1/*` on Hugging Face), with tone-aware and tone-insensitive scoring for Hausa/Yoruba/Igbo/Nigerian English.
Answers: "I changed this N-ATLaS model. Did I actually make it better on my task, and where did it get worse?"
Deadline: **12 Oct 2026, 11:59 PM WAT** (internal submit 11 Oct). Track: Developer Infrastructure, needs >=2 real external beta testers.

## Why this shape (short)
- N-ATLaS has **no public API**: gated open weights on HF. NAIC gives API credentials only to shortlisted teams. Licence: Awarri custom (1,000 active-user cap, attribution, "Powered by Awarri" on derivatives, commercial use needs agreement).
- Another NAIC team already ships "N-ATLAS Kit" (`natlas` on PyPI, GitHub Kambah123/N-ATLAS-Kit): SDKs, vLLM+FastAPI gateway, playground. It has **no** eval, compare or fine-tune tooling. So we renamed to AtlasForge and own that gap. We interoperate through an OpenAI-compatible backend, we do not build an SDK/gateway/playground.
- No LLM-as-judge and no "hallucination rate" (a non-N-ATLaS judge risks disqualification).

## Where everything is
ONE git repo: https://github.com/Brainers-Labs/atlasforge (public).
- `planning/` - planning docs. Start with `INDEX.md`, then `11_14_DAY_EXECUTION_PLAN.md`, `15_DECISION_LOG.md`, `21_NATLAS_DISCOVERY.md` (verified model facts + a blank verification log to fill on a machine with the models). Inside those docs, `docs/NN_...` paths mean `planning/NN_...`.
- `src/atlasforge/`, `tests/` - the product.

## Code state (29 Sep): 558 tests, 98% coverage; ruff + mypy strict clean
Built and tested on Windows / Python 3.12; the CI matrix covers Linux, macOS and Windows on Python 3.10 and 3.13 (it caught one 3.10-only numpy typing issue, fixed).

**Works and is tested (no model needed):**
- `eval/`: strict JSONL loader, resumable runner (manifest guard, crash-safe, circuit breaker, bounded threading), metrics (EM, accuracy, macro-F1, chrF/chrF++, WER, CER), scoring under tone-aware + tone-insensitive views (failures count as wrong), reports, `validate` (duplicates, leakage, Unicode, diacritics, class balance), `normalize`.
- `compare/`: paired bootstrap, exact McNemar, slice analysis (n>=30 rule), Markdown/JSON report.
- `backends/openai.py`: OpenAI-compatible HTTP backend, tested end-to-end against a real (fake-model) HTTP server.
- `asr/`: ffmpeg decode (incl. real opus/ogg), 30 s windowing, seam-aware merge, `LongAudioBackend`.
- CLI: `doctor`, `run`, `transcribe`, `eval`, `report`, `compare`, `dataset validate`.

**Written but NEVER run against real weights (do this on the Mac first):**
- `backends/local.py` (transformers). Only tested with stand-in torch/transformers modules.
- The official ASR models (loading them, real WER).
- Open questions for `planning/21`: chat template present? real context length? `return_timestamps` behaviour? VRAM/speed?

**Added 2 Oct (tested with stand-ins, 679 tests total):** `finetune/` (QLoRA config, data checks, `--dry-run`, TRL argument-name detection, `training_run.json`), `cards/` + `atlasforge card` (licence-aware model card from `comparison.json` / `training_run.json`), and `--adapter` on the local backend. **The training run itself has never executed: it needs an NVIDIA GPU.**

**Added later on 2 Oct (742 tests, 99% coverage, 9 CI jobs green):**
- **User documentation site** in `docs/` (MkDocs Material). Reference pages and every terminal transcript are generated from the code at build time (`docs_macros.py`); a strict build is a CI job; tests check that every documented command uses real flags. View it: `pip install -e ".[docs]"` then `mkdocs serve` **from inside this repo folder** (`atlasforge/`, where `pyproject.toml` is).
- `atlasforge demo`: synthetic dataset plus two pre-computed runs, so the quickstart works with no model, token, GPU or API.
- `--no-send-repetition-penalty` CLI flag.
- `planning/23_GAP_ANALYSIS.md`: audited gap register with a prioritised plan. **Read it first** for what is left.

**Added 7 Oct (943 tests, ruff + mypy strict clean, strict docs build green; 71 source files):**
- **`atlasforge.evaluate()` / `compare_runs()` / `score_finished_run()`** — the high-level Python API (`api.py`), with `__all__` declaring the stable surface.
- **Custom metrics** — your own `(prediction, reference, example) -> float`, in Python or as `--metric module:function`.
- **Failure-mode flags** — seven deterministic rules (empty, repeated, truncated, format, missing terms, number mismatch, script). Computed at scoring time into `report.json`, rendered in `report.md`, and compared run-by-run in `compare`. Not a quality score, not a hallucination rate.
- **ASR error analysis** — top substitutions/deletions/insertions with example ids, tone-only substitutions marked, pooled WER by reference length. Arithmetic over transcripts: reads no audio and calls no model.
- **HTML reports** — `report.html` and `comparison.html`, written by `eval`, `report` and `compare` beside the JSON and the Markdown. One self-contained file each: no JavaScript, no external stylesheet, no image, no font — so it works air-gapped and as an email attachment. Charts are inline SVG with a dark variant, a `<title>` and `role="img"` per chart, and everything drawn from a report is escaped.
- **`doctor` gated-access check** — probes each NCAIR1 repo and prints the exact accept-the-licence link; a timeout reports as unknown, never as denied.
- **Packaging in CI** — sdist and wheel built, each installed into a fresh venv and run; a core install is asserted to pull in no ML framework.
- **Optional `atlasforge.toml`** — project defaults for backend, base_url, model, quantize, device, timeout, retries; flag → env → file → default; `doctor` reports the file in effect.
- Docs for all of the above; `docs/concepts/failure-modes.md` is the flags page, `docs/reference/configuration.md` the config page, `docs/reference/file-formats.md` describes the HTML.
- **A latent CI failure fixed:** `ruff format --check .` was red on 10 committed files under ruff 0.16.9 (unpinned in CI as `ruff>=0.9`). Reformatted; keep the habit of running it locally, since a newer ruff can change its mind about formatting.

**Added 7 Oct, second pass (G13, G14, G18):**
- **The distribution name is `brainers-atlasforge`, and this was not optional.** Writing the release workflow meant deciding what `pip install` would say, so the name was finally checked against PyPI — and it is **taken**: an unrelated bioinformatics project (gene-family atlases, v0.1.0, 21 Aug 2026). `pip install atlasforge` would have installed someone else's package, so submission-checklist item "`pip install atlasforge` from PyPI" was unachievable as written. The import name, the `atlasforge` command and every example are unchanged (the Pillow/PIL split). The self-referencing extras (`atlasforge[local]`, `atlasforge[docs]`) had to change too — left alone they would have reached for the *other* `atlasforge`. Verified by building the wheel and reading its `METADATA`.
- **Version `0.1.0a1`** (was `0.1.0.dev0`), because `dev0` cannot satisfy the release workflow's tag/version/changelog agreement check.
- **`release.yml`** — on a `v*` tag: verify the tag, the package version and the changelog agree, build sdist + wheel, install the wheel into a fresh venv and run it, write `SHA256SUMS.txt`, create the GitHub release. **`publish.yml`** — manual `workflow_dispatch` only, PyPI trusted publishing, `environment: pypi`; a tag push cannot upload on its own. Human step left: the tag itself, which waits on G3.
- **`docs/project/releasing.md`** — the release steps and the PyPI trusted-publisher fields.
- **G13 needed no code:** `CODE_OF_CONDUCT.md`, both issue forms, `config.yml` and `PULL_REQUEST_TEMPLATE.md` have been in the tree since the first commit; the audit's evidence was stale. No placeholders left.
- **G18 coverage 97.96% → 98.48%:** `decode_audio`'s timeout, `OSError` and zero-sample branches and `__main__` (in-process, because a child process's coverage is invisible) are now covered, along with five unreachable renderer paths. Three defensive `None`-guards in the comparison chart stay uncovered on purpose — removing them trades a coverage number for a crash.
- **The rename's loose end.** Five install hints in `src/` (`backends/local.py`, `doctor.py`, `finetune/qlora.py`, `finetune/config.py`) still told users to run `pip install "atlasforge[local]"`, and four test assertions pinned that wrong string — so nothing went red. That hint appears exactly when an extra is missing, and after the rename it installs a stranger's package. Fixed; `tests/unit/test_packaging.py` now derives the distribution name from `pyproject.toml` and fails if any install command in the source, the docs or an extra names a different one. 953 tests.

**Added 7 Oct, third pass (G3's tooling, example reports, quickstart notebook; 1058 tests, 98.5%, ruff + mypy strict clean, strict docs build green; 77 files checked by mypy):**
- **`scripts/live_smoke.py` + `src/atlasforge/livecheck.py` — the run that fills in `planning/21`'s eight-row log.** This does not *close* G3: the weights, the token and the GPU are still not here, so the log is still empty. What is here is the instrument — the prompt list, the eight checks in the plan's own words, the redaction, the Markdown, and the collection that turns a check into a row. One command writes `evidence/live_<date>.md`; a check that cannot run is a `not run` row, never a blank or an estimate.
- **The module is deliberately split so the coverage number stays meaningful.** Everything that decides *what a row says* — the guards, the access verdicts, the VRAM arithmetic, the revision lookup, the rendering — is pure and injected-where-it-touches-the-world, and 100% covered. The eleven functions that genuinely need torch, the weights or a running server carry `# pragma: no cover` with the reason on the line. Suite 953 → 1028 tests, coverage 98.48% → **98.53%**; without the split the new module alone would have pulled the project total to 95.07%.
- **Nothing in the suite can start a download.** `tests/unit/test_livecheck.py` patches `live_steps`, `hub_revisions`, `environment_facts` and `gpu_facts` before the script's `main` ever runs, and asserts that building the steps executes none of them.
- **The redaction is tested by leaking on purpose:** a token in a caught exception's message and a token in model output both have to be gone from the rendered page.
- **`examples/reports/` — the submission checklist's "example reports committed" item.** A report and a comparison in Markdown, JSON and HTML, generated from the demo data. Because a report has no timestamp and no absolute path in it, regeneration is byte-identical, so `tests/unit/test_examples.py` regenerates them and fails if anything differs: committed example output cannot describe an older version of the tool. The same test asserts both HTML pages stay self-contained (no `<script`, no URL, no `<link>`).
- **`notebooks/atlasforge-quickstart.ipynb` — the whole workflow with no model, token or GPU**, so it runs on a free Colab runtime. Two cells are marked `# notebook: colab-only` (the `%pip install`, and the real-model example that needs a GPU); the rest are executed in order by `tests/unit/test_notebook.py`, which also checks the files the notebook claims to write exist. That test is why the notebook is not prose: it caught a cell passing a `str` where `build_demo` wants a `Path`.
- **`docs/concepts/n-atlas-integration.md` — the "only official models" page the submission asks for.** The five repositories, how each backend loads a model, how `lang` picks a speech checkpoint, and the boundary: the OpenAI protocol is a transport choice, and AtlasForge itself names, loads and installs nothing but the five. Read the claim's machine check with it: `tests/unit/test_model_ids.py`.
- **Three stale `NCAIR1/N-ATLaS-8B` identifiers, found by writing the check before believing the claim.** One in `config.py`'s own docstring example, two more in `docs/reference/configuration.md`, plus a test fixture. That name is not a repository: anyone copying the documented `atlasforge.toml` got a 404 rather than a model. The guard that found them reads every `NCAIR1/...` identifier out of `src/`, `docs/` and `tests/` and fails if it is not one of the five — the same shape as the `pip install` guard, and for the same reason: the project's central claim is that it uses only the official models, and a single wrong string in an example is enough to undermine it. 1154 tests.

**Added 7 Oct, fourth pass (G15, G16, G17 — the three remaining code-side gaps; 1319 tests, 98.64%, ruff + mypy strict clean, 80 files checked by mypy):**
- **`atlasforge bench afrobench` (G15).** A wrapper around `lm-evaluation-harness` for the published AfroBench-LITE suite: it builds the `lm_eval` argv for an `NCAIR1` model, runs it, reads the harness's own results file and writes `bench.json` / `bench.md` beside it, with that raw file kept and named in the report. Two deliberate limits: **task identifiers are discovered from the installed harness, never hard-coded** (they carry language suffixes and move between harness versions), and **the study's published figures are not carried** (they are someone else's measurements of a specific harness revision; copying them would make them read as ours — they stay in `planning/21` with their source). A family the harness has no task for is reported as a gap, and the report says "not zero" in as many words. New `bench` extra; `docs/guides/benchmark-with-afrobench.md`. Tested against stand-ins, `bench.py` at 100% coverage; it has never been run against the harness or the weights, and `docs/help/status.md` says so.
- **The stand-ins could not have caught what the end-to-end run did.** Running the real CLI through a throwaway `lm_eval` package in `%TEMP%` (a real subprocess, a real `--output_path`, a real file read back) found two ways the report described a run that had not happened: every headline metric was scaled by 100, so chrF 38.2 printed as `3820.00` — the harness's key names carry the scale and are now what decides it — and `bench.md` pointed at `harness/results.json` when the harness had written `harness/results_2026-10-07.json`, because the report printed the path it *asked for* rather than the one it read. Both are fixed; `raw_path` now records the file that was actually found, in both halves of the report.
- **Silence-aware chunking (G16).** `plan_windows_silence_aware()` moves each fixed-grid boundary up to `search_s` (2 s default) to the middle of the quietest nearby 20 ms frame, but only if that frame is at or below `silence_db` (-40 dBFS) — so a boundary is never moved on the strength of "less loud than the rest". Window count and overlap are unchanged, and a move that would push a window past the 30 s limit is not taken. Opt-in: `--silence-aware` on `transcribe`, `silence_aware=True` on `evaluate()`. **The fixed grid stays the default**, because which splitter is better is exactly what the live run has not measured.
- **Strict classification metrics (G17).** `accuracy_strict` and `macro_f1_strict`: the whole answer must *be* a label, so `"not positive"` matches nothing instead of `"positive"`. Chosen as two new metric names rather than a CLI flag, because metrics are report keys that already flow through `eval` / `report` / `compare` / `card` and the API — and because the mode travels in the key, a strict run can never be silently averaged with a loose one. The gap between the loose and strict figures is the measure of how often a model answers more than it was asked.
- **The coverage discipline held, and one guard turned out to be reachable.** `bench.py` is 100% by construction (the three functions that need a real harness carry `# pragma: no cover - needs lm-eval installed`). In `chunking.py`, the last uncovered branch was the check that a boundary's search range contains a frame *middle*; it looked defensive, but a test shows it fires on audio a hair over one window long with one sample of search slack, and the planner now has no uncovered line at all. 98.53% → **98.64%**.
- **A test that only failed because the API grew.** `test_speech_datasets_get_the_long_audio_splitter`'s stand-in splitter took the old positional signature, so passing `silence_aware` broke it. Rather than loosen the stand-in, it now records the keyword and asserts it: the default is the fixed grid, and `silence_aware=True` really does reach the constructor that reads it.

**Added 7 Oct, fifth pass (documentation audit — "does the documentation match the features?"; 1332 tests, 98.64%, ruff + mypy strict clean, `mkdocs build --strict` green):**
- **The question was answered by auditing, not by asserting.** The split that made it answerable: the generated reference (CLI reference and every shell fence, metrics / flags / config / errors tables, nav, runnable snippets) is machine-checked and cannot drift; the hand-written prose is where drift lives. Nine seams had: the missing `bench` extra on the installation page's extras table, the benchmark's three output files absent from File formats, no Benchmarks section in Troubleshooting (five real messages undocumented), no Benchmark row on the landing page, `eval.metrics` / `eval.format` / `eval.report` and with them `extract_label` and the strict-match rule missing from the Python API reference, AfroBench-LITE and "few-shot" undefined in the glossary, and the bench guide unreachable from the guides index. All fixed.
- **Documenting the benchmark found a tenth gap that was not documentation: `bench.json` did not record the few-shot setting.** Two runs at different `--num_fewshot` were indistinguishable in AtlasForge's own output. It is recorded now, as `null` rather than `0` when the flag was not passed — "not overridden" is a different claim from "zero-shot", and the harness's per-task default is not necessarily zero.
- **Four guards so these seams cannot drift again, each with a companion test proving the check is not vacuous:** extras ↔ installation page; every artifact a module writes (AST-scanned from module-level name constants) ↔ File formats; every `__all__` entry point and every documentable `eval` submodule ↔ the API reference; every credential (`SECRET_ENV`, `ATLASFORGE_BASE_URL`) and every guide ↔ its page. Writing the API guard first is what found the three missing modules.
- **The real end-to-end run again, since the stand-ins share the code's assumptions.** Both few-shot branches are now confirmed through the real CLI against the throwaway `lm_eval` in `%TEMP%`: no flag → "not overridden" and `"few_shot": null`; `--few-shot 5` → `5` in both halves. The documented JSON example on File formats is that run's output.

**Not written yet (see planning/23):** the Hausa quickstart (needs a native-speaker reviewer), real benchmark packs, and the benchmark-pack licensing decision (a CC-BY-SA share-alike question for an Apache-2.0 repo, deliberately not guessed). The quickstart notebook is written but has not been opened in Colab. Docs are not hosted anywhere yet (enable GitHub Pages if wanted). The release *workflow* exists; the tag does not, because it waits on G3.

## First thing to do on the new machine (Mac M1 16 GB)
```bash
git clone https://github.com/Brainers-Labs/atlasforge && cd atlasforge
python3 -m venv .venv && . .venv/bin/activate      # Python 3.10-3.13
pip install -e ".[dev]"
brew install ffmpeg
pytest && ruff check . && ruff format --check . && mypy
atlasforge doctor
```
Then, to get real evidence:
1. Accept the licence on all 5 NCAIR1 repos (Hugging Face), `export HF_TOKEN=...`.
2. **ASR (fits in memory, CPU/MPS):** `pip install -e ".[asr]"`, then try `atlasforge transcribe some.ogg --lang ha --backend local`. Record the result in `planning/21`.
3. **LLM on the Mac:** fp16 (16 GB) will not fit; 4-bit via bitsandbytes needs CUDA. Serve a quantised model with llama.cpp or Ollama (INFERRED to work for a Llama-3-8B fine-tune, untested) and point `--backend openai --base-url http://127.0.0.1:PORT/v1` at it. Never publish quantised N-ATLaS weights (licence).
4. Run `atlasforge eval` on a real dataset and `compare` two runs. Fill in the verification log in `planning/21`.

## Blockers and TODOs
1. Placeholders: security email in `SECURITY.md`; GitHub org in `pyproject.toml`. ~~confirm `atlasforge` is free on PyPI~~ — **checked 7 Oct and it is taken**; the distribution is `brainers-atlasforge`.
2. Human-only Day-1 actions, none done: secure a GPU with a spend cap (needed for the fine-tune demo and the shared beta endpoint); every member accepts the licence on all 5 NCAIR1 repos; email NAIC the open questions (team-size 2-5 vs 1-6, tester evidence format, whether HF-weights integration satisfies verification, public repo required?); post the beta-tester call (target 5 recruits, need >=2 complete); assign stream owners.
3. D026: find a small, clean-licence domain dataset for the flagship compare + fine-tune demo. Do not invent data.
4. Beta testers need something to run: they need either a GPU box/endpoint or the Mac-style llama.cpp path documented in a quickstart.

## Status vs plan
Phase 1 milestone M1 (29 Sep) is NOT met on the model side (no GPU, no model run yet) but the code side is well ahead of plan: the D3-D8 code (eval, metrics, compare, openai backend, ASR audio, validate, CLI) exists and is tested. As of 7 Oct **every code-side gap in `planning/23` is closed** — G1-G2 (docs, demo), G5-G12, G13 (was already done), G14 (release workflow; the tag is human), G15-G17 (AfroBench wrapper, silence-aware splitting, strict metrics) and G18. What remains is the two items no amount of code can close — a real model run (G3) and beta testers (G4). Both need a person with hardware, a token and the licence, and G3 has an instrument waiting for them (`scripts/live_smoke.py`).

## Rules to keep
Only NCAIR1 models in core paths. Never invent model behaviour (label VERIFIED/INFERRED/UNKNOWN). Never redistribute weights. No secrets/prompts/audio in logs. Core install must not import torch. Never fabricate validation.
