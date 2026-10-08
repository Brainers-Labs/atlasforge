# Gap Analysis and Closing Plan

Audited 2 Oct 2026 against the code at `eb202e9` (679 tests, 98% coverage, 55 source files, ~4,500 lines), the PRD ([03](03_PRD.md)), the API design ([05](05_SDK_API_DESIGN.md)) and the submission checklist ([16](16_SUBMISSION_CHECKLIST.md)).

## Verdict

The engine is solid and honest. The gaps are **(a) things the planning docs promised and the code does not do, (b) everything that needs a real model, and (c) the surface a stranger touches** (docs, examples, packaging). Almost no uncovered code remains (98%), so test coverage is not where the risk is.

## Gap register

Severity: **S1** blocks NAIC submission or user trust · **S2** a promised feature is missing · **S3** polish.

| # | Gap | Evidence | Sev | Effort | Plan |
|---|---|---|---|---|---|
| G1 | **No user documentation** | ~~Only README + planning docs~~ | S1 | 1-2 d | **DONE 2 Oct.** MkDocs Material site, 36 documented commands checked by tests, strict CI build |
| G2 | **No example data or sample runs** | ~~Nobody could try the tool without a model~~ | S1 | 0.5 d | **DONE 2 Oct.** `atlasforge demo` writes synthetic data and two runs; the quickstart works offline |
| G3 | **Nothing verified on real N-ATLaS** | `planning/21` log: 0 of 8 rows; local backend, ASR, finetune never ran on weights | S1 | needs hardware | **Instrument ready 7 Oct.; the run is still outstanding.** `scripts/live_smoke.py` (core in `src/atlasforge/livecheck.py`) runs the eight checks in `planning/21`'s order and words, and writes `evidence/live_<date>.md`. It has **never been executed** against the weights, so the log is still 0 of 8 — closing this needs the hardware, a token and the licence, not more code. Mac: ASR first, then llama.cpp/Ollama LLM via the `openai` backend; GPU box: the LLM and finetune |
| G4 | **No beta testers** | NAIC requires ≥2; not recruited | S1 | human | Recruit now; testers need the quickstart (G1) and a serving path |
| G5 | **No high-level Python API** | ~~[05](05_SDK_API_DESIGN.md) promises `evaluate(backend, dataset, ...)`; users must call `load_dataset` + `run` + `score_run` themselves~~ | S2 | 0.5 d | **DONE 7 Oct.** `atlasforge.evaluate()` / `compare_runs()` / `score_finished_run()` in `api.py`, `__all__` declared, documented in the Python API reference. `compare` keeps its name because the subpackage owns it |
| G6 | **No failure-mode flags** | ~~D023/PRD: format compliance, missing required terms, number mismatch, empty/repeated output~~ | S2 | 1 d | **DONE 7 Oct.** Seven deterministic flags in `eval/flags.py`, summarised in every report and compared run-by-run. One deviation, below |
| G7 | **No HTML report / visuals** | ~~PRD P1; users asked for visuals~~ | S2 | 1 d | **DONE 7 Oct.** `html.py` renders `report.html` and `comparison.html`: one self-contained file each (inline SVG, no JavaScript, no external reference), written by `eval`, `report` and `compare` beside the Markdown and JSON |
| G8 | **No ASR error analysis** | ~~PRD: top substitutions/deletions/insertions, WER by length. Only WER/CER exist~~ | S2 | 0.5 d | **DONE 7 Oct.** `eval/asr_analysis.py` over `jiwer` alignments; tone-only substitutions marked; rendered in every speech report and written to `report.json` under `asr` |
| G9 | **No custom metrics** | ~~PRD/[05](05_SDK_API_DESIGN.md): `(pred, ref, example) -> float` callable. Metrics are a fixed registry~~ | S2 | 0.5 d | **DONE 7 Oct.** Python callables and CLI `--metric module:function`; per-example values flow into `compare` |
| G10 | **Packaging never exercised** | ~~CI installs editable only; sdist/wheel never built~~ | S2 | 0.25 d | **DONE 7 Oct.** CI builds sdist + wheel, installs each into a fresh venv and runs `atlasforge --version`; a separate job asserts a core install pulls no ML package. **The PyPI name was then checked and it is taken** — an unrelated project owns `atlasforge`, so the distribution is `brainers-atlasforge`. See G14 |
| G11 | **`doctor` cannot check gated access** | ~~Planned per-repo access check; only token presence is checked~~ | S2 | 0.5 d | **DONE 7 Oct.** `check_gated_access` probes each NCAIR1 repo and prints the exact accept-the-licence link; a timeout is reported as unknown, never as denied |
| G12 | **No project config file** | ~~[05](05_SDK_API_DESIGN.md) precedence mentions `atlasforge.toml`; none read~~ | S3 | 0.5 d | **DONE 7 Oct.** `config.py` reads the nearest `atlasforge.toml` at or above the working directory; flag → env → file → default, enforced by Click's parameter source. Seven keys; an unknown key is an error. `doctor` reports the file |
| G13 | **Missing OSS hygiene** | ~~No `CODE_OF_CONDUCT.md`, issue/PR templates~~ | S3 | 0.25 d | **DONE (was already in the tree; the audit's evidence was stale).** `CODE_OF_CONDUCT.md`, both issue forms, `config.yml` and `PULL_REQUEST_TEMPLATE.md` are present, and no placeholder email or org remains |
| G14 | **Release process absent** | ~~No tag, no release workflow, version `0.1.0.dev0`~~ | S3 | 0.25 d | **DONE 7 Oct. for the workflow half.** `release.yml` (tag → verify tag/version/changelog agree → build → install the wheel in a fresh venv → checksums → GitHub release), `publish.yml` (manual, trusted publishing to PyPI), `docs/project/releasing.md`. Version is `0.1.0a1`. **Tagging is still the human step, deliberately: it waits on G3.** The distribution name had to change — see the closing log |
| G15 | **AfroBench wrapper** | ~~PRD P1; not started~~ | S3 | 1 d | **DONE 7 Oct.** `atlasforge bench afrobench` (module `bench.py`) builds the `lm_eval` command for an `NCAIR1` model, runs it, reads the harness's results file and writes `bench.json` / `bench.md` beside it. **Task identifiers are discovered from the installed harness, never hard-coded**, and a family that matches nothing is reported as a gap rather than a zero. The study's published figures are not copied into the report. Tested against stand-ins; never run against the harness or the weights |
| G16 | **Fixed-window ASR chunking** | ~~Words cut at seams can be misheard (documented limitation)~~ | S3 | 1 d | **DONE 7 Oct.** `plan_windows_silence_aware()` moves each boundary up to `search_s` to the middle of the quietest nearby 20 ms frame, provided that frame is at or below `silence_db`; window count and overlap are unchanged. Opt-in (`--silence-aware`, `evaluate(..., silence_aware=True)`); the fixed grid stays the default because which splitter is better is unmeasured |
| G17 | **Label extraction ignores negation** | ~~`extract_label("not positive")` returns `positive` (documented)~~ | S3 | 0.5 d | **DONE 7 Oct.** Two new metrics, `accuracy_strict` and `macro_f1_strict`: the whole answer must *be* a label, so `"not positive"` matches nothing. Loose and strict are separate report keys, so a strict run can never be averaged with a loose one; the gap between them measures how often the model answers more than it was asked |
| G19 | **CLI could not disable `repetition_penalty`** | The backend's own hint told users to, but no flag existed | S2 | 0.1 d | **DONE 2 Oct.** `--no-send-repetition-penalty` |
| G18 | **Residual uncovered lines** | `render.py` 89%, `asr/audio.py` 90%, `__main__` | S3 | 0.25 d | **DONE 7 Oct.** Timeout, `OSError` and zero-sample branches of `decode_audio` covered; `__main__` exercised in-process (the subprocess test proved it but coverage in a child process is invisible, which is how a permanently-0% file hides a real gap). Reachable HTML renderer paths covered too. Total 98.5%. What remains is three defensive guards that no caller can reach, left in deliberately |

## Priority order

1. **G1 + G2: documentation and examples.** Done in this change: an MkDocs Material site with a guided path, concepts, task guides, generated CLI reference, API reference, file formats, troubleshooting, FAQ and an honest status page. The quickstart runs offline on bundled sample data, so a reader can reach their first comparison in two minutes without any model.
2. **G5, G9, G10: done 7 Oct.** AtlasForge is usable as a library and installs cleanly from a built wheel.
3. **G6, G7, G8: done 7 Oct.** The "understand" features exist: flags, ASR error analysis, and a self-contained HTML page for each report.
4. **G3, G4 (in parallel, human + hardware):** real-model evidence and testers. These gate the submission and should start immediately. G3's instrument is now built and tested — the run itself is what is missing, and it needs the hardware, a token and the licence, not more code.
5. **G11, G12, G13, G14, G18 done (7 Oct.).** Hygiene and release. One release step stays human: the tag, which waits on G3.
6. **G15, G16, G17: done 7 Oct.** The three remaining code-side rows. Both of the ones the register had
   deferred to "after NAIC" are now in the tree, opt-in and off by default, so nothing that was already
   measured changes underneath it. What is left open is only G3 (real weights) and G4 (testers) — the two
   the register always said no amount of code could close.

## Closing log

### 7 Oct. 2026 — G5, G6, G8, G9, G10, G11, G12

Closed by code, each verified the same way: `ruff`, strict `mypy`, the full `pytest` suite, a strict
`mkdocs build`, and a real CLI run in a temp directory rather than a mocked one.

- **G6 deviation, deliberate.** The register said flags belong in `results.jsonl`. They are instead
  computed at scoring time and written to `report.json` (`flags`, `n_flagged`, `per_example_flags`).
  Scoring is free and re-runs on every `report`, so flags cannot go stale, and the results format —
  which every existing tool reads — does not change. Merging flags into results later is additive.
- **A bug the real run caught.** The ASR summary line printed a rate that disagreed with the WER in
  the metrics table above it: an insertion is an error but not a *reference* word, so the
  denominator is hits + substitutions + deletions. Unit tests with no insertions had passed.
  The test now includes one.
- **A trap worth recording.** Precedence is enforced by asking Click where each value came from,
  and the obvious check silently never fires: Typer vendors its own copy of Click, so its
  `ParameterSource` is a *different* enum class from `click.core.ParameterSource`, and `is`
  against the installed Click's member is always false. The test that a flag beats the file is
  what caught it. Source is compared by member name, with the reason in a docstring.
- **Still untouchable without new resources:** G3 (real weights — GPU for fine-tune, gated access
  for the rest) and G4 (beta testers, a human task). Everything else closes in code.

### 7 Oct. 2026 — G7

`html.py` is the whole of the change: one module, two entry points (`report_html`,
`comparison_html`), no templating dependency and no JavaScript. `eval`, `report` and `compare`
now write a third file beside the JSON and the Markdown, and the CLI says so.

- **Interpretation, stated rather than assumed.** The register's "static viewer page" is the
  generated `report.html` / `comparison.html` themselves — open them from the filesystem, no
  server and no network. A separate viewer page that loads a `report.json` would have needed
  JavaScript, which the same register line rules out. The status page now says what the charts
  are (static inline SVG: no zoom, no tooltips, no filtering) instead of leaving it implied.
- **Two renderers, one arithmetic.** `asr_summary()` moved into `eval/report.py` and both
  renderers call it, because the pooled word error rate is exactly the number that was wrong
  once before: an insertion is an error but not a *reference* word. Two copies of that
  denominator would have gone out of step again. A test asserts both renderings contain the
  same rate.
- **Chart bugs the inspection found, not the tests.** The count charts (flags, ASR length
  buckets) were drawn on a 0–100% scale and labelled in percent, which reads as a share of
  something the reader cannot see; they now carry count ticks. The grouped comparison chart
  then clamped a count against a percentage maximum, so a base of 6 and a candidate of 14 both
  drew full-width bars. Both were found by rendering the page and reading the SVG geometry.
- **Escaping is tested with hostile input**, not asserted in a comment: a manifest field and a
  transcript carrying `<script>` and `<img onerror=...>` must appear as text.
- **A CI gate that was already red.** `ruff format --check .` — a CI step, unpinned against
  `ruff>=0.9` — failed on 10 committed files under ruff 0.16.9. Nothing to do with G7, but it
  means CI could not have passed as the tree stood. `ruff format .` applied.

### 7 Oct. 2026 — G13, G14, G18

- **G13 was already closed.** `CODE_OF_CONDUCT.md`, both issue forms, `config.yml` and
  `PULL_REQUEST_TEMPLATE.md` have been in the tree since the first commit; the audit's evidence
  was stale. Checked for leftover placeholders: none. No code changed.
- **G14, and the finding that forced a rename.** Writing the release workflow meant deciding what
  `pip install` would say, so the name was checked against PyPI for the first time. **`atlasforge`
  is taken** — an unrelated bioinformatics project (gene-family atlases), v0.1.0, uploaded
  21 Aug 2026. `pip install atlasforge` would have installed it, and the submission checklist's
  "`pip install atlasforge` works from PyPI in a fresh venv" was therefore unachievable as
  written. The distribution is now `brainers-atlasforge`; the import name, the command and every
  example are unchanged. The extras that name the project (`atlasforge[local]`,
  `atlasforge[docs]`) had to change with it — left alone, `pip install -e ".[dev]"` would have
  reached for the *other* `atlasforge` on the index. Verified by building the wheel and reading
  its `METADATA`: the name is `brainers-atlasforge`, the console script is still `atlasforge`, and
  hatchling has expanded the self-referencing extras inline rather than leaving a foreign
  requirement behind.
- **The version moved to `0.1.0a1` (unreleased).** `0.1.0.dev0` could not be tagged as planned:
  the release workflow asserts that the tag, the package version and the changelog agree. Tagging
  itself is still the human step, and still waits on G3.
- **Publishing is a separate, manual workflow by design.** A tag push builds, verifies and
  attaches artefacts to a GitHub release; it cannot upload to PyPI. That needs a trusted publisher
  configured on PyPI once, and `docs/project/releasing.md` records the exact fields.
- **G18.** The three real untested branches in `decode_audio` (timeout, `OSError`, ffmpeg exiting
  0 with no output) now have tests, as does `__main__`. The ASR length chart with no buckets, a
  classification report with no flags section, a pooled-only metric in a comparison, a comparison
  with no slices and a pooled figure missing from one run were all unreachable by the suite and
  are now covered. Overall coverage 97.96% → 98.46%. Three guards in the comparison chart remain
  uncovered: they are `None`-checks the type checker requires but no caller can trigger, since
  every call site filters first. Removing them would trade a coverage number for a crash, so they
  stay.
- **The rename's loose end, found by grepping for it afterwards.** The rename changed
  `pyproject.toml`, the docs and the tests, but **five install hints in `src/` still said
  `pip install "atlasforge[local]"`** (`backends/local.py` twice, `doctor.py`, `finetune/qlora.py`,
  `finetune/config.py`) — and four test assertions pinned the wrong string, which is why nothing
  went red. Those hints are what a user sees at the moment the extra is missing, and after the
  rename they point at someone else's package. Fixed, and `tests/unit/test_packaging.py` now reads
  the distribution name from `pyproject.toml` and fails if any `pip install "..."` in the source,
  the docs or an extra names a different one — the guard the rename should have had.

### 7 Oct. 2026 — G3's instrument, the example artefacts, and the model-identity claim

G3 stays open, and this entry is deliberately not marked done. What it records is a distinction that
matters for the submission: the *evidence* was missing partly because nobody had built the thing
that produces it. The rest covers checklist items that needed no hardware — example reports, a
notebook, and the "only official models" page — and that are the same kind of work: artefacts a
reviewer will look for, which nothing in the repo previously produced.

- **`scripts/live_smoke.py` and `src/atlasforge/livecheck.py`.** One command runs the eight checks in
  `planning/21`'s order and its own words — gated access, LLM load and VRAM, four languages through
  the chat template, `max_position_embeddings`, a served completion, one ASR transcription per
  language, `return_timestamps`, and the five commit SHAs — and writes `evidence/live_<date>.md`.
  `--audio-dir` and `--base-url` are optional, and a check whose input is missing is a `not run` row
  naming what was missing, not a skipped line. The eight rows are the deliverable; a partly filled
  log is still evidence, an invented one is not.
- **A failure is a row, not a crash.** `collect` catches per step, keeps the exception's own words
  and continues, because the purpose of the run is to find out *which* checks work. A `hub_revisions`
  failure at the top is caught the same way and the file is still written.
- **The split that keeps the coverage number honest.** The new module could have dropped the project
  from 98.48% to 95.07% — 243 statements, most of which only execute on a GPU. Instead each probe is
  a tested guard plus a small body marked `# pragma: no cover - needs a GPU / the weights / a server`.
  The guards (access verdicts, VRAM arithmetic, clip lookup, revision reading, the `not run`
  decisions) and the delegation are 100% covered; the excluded lines are exactly the ones that have
  never run, which is the claim `docs/help/status.md` already makes. Net: 1028 tests, 98.53%.
- **The suite cannot download a model.** `tests/unit/test_livecheck.py` replaces `live_steps`,
  `hub_revisions`, `environment_facts` and `gpu_facts`, and asserts that building the steps executes
  none of them — so no test can start a 16 GB fetch on a laptop.
- **Credentials are tested by trying to leak them.** A token echoed inside a caught exception, and a
  token appearing in a model's own output, both have to be absent from the rendered page; the
  environment values are replaced first and an `hf_`-shaped string is masked even when no variable
  named it.
- **`examples/reports/`, because "example reports committed" was a checklist box with nothing behind
  it.** A report and a comparison in all three formats, from the demo data. The interesting part is
  not the files but why they can be committed safely: a report carries no timestamp and no absolute
  path, so regenerating it is byte-identical, and `tests/unit/test_examples.py` regenerates all six
  and compares. Committed example output is normally a liability — a description of a past version
  of the code — and this is the version of that problem with a machine check attached. A change that
  alters a report now cannot land without the example moving in the same commit.
- **A quickstart notebook, and the convention that makes it checkable.** Nothing imports a notebook,
  so nothing notices it going stale. Two cells carry `# notebook: colab-only` and are skipped by the
  suite (the `%pip install`, and the real-model example that needs a GPU); every other cell is
  executed in order by `tests/unit/test_notebook.py` and the files it claims to write are asserted
  present. It earned its keep immediately: the first run failed because a cell passed `"atlasforge-demo"`
  where `build_demo` requires a `Path`.
- **The "only official models" claim, written down and then checked — and it was false in three
  places.** `docs/concepts/n-atlas-integration.md` states the claim the submission rests on: the five
  repositories, how each backend loads one, how `lang` selects a speech checkpoint, and the
  distinction between a transport choice (any OpenAI-compatible server) and the models AtlasForge
  itself names. Writing the machine check *before* believing the claim found `NCAIR1/N-ATLaS-8B` — a
  repository that does not exist — in `config.py`'s own docstring example, twice in
  `docs/reference/configuration.md`, and once in a test fixture. Anyone copying the documented
  `atlasforge.toml` would have got a 404 from Hugging Face rather than a model.
  `tests/unit/test_model_ids.py` now reads every `NCAIR1/...` identifier out of the source, the docs
  and the tests and fails if one is not among the five, with the adapter form (`+my-adapter`) accepted
  as its base. Same shape as the `pip install` guard: one wrong string in an example is enough to
  undermine a claim the whole submission makes.

### 7 Oct. 2026 — G15, G16, G17

The last three code-side rows, and the two the register had deferred to "after NAIC" are now in the
tree, opt-in. 1319 tests, 98.64% statement coverage, `ruff` and strict `mypy` clean.

- **G17, and why it is two metric names instead of a `--strict` flag.** A mode flag would have to be
  threaded through `eval`, `report`, `compare`, `card` and the API, and a run scored in one mode could
  be compared against another without anything saying so. Two metric names cost nothing: metrics are
  already report keys that everything downstream iterates over, the mode is *in* the key, and a strict
  result can therefore never be averaged with a loose one. The gap between `accuracy` and
  `accuracy_strict` is itself the useful number — how often the model answers more than it was asked.
  The docs table is generated from the registry, so adding a metric without documenting it is a build
  failure rather than a silent omission.
- **G16, and the fallback that was deleted.** The first draft guarded every plan with a re-check and
  fell back to the fixed grid. Working the window arithmetic instead showed the fallback unreachable:
  each window's length is bounded at the boundary that *ends* it, and the search is additionally capped
  at the slack between the nominal window and the models' 30 s limit — so a move cannot push a window
  over, and the final window is bounded by `plan_windows` plus a search that stays inside the cap. The
  dead branch went, and a parametrised property grid (seven lengths × five pause positions) asserts the
  legality the fallback used to paper over: same window count as the fixed grid, full coverage, every
  window inside the limit, contiguous with the requested overlap.
- **Two design errors caught by reasoning rather than by a test.** A tail constraint was briefly applied
  to the *upper* end of a boundary's reach, which would have made `low > high` for the final boundary
  and silently disabled snapping there — the one boundary where a pause matters most. And a legality
  check compared against the window length instead of the 30 s limit. Both were found by recomputing
  the ranges by hand before the code was believed.
- **A guard that looked defensive and was not.** The planner refuses to look for a frame whose *middle*
  is outside the reachable range. That reads like dead code for the type checker, and the branch was
  the last uncovered line in `chunking.py` — so the case was constructed rather than assumed: audio one
  sample over a single window, with window settings leaving exactly one sample of search slack, puts the
  only boundary's whole reach between two frame centres. It fires, keeps the grid and does not slice past
  the end of the array. `chunking.py` is now 100% statements and branches, with no pragma in it.
- **G15, and the two things the wrapper refuses to do.** It does not hard-code task identifiers: the
  harness names them per language (`afrixnli_yo`, `belebele_hau`) and the set changes between versions,
  so they are discovered from the installed harness and a family that matches nothing is reported as a
  **gap, not a zero** — "we did not run it" and "it scored nothing" are different claims, and the report
  says "not zero" in as many words. And it does not carry the study's published figures: those are
  someone else's measurements of a specific harness revision on a specific dataset revision, they live in
  `planning/21` with their source, and copying them into a report would make them read as ours. The
  module docstring says plainly that it has been tested against stand-ins and never run against the
  harness or the weights; so does `docs/help/status.md`.
- **Both of these paths are off by default.** The fixed grid stays the default splitter and the loose
  metrics stay the default names, so no previously-measured number changes meaning underneath it. That
  is the same rule the register used for G6's deviation: additive, or documented as a change.
- **A precedence rule that had no test pinning it, found by a coverage number that moved.** Re-running
  the suite gave 98.64% and then 98.62% with the same tests passing — one branch in
  `eval/metrics.py` covered on one run and not the next. The cause was in the tests, not the code:
  `extract_label`'s label fixtures were `frozenset`s, and a set's iteration order varies with
  `PYTHONHASHSEED`, so whether a *worse* candidate was ever compared against the best so far was a
  property of the hash seed. The function's own answer is order-independent — the candidate tuple
  `(index, -len(label), label)` is a total order, so first-occurrence-wins is deterministic whatever
  container a caller passes — but nothing asserted it. The fixtures are tuples now, ordered the way the
  docstring states the rule, and there is an explicit test that a later match does not displace an
  earlier one. That test failed when first written, for a reason worth keeping: `"positive,"` is not a
  whole-word match, so the comma version of the case tested nothing at all. Stable at 98.64% across
  seeds, and `metrics.py` at 100%.
- **Two bugs the end-to-end run caught, which no unit test would have.** The wrapper's tests use
  stand-ins, so they encode the same assumptions as the code. Running the real CLI through a throwaway
  `lm_eval` package in `%TEMP%` — a real subprocess, a real `--output_path`, a real file to read back —
  caught two things at once. First, the report scaled *every* headline metric by 100, because the code
  treated "the harness reported a 0-1 fraction" as a property of harness figures in general; chrF came
  out as `3820.00` for 38.2. The harness's own key names carry the scale, so `fmt_metric` reads them and
  chrF and BLEU stay where they are. Second, `bench.md` announced the raw output at
  `harness/results.json` while the harness had written `harness/results_2026-10-07.json`: the report was
  printing the path it *asked for*, not the path `find_results` had just returned. It now records the
  file it actually read. Both are the same failure mode — the report asserting something about the run
  that the run did not do — and it is the failure a wrapper most needs to avoid, since the whole point of
  keeping the raw file beside the report is that a reader can check one against the other. A third, of
  the same shape and found by re-reading rather than by running: the error raised for harness output
  without a `results` object carried the hint "Point `--results` at the file lm_eval writes it to" — and
  no such flag exists on this CLI, so the fix it suggested was a second error. The hint now names the
  file, and its test asserts the absence of the bogus flag alongside the presence of the real one. Every
  other `--flag` a hint or help string names was then checked against the live command tree: the rest
  are real.

### 7 Oct. 2026 — the documentation audit

Asked directly whether the docs matched the features, and answered by auditing rather than by
asserting. The distinction that made it tractable: some pages **cannot** drift, because a test
generates or checks them (the CLI reference and every shell fence in the guide, the metrics, flags,
config and errors tables, the nav, the runnable snippets). Everything else is hand-written, and that
is where the question had an answer. 1332 tests, 98.64%, `ruff` and strict `mypy` clean, strict
`mkdocs build` green.

- **Nine seams that had drifted, all in prose rather than in the generated half.** The `bench` extra
  was declared and hinted by its own command but missing from the installation page's extras table —
  the one place the six are listed together. The benchmark's three output files were absent from
  [File formats]. Troubleshooting had no Benchmarks section, so the wrapper's five messages (no
  harness installed; no task matched the families; the harness does not have `X`; the harness exited
  `1`; it exited cleanly and wrote nothing) were undocumented. The landing page's "what you get" table
  never mentioned `bench`. Three modules — `eval.metrics`, `eval.format`, `eval.report` — and with
  them `extract_label` and the strict-match rule were missing from the Python API reference.
  AfroBench-LITE and "few-shot" were undefined in the glossary, and the bench guide was unreachable
  from the guides index.
- **A tenth gap was not a documentation gap at all, which is what documenting it found.**
  Re-reading what `bench.json` actually recorded showed the few-shot setting was in neither half of
  the report, so two runs at different `--num_fewshot` were indistinguishable in AtlasForge's own
  output. The field is now there, and it is `null` rather than `0` when the flag was not passed,
  printed as "the harness's own default for each task (not overridden)" — because "not overridden" is
  a different claim from "zero-shot", and the harness's per-task default is not necessarily zero.
- **Four guards, and a companion test for each, because a guard that cannot fail is prose.** The
  extras declared in `pyproject.toml` must each appear on the installation page; every artifact a
  module writes (scanned by AST for uppercase module-level name constants that look like filenames)
  must appear on the file-formats page; every entry point named in `__all__` and every documentable
  `eval` submodule must appear in the API reference; every credential and every guide must be
  documented. Each has a second test asserting the check finds the thing it is looking for, so a
  regex that silently matches nothing cannot pass. Writing the API guard first is what found the three
  missing modules — a claim tested before it is believed.
- **The end-to-end run again, because the stand-ins share the code's assumptions.** The real CLI
  through the throwaway `lm_eval` in `%TEMP%` confirms both few-shot branches now: with no flag,
  `bench.md` says the setting was not overridden and `bench.json` carries `"few_shot": null`; with
  `--few-shot 5`, both say `5`. That run is also what keeps the documented JSON example in
  [File formats] honest — its `raw_path`, `few_shot` and `missing_families` are the values this run
  produced.

## Documentation standards adopted

Modelled on cloud and API documentation (Stripe, Anthropic, AWS):

- **Task-oriented navigation:** Get started → Concepts → Guides → Reference, plus Troubleshooting, FAQ and Status.
- **Every guide ends in a result the reader can see,** and the first one works with no model.
- **Reference is generated from the code** (CLI from the Typer app, API from docstrings), so it cannot drift.
- **Snippets are tested:** a test checks that every `atlasforge ...` command shown in the docs uses flags that really exist.
- **Honesty page:** `status.md` lists what is verified against real models and what is not.
- **Strict build:** broken links or missing pages fail CI.
