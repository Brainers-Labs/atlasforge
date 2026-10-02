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
| G3 | **Nothing verified on real N-ATLaS** | `planning/21` log: 0 of 8 rows; local backend, ASR, finetune never ran on weights | S1 | needs hardware | Mac: ASR first, then llama.cpp/Ollama LLM via `openai` backend; GPU box: finetune. `scripts/live_smoke.py` to produce the evidence file |
| G4 | **No beta testers** | NAIC requires ≥2; not recruited | S1 | human | Recruit now; testers need the quickstart (G1) and a serving path |
| G5 | **No high-level Python API** | [05](05_SDK_API_DESIGN.md) promises `evaluate(backend, dataset, ...)`; users must call `load_dataset` + `run` + `score_run` themselves. `atlasforge/__init__.py` exports only `__version__` | S2 | 0.5 d | Add `atlasforge.evaluate()` / `compare()` convenience layer, define `__all__`, document the stable surface |
| G6 | **No failure-mode flags** | D023/PRD: format compliance, missing required terms, number mismatch, empty/repeated output, language heuristic. Not implemented | S2 | 1 d | `eval/flags.py`, per-example flags in `results`, summarised in reports and `compare` |
| G7 | **No HTML report / visuals** | PRD P1; users asked for visuals | S2 | 1 d | Self-contained `report.html` / `comparison.html` (inline SVG, no JS deps) + static viewer page |
| G8 | **No ASR error analysis** | PRD: top substitutions/deletions/insertions, WER by length. Only WER/CER exist | S2 | 0.5 d | Use `jiwer` alignments; add to ASR reports |
| G9 | **No custom metrics** | PRD/[05](05_SDK_API_DESIGN.md): `(pred, ref, example) -> float` callable. Metrics are a fixed registry | S2 | 0.5 d | Registry + Python API hook; CLI `--metric module:function` |
| G10 | **Packaging never exercised** | CI installs editable only; sdist/wheel never built; name `atlasforge` unchecked on PyPI | S2 | 0.25 d | CI job: build wheel+sdist, install in a clean venv, run `atlasforge --version`; check name |
| G11 | **`doctor` cannot check gated access** | Planned per-repo access check; only token presence is checked | S2 | 0.5 d | `huggingface_hub` call per NCAIR1 repo, with the exact "accept licence" link |
| G12 | **No project config file** | [05](05_SDK_API_DESIGN.md) precedence mentions `atlasforge.toml`; none read | S3 | 0.5 d | Optional `atlasforge.toml` for defaults (backend, base_url, model) |
| G13 | **Missing OSS hygiene** | No `CODE_OF_CONDUCT.md`, issue/PR templates; security email and GitHub org are placeholders | S3 | 0.25 d | Add; humans fill placeholders |
| G14 | **Release process absent** | No tag, no release workflow, version `0.1.0.dev0` | S3 | 0.25 d | Tag `v0.1.0a1` once G3 has one real run; tag-triggered build workflow |
| G15 | **AfroBench wrapper** | PRD P1 | S3 | 1 d | Thin `lm-evaluation-harness` wrapper; lowest priority |
| G16 | **Fixed-window ASR chunking** | Words cut at seams can be misheard (documented limitation) | S3 | 1 d | Silence-aware splitting; post-NAIC |
| G17 | **Label extraction ignores negation** | `extract_label("not positive")` returns `positive` (documented) | S3 | 0.5 d | Optional strict mode: whole-answer match only |
| G19 | **CLI could not disable `repetition_penalty`** | The backend's own hint told users to, but no flag existed | S2 | 0.1 d | **DONE 2 Oct.** `--no-send-repetition-penalty` |
| G18 | **Residual uncovered lines** | `render.py` 89%, `asr/audio.py` 90%, `__main__` | S3 | 0.25 d | Cover the timeout and OSError branches |

## Priority order

1. **G1 + G2: documentation and examples.** Done in this change: an MkDocs Material site with a guided path, concepts, task guides, generated CLI reference, API reference, file formats, troubleshooting, FAQ and an honest status page. The quickstart runs offline on bundled sample data, so a reader can reach their first comparison in two minutes without any model.
2. **G5, G9, G10:** make AtlasForge pleasant to use as a library and prove it installs cleanly.
3. **G6, G8, G7:** the "understand" features: flags, ASR error analysis, then visual reports.
4. **G3, G4 (in parallel, human + hardware):** real-model evidence and testers. These gate the submission and should start immediately.
5. **G11, G12, G13, G14, G18:** hygiene and release.
6. **G15, G16, G17:** only after NAIC.

## Documentation standards adopted

Modelled on cloud and API documentation (Stripe, Anthropic, AWS):

- **Task-oriented navigation:** Get started → Concepts → Guides → Reference, plus Troubleshooting, FAQ and Status.
- **Every guide ends in a result the reader can see,** and the first one works with no model.
- **Reference is generated from the code** (CLI from the Typer app, API from docstrings), so it cannot drift.
- **Snippets are tested:** a test checks that every `atlasforge ...` command shown in the docs uses flags that really exist.
- **Honesty page:** `status.md` lists what is verified against real models and what is not.
- **Strict build:** broken links or missing pages fail CI.
