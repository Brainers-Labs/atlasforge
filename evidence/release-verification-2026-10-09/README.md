# Release verification, 9 October 2026

A black-box test of AtlasForge before public release. It installs the **published** package from PyPI in clean virtual environments (and, separately, the unreleased repository build), then drives it as a user would: the command line, and the documented Python API. Real model weights and real audio are used wherever this machine can run them.

**Start with [`REPORT.md`](REPORT.md).** It has the verdict, the findings ranked by severity, the real speech and LLM results, what was not tested, and a table of every one of the 197 checks with its result on each build.

| File | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | the full report |
| `results.jsonl` | one line per check per build: id, claim, result, evidence |
| `speech/asr_summary_*.json` | the formal word and character error rates per language |
| `logs/` | the full transcript (command, exit code, stdout, stderr) behind each result, with local paths removed |
| `harness/` | the scripts that produced everything, so it can be re-run |

## Environment

A MacBook with an Apple M1 and 16 GB of memory, macOS 27.0, Python 3.10.12 (and 3.14.7 for one group). No NVIDIA GPU. The N-ATLaS LLM was served as an int4 import through Ollama; the four speech models ran locally with Hugging Face `transformers`. Test audio is 10 clips per language from Google FLEURS (CC BY 4.0), which the harness downloads itself and which is not committed here.

## Re-running it

The scripts expect this layout (paths are relative to the folder holding the scripts) and one virtual environment per build under test:

```bash
cd evidence/release-verification-2026-10-09/harness
python3 -m venv venv-pypi       && venv-pypi/bin/pip install brainers-atlasforge
python3 -m venv venv-pypi-asr   && venv-pypi-asr/bin/pip install "brainers-atlasforge[asr]"
python3 -m venv venv-head       && venv-head/bin/pip install "../../.."        # the repository build

venv-pypi/bin/python t01_core.py            # install surface, doctor, secret masking
venv-pypi/bin/python t02_offline.py         # demo, report, compare, independent recomputation of the statistics
venv-pypi/bin/python t03_data.py            # dataset validation, normalisation, metrics against known answers
venv-pypi/bin/python t04_eval.py            # resume, circuit breaker, retries, timeouts, secrets (stub server)
venv-pypi/bin/python t07_toolkit.py         # fine-tune dry run, model cards, Python API, wheel contents
venv-pypi/bin/python t08_public.py          # live docs, PyPI, repository, CI, the notebook (needs `gh`)
venv-pypi/bin/python t10_extras.py          # HTTP speech endpoint, HTML escaping, partial runs, config discovery

# Needs ffmpeg, a Hugging Face token with the NCAIR1 licences accepted, and (for t06) Ollama serving the model:
../../../.venv/bin/python prep_audio.py     # any environment with pyarrow, soundfile, huggingface_hub
AF_VENV=$PWD/venv-pypi-asr AF_LABEL=pypi-asr-scipyfix venv-pypi-asr/bin/python t05_asr.py
venv-pypi/bin/python t06_llm.py

# Run the same scripts against the repository build by setting AF_VENV=$PWD/venv-head AF_LABEL=head
python3 report.py                           # rebuilds the report from results.jsonl
```

The stub server in `common.py` is an independent implementation of just enough of the OpenAI protocol to test failure handling. It does not import AtlasForge's own test code.

## Not a benchmark

Speech error rates come from 10 clips per language and the LLM checks from a quantised copy of the model. They show the tool measures correctly, not how good N-ATLaS is.
