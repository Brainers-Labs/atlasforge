# Release verification, AtlasForge 0.1.0a3, 12 October 2026

The same black-box suite that tested 0.1.0a2 ([report](../release-verification-2026-10-09/REPORT.md)), re-run against **what PyPI serves for 0.1.0a3**, installed with a plain `pip install` into fresh virtual environments with no workarounds. It adds a check for each fix.

**Start with [`REPORT.md`](REPORT.md).** It has the verdict, a table of every finding from the 0.1.0a2 report with its state now and the check that proves it, the real speech results, what is still untested, and a table of every check showing its result on 0.1.0a2 beside 0.1.0a3.

| File | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | the full report |
| `results.jsonl` | one line per check per environment: id, claim, result, evidence |
| `speech/asr_summary_rel-asr.json` | the formal word and character error rates per language |
| `logs/` | the transcript (command, exit code, stdout, stderr) behind each result, with local paths removed |
| `harness/` | the scripts that produced it; `run_release.sh` builds the three environments from PyPI and runs everything |

## Re-running it

```bash
cd evidence/release-verification-0.1.0a3-2026-10-12/harness
../../../.venv/bin/python prep_audio.py     # downloads the FLEURS test clips (needs a Hugging Face token)
./run_release.sh                            # fresh venvs from PyPI, then every group
python3 report_rel.py                       # rebuilds the report from results.jsonl
```

It needs ffmpeg, a Hugging Face token with the NCAIR1 licences accepted, `gh` for the CI check, and Ollama serving the model for the LLM group. Python 3.14 (Homebrew) is used for one group. `report_rel.py` also reads the 0.1.0a2 results from an `archive_0.1.0a2/results.jsonl` next to it; copy `../../release-verification-2026-10-09/results.jsonl` there.

## Not a benchmark

Speech error rates come from 10 clips per language, and the LLM checks from a quantised copy of the model. They show the tool measures correctly, not how good N-ATLaS is. Real training, vLLM, full precision and Nigerian-accented speech were not tested; see the report.
