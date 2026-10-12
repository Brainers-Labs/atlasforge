# AtlasForge 0.1.0a3 release verification report

*Generated 12 October 2026, 07:51 WAT by the same black-box harness used for 0.1.0a2. This run installs **what PyPI serves now** in fresh virtual environments with no workarounds. Every status is a recorded result; the evidence is in `logs/` and `results.jsonl`.*

## Verdict

| Build | Checks run | PASS | FAIL | Not testable |
|---|---|---|---|---|
| **PyPI `brainers-atlasforge` 0.1.0a3** | 202 | 199 | 0 | 3 |

## Summary

**AtlasForge 0.1.0a3 is released and verified.** It was published to PyPI on 12 October 2026 and installed from there, in fresh virtual environments with a plain `pip install` and no workarounds, then put through the same black-box suite that found the problems in 0.1.0a2, plus new checks for each fix. **Every check that can run on this machine passes (199 of 199 distinct checks). The 3 that cannot run need an NVIDIA GPU.** Each defect found in 0.1.0a2 is fixed, and the fix is shown by a check on the released package, not only by a unit test.

## What was wrong in 0.1.0a2, and what is true now

| # | Finding in 0.1.0a2 | Severity | State in 0.1.0a3 | Proof on the release |
|---|---|---|---|---|
| F1 | `pip install "brainers-atlasforge[asr]"` failed to load on Apple Silicon macOS (scipy 1.15.3), so every speech command broke | High | **Fixed.** The `scipy<1.15` pin is released; a plain install resolves `scipy 1.14.1` | 5.01 passes on a fresh install with no workaround, and all 23 speech checks pass |
| F2 | Speech runs recorded `NCAIR1/N-ATLaS` (the LLM) as the model in `run.json` and the report | Medium | **Fixed.** A speech run names the speech model per language | 5.20 passes; 11.05 on a real mixed Hausa and Yoruba run records `NCAIR1/Hausa-ASR, NCAIR1/Yoruba-ASR` |
| F3 | The PyPI description and the README said the speech models were never run on real weights | Medium | **Fixed.** The new description states what was run and what was not, and links the verification report | 8.06 passes: none of the stale phrases appear on the PyPI page |
| F4 | `dataset validate` printed the line number twice | Low | **Fixed** | 3.04 passes on the release |
| F5 | `finetune` without a GPU hung or crashed on a Mac when `quantize: none` | Low | **Fixed.** It refuses at once with a clear message for every setting, before the other heavy imports | 9.03 and 9.03b: exit 2 in 1.0 s with the NVIDIA message; 7.05 passes |
| F6a | The macOS scipy pin made `pip install` fail on Python 3.14 | Info | **Fixed.** The pin stops at Python 3.13 | 11.04: `[finetune]` installs on macOS Python 3.14 |
| F6b | `run -` sent the pipe's trailing newline to the model | Info | **Fixed** | 11.01 |
| F6c | An all-401 abort did not mention the API key | Info | **Fixed.** The hint names `ATLASFORGE_API_KEY` | 11.02 |
| F6d | `doctor` said only "no NVIDIA GPU" on an Apple Silicon Mac | Info | **Improved.** It notes Apple Silicon and says the speech models can use the Apple GPU | 11.03 |

The first run of this release also exposed a mistake of mine, which is worth recording: a new end-to-end test needed `ffmpeg`, which CI runners lack, so CI failed on a push. It was caught before tagging, fixed by giving the test the same skip marker as its neighbours, and the suite is now run both with and without `ffmpeg` before every push.

## What was proved on the release

- **Install:** `pip install brainers-atlasforge`, `[asr]` and `[finetune]` (Python 3.14) all succeed from PyPI with no workaround; the core install pulls in no ML framework.
- **Statistics and scoring:** unchanged and still matching independent recomputation (means, win/tie/loss counts, exact McNemar p-value, bootstrap interval, tone-aware and tone-insensitive scoring, WER, CER, chrF, accuracy and macro-F1).
- **Resilience and secrets:** resume after `kill -9`, the circuit breaker, retries, timeouts, concurrency, no secret or response body in any output, hostile text escaped in the HTML reports.
- **Real models:** all four official speech models, the long-audio chunking, and the real LLM (int4 import) through `run`, `eval` and `compare`.
- **The public surface:** the live docs (42 pages, 2,488 internal links, none broken), the PyPI page, CI green on Linux, macOS and Windows for Python 3.10 and 3.13, and the quickstart notebook run from the published package.

Real speech results (10 FLEURS clips per language, indicative only) are unchanged from 0.1.0a2: Hausa WER 0.291, Yoruba 0.612 (0.457 ignoring tone marks), Igbo 0.409, English 0.085 on US-accent audio.

## What is still not tested

Real LoRA or QLoRA training and 4-bit loading (no NVIDIA GPU here), N-ATLaS at full precision, vLLM and llama.cpp serving, the AfroBench-LITE wrapper with the real harness, Nigerian-accented speech, speech over HTTP against a real endpoint (a stub was used), HTTPS/TLS, Linux and Windows locally (CI only), Python 3.11 to 3.13 locally, the Colab runtime itself, and any real-user (beta tester) evidence. The README and docs state the same limits.

## Before announcing

1. Get at least two external beta testers to run the quickstart and record their sessions (the NAIC beta-test requirement is still unmet).
2. Optionally verify fine-tuning, vLLM and full precision on a machine with an NVIDIA GPU.

## Test environment

- Machine: MacBook, macOS 27.0.1 / arm64 / 16 GB, Apple MPS or CPU only (no NVIDIA GPU)
- Python: 3.10.12 (core and speech), 3.14.7 (fine-tune install and adapter checks)
- Published package under test: `brainers-atlasforge 0.1.0a3` installed from PyPI with a plain `pip install` in a fresh environment; speech stack: librosa=0.11.0 scipy=1.14.1 torch=2.14.1 transformers=5.19.0
- Model server: Ollama ollama version is 0.40.2 serving the official `NCAIR1/N-ATLaS` weights imported as int4 (chat template restored); ffmpeg: ffmpeg version 9.0.1
- Speech models: the four official `NCAIR1` ASR repositories, run locally with the Hugging Face cache and a real token
- Test audio: real clips from Google FLEURS (CC BY 4.0), 10 per language
- All checks drive the installed `atlasforge` as a user would (command line, and the documented Python API). The stub server in `common.py` is an independent implementation, not AtlasForge's own test code.

## Formal speech results (real models, real audio)

Pooled word error rate (WER) and character error rate (CER) through `atlasforge eval --task asr`, 10 FLEURS test clips per language. **Small samples: indicative only, not a benchmark.** Lower is better.

| Language | Model | WER (tone-aware) | WER (tone-insensitive) | CER | Mean latency per clip | Failed |
|---|---|---|---|---|---|---|
| ha | NCAIR1/Hausa-ASR | 0.291 | 0.291 | 0.097 | 2805 ms | 0 |
| yo | NCAIR1/Yoruba-ASR | 0.612 | 0.457 | 0.260 | 4844 ms | 0 |
| ig | NCAIR1/Igbo-ASR | 0.409 | 0.409 | 0.159 | 4165 ms | 0 |
| en | NCAIR1/NigerianAccentedEnglish (US-accent audio, not Nigerian-accented) | 0.085 | 0.085 | 0.041 | 1631 ms | 0 |

## Every check, by group

`0.1.0a2` = the result of the same check on the previous release (worst result across its runs, so a failure that a workaround later hid is still shown). `0.1.0a3` = this run, on a fresh install from PyPI with no workaround. `-` = not run.

### 1 Install and CLI surface

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 1.01 | atlasforge --version prints the installed version | PASS | PASS | atlasforge 0.1.0a3 |
| 1.02 | `python -m atlasforge` is equivalent to the command | PASS | PASS | atlasforge 0.1.0a3 |
| 1.03 | Importing atlasforge and its CLI loads no torch/transformers/librosa/peft/trl | PASS | PASS | HEAVY [] |
| 1.04 | Core install pulls in no ML framework packages | PASS | PASS | none |
| 1.05 | Top-level help lists every documented command | PASS | PASS | missing=[] |
| 1.06 | Every command and subcommand has working --help (exit 0) | PASS | PASS | bad=[] |
| 1.07 | An unknown command fails cleanly (non-zero, no traceback) | PASS | PASS | rc=2 |
| 1.08 | doctor runs on a bare install and exits 0 when nothing hard-fails | PASS | PASS | rc=0; head: ┏━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ / ┃ Status ┃ Check          ┃ Detail                                                ┃ / ┡━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━ |
| 1.09 | doctor --json is valid JSON with name/status/detail/hint per check | PASS | PASS | checks=['config', 'disk', 'extra:asr', 'extra:finetune', 'extra:local', 'ffmpeg', 'gpu', 'hf-token', 'model-access', 'python'] |
| 1.10 | doctor masks a Hugging Face token (shows hf_****abcd, never the secret) | PASS | PASS | │ OK     │ hf-token       │ found (hf_****abcd)                                   │ |
| 1.11 | doctor --json also never contains the token | PASS | PASS | token absent from JSON |
| 1.12 | doctor reports a missing ffmpeg with the install command for the OS | PASS | PASS | ['│ WARN   │ ffmpeg         │ not found (needed for ogg/opus/m4a/mp3 audio)         │', 'ffmpeg: Windows: winget install Gyan.FFmpeg / macOS: brew install ffmpeg / Linux: apt install ffmpeg'] |

### 2 Offline workflow, reports and statistics

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 2.01 | demo writes a dataset and two finished runs with no model, token or network | PASS | PASS | files=['runs/base/results.jsonl', 'runs/base/run.json', 'runs/tuned/results.jsonl', 'runs/tuned/run.json', 'toy_qa.jsonl'] |
| 2.02 | The demo dataset has 92 questions (docs claim) | PASS | PASS | n=92 |
| 2.03 | Demo runs are labelled synthetic in their manifest so they cannot pass for real results | PASS | PASS | model=synthetic-demo/base |
| 2.04 | dataset validate accepts the demo dataset (exit 0, OK) | PASS | PASS | OK: 92 valid example(s) from 92 line(s), 0 error(s), 0 warning(s). |
| 2.05 | report writes report.md, report.json and report.html with no model | PASS | PASS | , 0 missing (failures count as wrong) / Wrote atlasforge-demo/runs/base/report.md, atlasforge-demo/runs/base/report.json and atlasforge-demo/runs/base/report.html |
| 2.06 | Every text metric is reported under both tone-aware and tone-insensitive views | PASS | PASS | views=['tone_aware', 'tone_insensitive'] |
| 2.07 | report states failed examples (they count as wrong, never dropped) | PASS | PASS | ok=90 failed=2 missing=0 total=92 |
| 2.08 | compare writes comparison.md, comparison.json and comparison.html | PASS | PASS | = numeracy (n=40): -20.0 pts / 2 slice(s) had too few examples to judge (see the report). / Wrote cmp1/comparison.md, cmp1/comparison.json and cmp1/comparison.html |
| 2.09 | Demo behaves as documented: tuned is better overall but REGRESSES on has_number (a regression is surfaced, not hidden) | PASS | PASS | overall=exact_match@tone_aware:improved delta=0.174; has_number=yes status=regressed |
| 2.10a | report.html is one self-contained page (no <script>, no external reference) | PASS | PASS | size=7726B scripts=0 external_refs=0 |
| 2.10b | comparison.html is one self-contained page (no <script>, no external reference) | PASS | PASS | size=17367B scripts=0 external_refs=0 |
| 2.11 | compare is deterministic: the same inputs and seed give byte-identical comparison.json | PASS | PASS | identical |
| 2.12 | A different --seed changes only the bootstrap interval, never the means or counts | PASS | PASS | low 0.0326 vs 0.0435 |
| 2.13a | Exact-match means and win/tie/loss counts match an independent recomputation (tone_aware) | PASS | PASS | base=0.6196 cand=0.7935 W/T/L=31/46/15 |
| 2.13b | Exact-match means and win/tie/loss counts match an independent recomputation (tone_insensitive) | PASS | PASS | base=0.7500 cand=0.8152 W/T/L=21/56/15 |
| 2.14 | Exact McNemar p-value matches an independent exact binomial computation | PASS | PASS | discordant=15/31 p_independent=2.590e-02 p_tool=2.590e-02 |
| 2.15 | The 95% bootstrap interval agrees with an independent 20,000-resample bootstrap (within 0.03) | PASS | PASS | tool=[0.033,0.315] independent=[0.033,0.315] delta=0.174 |
| 2.16 | Slices under 30 examples are reported as insufficient data, larger ones get a verdict | PASS | PASS | numeracy n=40 regressed; agri n=40 improved; greetings n=12 insufficient data |
| 2.17 | Comparing a run with itself gives delta 0, a zero-width interval and 'no clear change' | PASS | PASS | verdicts=['no clear change'] |
| 2.18 | compare refuses to pair runs made from a different dataset (hash check), with a hint | PASS | PASS | error: atlasforge-demo/runs/base was not produced from this dataset and task. /   -> Re-run it on the same dataset file, or pass the dataset it was run on. |
| 2.19 | report re-scores with different metrics (chrf++) and never modifies results.jsonl | PASS | PASS | results.jsonl unchanged=True; metrics=['chrf++', 'exact_match'] |
| 2.20 | Asking for a classification-only metric on a generation dataset fails with a clear hint | PASS | PASS | error: accuracy needs a classification dataset. /   -> Use task 'classification', or pick exact_match / chrf / wer / cer. |
| 2.21 | Pointing report at a non-run directory gives a clean error, not a traceback | PASS | PASS | error: /nonexistent is not a run directory (no run.json). /   -> Produce one with `atlasforge eval`. |

### 3 Dataset validation, normalisation and metrics

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 3.01 | A valid dataset passes (exit 0, no issues, stats present) | PASS | PASS | rc=0 n=30 |
| 3.02 | validate reports EVERY malformed line at once, each with its line number (exit 1) | PASS | PASS | error lines=[2, 3, 4, 5, 6, 7, 8] |
| 3.03 | Typos in keys are caught loudly ('refrence' is named), duplicates name the first line | PASS | PASS | duplicate id 'a' (first seen on line 1) unknown key(s): refrence provide exactly one of 'input' or 'messages' provide exactly one of 'input' or 'messages' Unsupported language 'klingon'. invalid JSON |
| 3.04 | Line numbers are not printed twice in messages (regression fixed in HEAD) | **FAIL** | PASS | first message: duplicate id 'a' (first seen on line 1) |
| 3.05 | A non-UTF-8 file is refused with a clear message, not a traceback | PASS | PASS | error: nonutf8.jsonl is not valid UTF-8 (bad byte at offset 22). /   -> Re-save the file as UTF-8. |
| 3.06 | An empty dataset is an error | PASS | PASS | codes=['no-examples'] |
| 3.07 | Train/test leakage is detected even when it differs only in case and punctuation (error, exit 1) | PASS | PASS | codes=['train-test-leakage'] |
| 3.08 | Without --against the same file raises no leakage error | PASS | PASS | ok |
| 3.09 | Duplicate content and conflicting references are warnings, not errors (exit 0) | PASS | PASS | codes=['duplicate-content', 'conflicting-references'] |
| 3.10 | Broken Unicode is flagged: replacement character, non-NFC text, control characters | PASS | PASS | codes=['replacement-character', 'control-characters', 'non-nfc'] |
| 3.11 | Stripped Yoruba diacritics (no tone marks or underdots in 25 'yo' texts) raise a warning | PASS | PASS | codes=['diacritics-missing', 'diacritics-missing'] |
| 3.12 | Properly marked Yoruba text does NOT raise the diacritics warning (no false alarm) | PASS | PASS | codes=[] |
| 3.13 | Class imbalance is flagged for classification datasets | PASS | PASS | codes=['class-imbalance'] |
| 3.14 | A single-class classification dataset is flagged | PASS | PASS | codes=['single-class'] |
| 3.15 | An empty reference in a classification dataset is an error (cannot be scored) | PASS | PASS | rc=1 codes=['surrounding-whitespace', 'empty-reference'] |
| 3.16 | An ASR dataset pointing at a missing audio file is an error naming the file | PASS | PASS | codes=['line-error', 'no-examples'] |
| 3.17 | Precomposed and combining Unicode of the same Yoruba text score identically (both views = 1) | PASS | PASS | aware=1.0 insens=1.0 |
| 3.18 | Tone marks matter in the tone-aware view (0) and are ignored in the tone-insensitive view (1); the underdot is kept | PASS | PASS | aware=0.0 insens=1.0 |
| 3.19 | Hausa hooked letters are never stripped (barna != ɓarna in BOTH views) | PASS | PASS | both 0 |
| 3.20 | Igbo underdot letters are never stripped (ulo != ụlọ in BOTH views) | PASS | PASS | both 0 |
| 3.21 | Case and punctuation are normalised away ('HELLO world!!!' == 'hello world') | PASS | PASS | match |
| 3.22 | WER is exactly 1/3 for one wrong word of three; CER = 1 wrong char of 5 (jiwer-consistent) | PASS | PASS | wer=0.3333 cer=0.2000 |
| 3.23 | An empty answer scores 0 on chrF and 1.0 on WER (all words deleted) | PASS | PASS | chrf=0.0 wer=1.0 |
| 3.24 | An answer identical after normalisation scores chrF 100 | PASS | PASS | 100.0 |
| 3.25 | Classification accuracy matches a hand calculation (loose label match; 4 of 6) | PASS | PASS | accuracy=0.6667 |
| 3.26 | Macro-F1 matches a hand calculation ((6/7 + 1/2) / 2); an unlabelled answer counts as a miss | PASS | PASS | macro_f1=0.6786 expected=0.6786 |
| 3.27 | The strict metric rejects negation ('not positive') that the loose metric accepts (strict < loose) | PASS | PASS | loose=0.667 strict=0.333 |

### 4 Evaluation engine robustness and secret handling

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 4.01 | eval runs a dataset through an OpenAI-compatible server and writes run.json, results.jsonl, report.* | PASS | PASS | rc=0 records=40 request_id=stub-1 |
| 4.02 | The API key is sent as a Bearer token (server accepted it) ... | PASS | PASS | 40 requests authenticated |
| 4.03 | ... and the key never appears in stdout, stderr, run.json, results, or any report | PASS | PASS | searched all outputs |
| 4.04 | run.json records model and settings but no URL, key or token | PASS | PASS | manifest clean |
| 4.05 | There is no --api-key flag (so a key cannot land in shell history or the process list) | PASS | PASS | flag absent |
| 4.06 | A server demanding a key, given none: reported as HTTP 401 failures and the breaker stops the run (exit 2), never a traceback | PASS | PASS | error: Stopped after 20 failures in a row. Last error: BackendHTTPError: Endpoint returned HTTP 401. /   -> The server refused the credentials |
| 4.07 | After kill -9 mid-run, re-running the same command resumes: finished examples are NOT run again, and all end up complete | PASS | PASS | finished before kill=15; resumed message=True; re-requested finished=0; final unique ids=40/40 |
| 4.08 | Re-using a run directory with a different model is refused (never mixes results) | PASS | PASS | error: guard holds a different run (changed: model). /   -> Use a new output directory, or delete this one to start over. |
| 4.09 | ... or with different generation settings | PASS | PASS | error: guard holds a different run (changed: gen_params). /   -> Use a new output directory, or delete this one to start over. |
| 4.10 | ... or with an edited dataset | PASS | PASS | error: guard holds a different run (changed: dataset_sha256). /   -> Use a new output directory, or delete this one to start over. |
| 4.11 | A dead server trips the circuit breaker: stops after 5 failures in a row (exit 2) instead of hammering all 40 examples | PASS | PASS | rc=2 requests sent=5 (of 40); msg=error: Stopped after 5 failures in a row. Last error: BackendHTTPError: Endpoint returned HTTP 500. /   -> Fix t |
| 4.12 | After fixing the server, the same command resumes and completes everything that failed | PASS | PASS | rc=0 ok=40/40 |
| 4.13 | With the breaker off and everything failing: all 40 are tried, exit code 1, and the report scores them as wrong (0%), not as missing | PASS | PASS | rc=1 requests=40 n_failed=40 mean=0.0 |
| 4.14 | 3 transient failures: exit 0 with a warning; the report counts 3 failed and the score drops by exactly those (37/40) | PASS | PASS | n_failed=3 mean=0.9250; stderr=warning: 3 of 40 example(s) failed; see partial/results.jsonl |
| 4.15 | --no-retry-errors leaves failed examples as failures on resume | PASS | PASS | n_failed=3 |
| 4.16 | By default a resume retries failed examples and fixes them (40/40) | PASS | PASS | n_failed=0 mean=1.0 |
| 4.17 | --retries masks transient 503s: every example succeeds and the server saw the extra attempts | PASS | PASS | n_failed=0 requests=42 (expected 42) |
| 4.18 | A timeout is reported ('No response within 1s') and is NOT retried (3 examples = 3 requests) | PASS | PASS | requests=3 error=['BackendTimeout: No response within 1s.'] |
| 4.19 | --concurrency 8 gives exactly the same scores as sequential | PASS | PASS | identical per-example scores for 40 examples |
| 4.20 | A server error body (which may echo the prompt) never reaches output, results or reports | PASS | PASS | canary absent everywhere |
| 4.21 | Plain http:// to a non-local host is refused before any request is made | PASS | PASS | error: Refusing plain http to a non-local host (192.0.2.10). /   -> Use https, or pass allow_insecure_http=True if you trust the network. |
| 4.22 | --allow-insecure-http lets the user opt in knowingly (fails later on connection, not on policy) | PASS | PASS | error: No response within 2s. /   -> Raise timeout=..., or check that the server is not overloaded. |
| 4.23 | run prints the answer | PASS | PASS | 'ans3' |
| 4.24 | run --json returns text, model, latency, finish_reason, request_id, usage | PASS | PASS | keys=['finish_reason', 'latency_ms', 'model', 'request_id', 'revision', 'text', 'usage'] |
| 4.25 | run forwards --system, --temperature, --max-new-tokens, --seed and the model-card repetition penalty (1.12) to the server | PASS | PASS | {"model": "m", "temperature": 0.0, "max_tokens": 77, "seed": 5, "repetition_penalty": 1.12} |
| 4.26 | --no-send-repetition-penalty omits the field for servers that reject it | PASS | PASS | field absent |
| 4.27 | run - reads the prompt from standard input | PASS | PASS | 'ans6' |
| 4.28 | Without a server URL the error says exactly what to do | PASS | PASS | error: The openai backend needs a server URL. /   -> Pass --base-url (or set ATLASFORGE_BASE_URL), e.g. http://127.0.0.1:8000/v1 |
| 4.29 | Precedence works as documented: flag beats environment beats atlasforge.toml | PASS | PASS | file-server hits=1, env-server hits=1, flag-server hits=1 |
| 4.30 | A misspelled setting in atlasforge.toml is an error naming the valid keys (never a silent no-op) | PASS | PASS | error: <test-dir>/work/eval/proj/atlasforge.toml has unknown setting(s): base-url. /   -> Valid keys under [atlasforge]: backend, base_url, device, model, quantize, retries, |
| 4.31 | doctor shows the config file it found | PASS | PASS | ['│ FAIL   │ config         │ <test-dir>/work/eval/proj/atlasforge.toml has unknown setting(s): base-url. │'] |
| 4.32 | A custom metric named module:function runs through eval and is reported under both views | PASS | PASS | mean=1.0; views=['tone_aware', 'tone_insensitive'] |

### 5 Speech models on real audio

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 5.01 | [asr] extra: `transcribe --backend local` loads the official Hausa model and returns text on a fresh install | **FAIL** | PASS | an kwatanta faretin gine-ginen da ke yin sararin samaniya, hunkunk, da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan victoria habu. |
| 5.02 | Hausa transcription of a real clip is close to the reference (WER < 0.35, tone-aware view) | PASS | PASS | WER=0.167 / ref: An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbor. / hyp: an kwatanta faretin gine-ginen da ke yin sararin samaniya, hunk |
| 5.03 | Any format ffmpeg reads works, including WhatsApp-style opus/ogg, mp3, m4a and flac (result within WER 0.35 of the wav result) | PASS | PASS | WER vs wav: {'ogg': 0.0, 'mp3': 0.0, 'm4a': 0.0, 'flac': 0.0} |
| 5.04 | --lang selects a different model: Hausa audio through the Yoruba model is clearly worse than through the Hausa model | PASS | PASS | WER via ha model=0.167; via yo model=1.125 |
| 5.05 | Several files in one command: the missing one is reported on stderr (exit 1) but the others are still transcribed and written to --out | PASS | PASS | rc=1 written=['audio/ha_01.wav', 'audio/ha_02.wav']; stderr=rce cleanup anyway. / error: audio/does_not_exist.wav: Audio file not found: audio/does_not_exist.wav /   -> Check the path. |
| 5.06 | --json records carry file, lang, text, latency_ms and chunks | PASS | PASS | keys=['chunks', 'file', 'lang', 'latency_ms', 'text'] |
| 5.07 | A file that is not audio gives a clean error (exit 1, no traceback) | PASS | PASS | error: fake.wav: ffmpeg could not decode the audio: [in#0 @ 0x7a7f014000] Error opening input: Invalid data found when processing input / Error opening input file |
| 5.08 | An unsupported language is a clear error naming the valid ones | PASS | PASS | error: Unsupported language 'xx'. /   -> Use one of: ha, yo, ig, en (or hausa, yoruba, igbo, english). |
| 5.09 | A 79 s recording (limit is 30 s) is split into overlapping windows automatically and merged into one transcript | PASS | PASS | windows=3 boundaries=[(0.0, 28.0), (26.0, 54.0), (52.0, 79.2)] WER vs concatenated reference=0.378 |
| 5.10 | Overlap merging does not leave duplicated passages (no repeated 4-word sequence at the seams) | PASS | PASS | repeated 4-grams=0 |
| 5.11 | --silence-aware moves window cuts to a pause: each interior boundary falls inside/near a silent gap | PASS | PASS | boundaries=[26.51, 53.63] near_silence=[True, True] WER=0.396 |
| 5.12en | eval --task asr on 10 real FLEURS 'en' clips: pooled WER equals an independent jiwer computation (both tone views), nothing failed | PASS | PASS | WER tone-aware=0.085 tone-insens=0.085 CER=0.041 mean latency=1631 ms n_failed=0 |
| 5.12ha | eval --task asr on 10 real FLEURS 'ha' clips: pooled WER equals an independent jiwer computation (both tone views), nothing failed | PASS | PASS | WER tone-aware=0.291 tone-insens=0.291 CER=0.097 mean latency=2805 ms n_failed=0 |
| 5.12ig | eval --task asr on 10 real FLEURS 'ig' clips: pooled WER equals an independent jiwer computation (both tone views), nothing failed | PASS | PASS | WER tone-aware=0.409 tone-insens=0.409 CER=0.159 mean latency=4165 ms n_failed=0 |
| 5.12yo | eval --task asr on 10 real FLEURS 'yo' clips: pooled WER equals an independent jiwer computation (both tone views), nothing failed | PASS | PASS | WER tone-aware=0.612 tone-insens=0.457 CER=0.260 mean latency=4844 ms n_failed=0 |
| 5.13 | The ASR report includes error analysis (substitutions, deletions, insertions) | PASS | PASS | ['10 utterance(s) aligned: 154 correct words, 46 substituted, 6 dropped, 8 inserted (pooled word error rate 29.1%).'] |
| 5.14 | ASR report.html is self-contained | PASS | PASS | 13721B |
| 5.15 | Clips longer than 30 s inside an eval dataset (3 Igbo clips of 31 to 34 s) are chunked automatically and transcribed | PASS | PASS | long clips=['ig-05', 'ig-07', 'ig-08'] |
| 5.16 | Re-running a finished ASR eval resumes instantly and does not reload/re-transcribe | PASS | PASS | 1.0s; ['Resumed: 10 example(s) were already finished.'] |
| 5.17 | --device cpu works and gives (nearly) the same WER as the default device | PASS | PASS | cpu WER=0.291 default=0.291 |
| 5.18 | compare works on real ASR runs; lower-is-better WER is oriented correctly and the two devices show no regression | PASS | PASS | verdicts=[('wer@tone_aware', 'no clear change'), ('cer@tone_aware', 'no clear change'), ('wer@tone_insensitive', 'no clear change'), ('cer@tone_insensitive', 'no clear change')] |
| 5.19 | doctor sees the cached Hugging Face login and the installed asr extra, and checks gated access to each NCAIR1 model | PASS | PASS | hf-token=cached login found; model-access=ok licence accepted on all 5 models |
| 5.20 | The run manifest/report name the speech model that actually transcribed (NCAIR1/Hausa-ASR), not the LLM default | **FAIL** | PASS | manifest model='NCAIR1/Hausa-ASR'; report header: ['- **Model:** NCAIR1/Hausa-ASR'] |

### 6 Real LLM through the openai backend

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 6.01 | Precondition: the official N-ATLaS weights are imported in Ollama (int4) and served | PASS | PASS | models=['natlas-local-chat:latest', 'natlas-local-q4:latest', 'qwen3.6:latest'] |
| 6.02 | The served model is an 8.0B Llama with 131072 context, int4 (matches config.json, not the card's 8,092) | PASS | PASS | Model architecture llama parameters 8.0B context length 131072 embedding length 4096 quantization int4 requires 0.19.0 Capabilities completion Parameters temperature 0.1 num_predict 1000 repeat_penalt |
| 6.03en | run: the real model answers a en prompt (non-empty text, token usage and latency reported) | PASS | PASS | 'The capital of Nigeria is Abuja.' tokens={'prompt_tokens': 46, 'completion_tokens': 8} 4810ms |
| 6.03ha | run: the real model answers a ha prompt (non-empty text, token usage and latency reported) | PASS | PASS | 'Kwam a Kano, jihar Kano a arewa Najeriya.' tokens={'prompt_tokens': 54, 'completion_tokens': 18} 901ms |
| 6.03ig | run: the real model answers a ig prompt (non-empty text, token usage and latency reported) | PASS | PASS | "Naịjirịa bụ obodo a ma ama maka nnukwu onyinye omenala ya, ebe ndị si n'ebe dị iche iche na-akwaga kemgbe ọtụt" tokens={'prompt_tokens': 61, 'completion_tokens': 98} 3648ms |
| 6.03yo | run: the real model answers a yo prompt (non-empty text, token usage and latency reported) | PASS | PASS | 'Ìlú Eko sábà máa ń fani lọ́kàn mọ́ni, ó sì jẹ́ ilé iṣẹ́ àbójútó olùrànlọ́wọ́ ní gbogbo orílẹ̀-èdè.' tokens={'prompt_tokens': 59, 'completion_tokens': 65} 2508ms |
| 6.04 | Real model, --system and --temperature 0: answers 'Abuja' | PASS | PASS | Abuja |
| 6.05 | --no-send-repetition-penalty works against the real server too | PASS | PASS | Abuja |
| 6.06 | The 60-question real dataset validates | PASS | PASS | OK: 60 valid example(s) from 60 line(s), 0 error(s), 0 warning(s). |
| 6.07 | eval of 60 real questions on the real model completes with no failures, reports latency, and a custom metric | PASS | PASS | n_failed=0 contains_ref=0.9833333333333333 mean latency=322ms p95=390ms |
| 6.08 | Every record has a prediction and a request id/latency; examples are the model's real text, not echoes | PASS | PASS | 60 records; sample: ['Abuja', 'Accra', 'Nairobi'] |
| 6.09 | run.json records the served model name and the settings used (temperature 0, max_new_tokens 40) | PASS | PASS | {"model": "natlas-local-chat", "backend": "openai", "gen_params": {"temperature": 0.0, "repetition_penalty": 1.12, "max_new_tokens": 40, "top_p": null, "seed": null}} |
| 6.10 | A second real run (same weights, without the chat template) completes | PASS | PASS | ailures count as wrong) / Wrote raw/report.md, raw/report.json and raw/report.html |
| 6.11 | compare on two REAL runs: per-metric verdicts, win/tie/loss, and domain slices with n=30 each get real verdicts (not 'insufficient data') | PASS | PASS | exact_match@tone_aware:improved d=+0.967; chrf@tone_aware:improved d=+86.958; contains_ref@tone_aware:improved d=+0.100 / capitals n=30 improved; arithmetic n=30 improved |
| 6.12 | A custom metric gets the full paired statistics in compare (interval + wins/ties/losses) | PASS | PASS | (0.8833333333333333, 0.9833333333333333, 0.016666666666666666, 0.2) |
| 6.13 | Classification task on the real model reports accuracy and macro-F1 (12 reviews) | PASS | PASS | accuracy=1.0 macro_f1=1.0 |
| 6.14 | --concurrency 2 against the real server completes with no failures | PASS | PASS | n_failed=0 wall=20s |

### 7 Fine-tune plan, model cards, Python API and packaging

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 7.01 | finetune --dry-run validates data and prints the plan with no GPU, writes nothing, exit 0 | PASS | PASS | Base model: NCAIR1/N-ATLaS Training file: train.jsonl (30 examples, fingerprint 31ed4abc2ed11a7d) Held-out file: test.jsonl Median length: 6 words per example Method: QLoRA (4-bit), r=16, alpha=32 Sch |
| 7.02 | Train/test leakage is refused before any training (even when it differs only in case/punctuation) | PASS | PASS | error: The training data has 1 error(s): 1 example(s) also appear in test_leak.jsonl (0 identical, 1 differing only in case, punctuation or tone marks): t3 /   -> Run `atlasforge dataset validate train. |
| 7.03 | A tiny training set (5 examples) is refused: 'at least 20 are required' | PASS | PASS | error: Only 5 training example(s); at least 20 are required. /   -> Fine-tuning on a handful of examples mostly teaches the model to repeat them. Add data, or lower min_examples if you really mean it. |
| 7.04 | Config typos and every invalid value are reported together (unknown key, bad learning_rate, bad quantize) | PASS | PASS | error: Unknown setting(s) in ft_typo.json: lora_rank. -> Known settings: base_model, batch_size, eval_file, grad_accum, learning_rate, logging_steps, lora_alpha, lora_dropout, lora_r, max_seq_len, min_examples, num_epochs, output_ |
| 7.05 | Real training without the needed hardware/extras fails early with a clear message and no partial output (torch missing -> install hint; torch present but no NVIDIA GPU -> says it needs an NVIDIA GPU) | PASS | PASS | torch_installed=False; error: Fine-tuning needs 'torch', which is not installed. /   -> pip install "brainers-atlasforge[finetune]" |
| 7.06 | An actual QLoRA training run and evaluating the resulting adapter (--adapter) | n/a | n/a | Needs an NVIDIA GPU with CUDA; this Mac has none. The documentation says the same: the training run has never been executed. |
| 7.07 | card adds the required 'Powered by Awarri' suffix to the model name when missing | PASS | PASS | title line: '# Hausa Agri Helper - Powered by Awarri' |
| 7.08 | The card carries the licence obligations: attribution to Awarri Technologies, 1,000 user cap, no redistribution of base weights | PASS | PASS | has Awarri=True cap=True |
| 7.09 | The card includes the evaluation numbers and states regressions from comparison.json (does not hide them) | PASS | PASS | mentions regression=True |
| 7.10 | The card includes the stated training-data licence from the file given with @ | PASS | PASS | found 'CC BY 4.0' |
| 7.11 | card refuses to run without a stated training-data source and licence (required option) | PASS | PASS | ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯ |
| 7.12 | bench afrobench without the harness fails with a clear instruction, not a traceback | PASS | PASS | error: lm-evaluation-harness is not installed, so there are no tasks to run. -> Install it with: pip install "brainers-atlasforge[bench]" |
| 7.13 | Running AfroBench-LITE against N-ATLaS through lm-evaluation-harness | n/a | n/a | Needs the bench extra, a GPU and the gated weights loaded in-process; not possible on this machine. Docs mark it unverified. |
| 7.14 | Python API: atlasforge.evaluate() with your own Backend object runs, scores and writes run.json/results/report.md/json/html | PASS | PASS | EM 1.0 files ['report.html', 'report.json', 'report.md', 'results.jsonl', 'run.json'] API ['Evaluation', 'compare_runs', 'evaluate', 'score_finished_run', 'write_comparison', 'write_reports'] RESCORED 12 CMP ['no clear change', 'n |
| 7.15 | A Backend object you pass in stays yours: AtlasForge does not close it | PASS | PASS | close() not called |
| 7.16 | score_finished_run and compare_runs work from the Python API | PASS | PASS | uate', 'score_finished_run', 'write_comparison', 'write_reports'] RESCORED 12 CMP ['no clear change', 'no clear change'] |
| 7.17 | The distribution contains no model weights or audio (licence: nothing is redistributed), and is small | PASS | PASS | 52 files, 134 KB; weights/audio=[]; files >500KB=[] |
| 7.18 | The wheel ships type information (py.typed) and the licence | PASS | PASS | py.typed and LICENSE present |
| 7.19 | Only the five official NCAIR1 model ids appear in the shipped code (no other foundation model) | PASS | PASS | ids=['NCAIR1/Hausa-ASR', 'NCAIR1/Igbo-ASR', 'NCAIR1/N-ATLaS', 'NCAIR1/NigerianAccentedEnglish', 'NCAIR1/Yoruba-ASR'] |
| 7.20 | No LLM-as-judge and no third-party model API is referenced in the shipped code | PASS | PASS | found=[] |

### 8 Public site, PyPI, CI and notebook

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 8.01 | The live documentation site is up and every page in its sitemap loads (HTTP 200), logged out | PASS | PASS | 42 pages; failing=[] |
| 8.02 | No broken internal links on the live site | PASS | PASS | 2488 internal links checked; broken=[] |
| 8.03 | The live home page and navigation are intact (Quickstart, Reference, Troubleshooting present) | PASS | PASS | 48120 bytes |
| 8.04 | The live status page and integration page reflect what was proved: ASR models and Ollama serving 'Run on real weights', context length 131072 recorded | PASS | PASS | ASR row: 'Official ASR models Run on real weights Close transcriptions of single Hausa, Yoruba and I' / Ollama row: 'Serving N-ATLaS with Ollama (int4) Run on real weights Imported from the officia' / 131072 on integration page: T |
| 8.05 | PyPI: name, version, Python range, licence and project URLs are correct | PASS | PASS | version=0.1.0a3 python=>=3.10 license=Apache-2.0 urls={'Homepage': 'https://github.com/Brainers-Labs/atlasforge', 'Issues': 'https://github.com/Brainers-Labs/atlasforge/issues'} |
| 8.06 | PyPI project description is accurate now that the speech models and LLM were run on real weights | **FAIL** | PASS | stale phrases found in the PyPI description: []. (A published description can only change with a new release.) |
| 8.07 | The PyPI project page loads | PASS | PASS | HTTP 200 |
| 8.08 | The GitHub repository is public (loads logged out) | PASS | PASS | HTTP 200 |
| 8.09-CHANGELOG.md | Repository has CHANGELOG.md | PASS | PASS | HTTP 200 |
| 8.09-CONTRIBUTING.md | Repository has CONTRIBUTING.md | PASS | PASS | HTTP 200 |
| 8.09-LICENSE | Repository has LICENSE | PASS | PASS | HTTP 200 |
| 8.09-README.md | Repository has README.md | PASS | PASS | HTTP 200 |
| 8.09-SECURITY.md | Repository has SECURITY.md | PASS | PASS | HTTP 200 |
| 8.10 | SECURITY.md has a real private reporting route (advisory form or email) and no leftover TODO placeholder | PASS | PASS | # Security Policy ## Reporting a vulnerability Report privately through GitHub's [security advisory form](https://github.com/Brainers-Labs/atlasforge/security/advisories/new) with details and reproduc |
| 8.11 | pyproject.toml has no placeholder organisation text | PASS | PASS | no placeholder markers |
| 8.12 | Latest CI runs on main are green (CI, docs, release workflows where present) | PASS | PASS | {'Publish to PyPI': 'success', 'Docs': 'success', 'CI': 'success'} |
| 8.13 | CI passes on Linux, macOS and Windows with Python 3.10 and 3.13 (the support claim) | PASS | PASS | 10 jobs: build sdist + wheel and install them clean=success; core install stays light=success; secrets + dependency audit=success; test (ubuntu-latest, py3.10)=success; test (macos-latest, py3.10)=success; docs build (strict)=succ |
| 8.14 | The Colab link resolves | PASS | PASS | HTTP 200 |
| 8.15 | The notebook exists in the public repository (so the Colab link has something to open) | PASS | PASS | HTTP 200, 7110 bytes |
| 8.16 | The quickstart notebook runs top to bottom in a clean environment using the PUBLISHED package, with no errors | PASS | PASS | CELLS_OK 6 / HAS_REGRESSION True True |

### 9 Fine-tune and local LLM plumbing (tiny stand-in model)

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 9.01 | The package installs and starts on Python 3.14.7 (outside the documented 3.10 to 3.13: forward-compatibility data point) | - | PASS | atlasforge 0.1.0a3 on Python 3.14.7 |
| 9.02 | finetune --dry-run plans a LoRA run on a non-default base model | - | PASS | Base model: HuggingFaceTB/SmolLM2-135M-Instruct Training file: train.jsonl (40 examples, fingerprint ffaaed1aad367a0a) Held-out file: (none given: leakage was NOT checked) Median length: 7 words per example Method: LoRA |
| 9.03 | finetune without an NVIDIA GPU is refused at once with a clear message, for quantize=none as well as 4bit (earlier versions hung or crashed on a Mac) | - | PASS | exit 2 in 1.0s; error: LoRA training needs an NVIDIA GPU, and none was found. /   -> Use a GPU machine (Colab, a cloud GPU, a university cluster). Training an 8B model on a Mac or CPU is not supported; on Apple Silicon |
| 9.03b | No crash or hang when fine-tuning is attempted without a GPU | - | PASS | exit 2 in 1.0s |
| 9.03t | A real LoRA training run (needs CUDA; the docs say it will not work on a Mac) | - | n/a | No NVIDIA GPU here. Attempting it on Apple's GPU hangs or crashes inside torch even with plain transformers (see the 0.1.0a2 report), so training remains unverified. |
| 9.04 | Precondition: a real PEFT LoRA adapter (built with plain peft, not by AtlasForge) exists to evaluate | - | PASS | files=['README.md', 'adapter_config.json', 'adapter_model.safetensors', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json'] |
| 9.06 | eval --backend local runs the real tokenizer chat template and generation, and --adapter loads a LoRA adapter | - | PASS | base rc=0, tuned rc=0;  deprecated! Use `dtype` instead! /  / Loading weights:   0%/          / 0/272 [00:00<?, ?it/s] / Loading weights: 100%/██████████/ 272/272 [00:00<00:00, 6785.12it/s] |
| 9.07 | An adapter is part of the model's identity in the manifest (a tuned run can never be mistaken for the base run) | - | PASS | base model='HuggingFaceTB/SmolLM2-135M-Instruct'  tuned model='HuggingFaceTB/SmolLM2-135M-Instruct+adapter' |
| 9.08 | The adapter is actually APPLIED: with a non-zero LoRA the tuned model's answers differ from the base model's | - | PASS | 40 of 40 answers differ; e.g. base='Here is the code to generate an item with the given name' tuned='```python\ndef get_item(name):\n    return' |
| 9.09 | compare works between a base run and an adapter run on real local-backend output (with slices) | - | PASS | verdicts=[('exact_match@tone_aware', 'no clear change'), ('chrf@tone_aware', 'regressed')] |
| 9.10 | local backend: greedy decoding is deterministic, max-new-tokens is honoured, token usage and finish_reason are reported | - | PASS | 'One of the most popular fruits in the world is the banana' tokens={'prompt_tokens': 34, 'completion_tokens': 12} finish=length |
| 9.11 | A model whose tokenizer has no chat template is refused with a clear error (never a guessed prompt format) | - | PASS | [transformers] `torch_dtype` is deprecated! Use `dtype` instead! /  / Loading weights:   0%/          / 0/29 [00:00<?, ?it/s] / Loading weights: 100%/██████████/ 29/29 [00:00<00:00, 24622.43it/s] / [transform |
| 9.12 | --adapter with the openai backend is refused with an explanation | - | PASS | error: --adapter only works with the local backend. /   -> With a server, load the adapter there and pass its served name as --model. |

### 10 Extra features and edge cases

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 10.01 | transcribe over HTTP (--backend openai): posts the audio as WAV with language and model to /audio/transcriptions and prints the text | PASS | PASS | out='stub transcript 1' request={'path': '/v1/audio/transcriptions', 'len': 607234, 'ctype': 'multipart/form-data; boundary=6f472ef7df10dba20a8bb5b9ea7c89a3', 'has_riff': True, 'lang': True, 'model': True} |
| 10.02 | A 79 s recording over HTTP is split client-side into 3 windows (3 requests, each a WAV under 30 s) and merged | PASS | PASS | requests=3 sizes=[896514, 896514, 870914] chunks=3 |
| 10.03 | Hostile text in a model name, a model answer, or a slice value is escaped in report.html and comparison.html (no injected tags) | PASS | PASS | unescaped in: []; escaped marker present=True |
| 10.04 | Scoring a run that was killed halfway counts the unfinished examples as MISSING and wrong (not silently ignored): score = finished/40 | PASS | PASS | finished=12 n_missing=28 exact_match=0.300 (expected 0.300) |
| 10.05 | atlasforge.toml is found from any subdirectory below it (nearest file at or above the working directory) | PASS | PASS | server hits=1 model=from-parent-file |
| 10.06 | A config file in a SIBLING/other directory is not picked up (nothing read from elsewhere or from the home directory) | PASS | PASS | error: The openai backend needs a server URL. /   -> Pass --base-url (or set ATLASFORGE_BASE_URL), e.g. http://127.0.0.1:8 |
| 10.07 | doctor exits 1 when a check FAILS (a broken atlasforge.toml), and says which line/why, rather than crashing | PASS | PASS | │ FAIL   │ config         │ <test-dir>/work/extras_rel/badcfg/atlasforge.toml is not valid TOML: Expected '=' after a key in a key/value pair (at line 2, column 6) │ |
| 10.08 | -V is the short form of --version | PASS | PASS | atlasforge 0.1.0a3 |
| 10.09 | Running `demo` twice in the same place is safe (no traceback, data still valid afterwards) | PASS | PASS | second run rc=2: error: atlasforge-demo already exists and is not empty. /   -> Choose a new folder, e.g. `atlasforge demo my-demo`. |
| 10.10 | A missing dataset path is a clean error, not a traceback | PASS | PASS | error: cannot read /does/not/exist.jsonl: No such file or directory /   -> Check the path. |

### 11 Fixes from the 0.1.0a2 verification, on the release

| ID | What was claimed or checked | 0.1.0a2 | 0.1.0a3 | Evidence |
|---|---|---|---|---|
| 11.01 | `echo hello / atlasforge run -` sends 'hello', not 'hello' plus a newline (was a minor bug in 0.1.0a2); inner newlines are kept | - | PASS | sent=['hello', 'a\nb\n'] |
| 11.02 | When every request is refused with HTTP 401, the abort hint tells the user to set ATLASFORGE_API_KEY (generic advice before) | - | PASS | error: Stopped after 3 failures in a row. Last error: BackendHTTPError: Endpoint returned HTTP 401. /   -> The server refused the credentials. Set ATLASFORGE_API_KEY to a valid key, then re-run with the same --out to resume; every |
| 11.03 | doctor recognises Apple Silicon: still warns about NVIDIA, but says the speech models can use the Apple GPU and the LLM needs a quantised server | - | PASS | no NVIDIA GPU detected (Apple Silicon found) / The speech models run on the Apple GPU (MPS) or the CPU. The 8B LLM does not fit in fp16 on a 16 GB Mac: serve a quantised copy (Ollama or l |
| 11.04 | `pip install "brainers-atlasforge[finetune]"` on macOS with Python 3.14 now succeeds (the scipy<1.15 pin used to force a failing source build) | - | PASS | installed 0.1.0a3 with peft/trl/datasets on Python 3.14 |
| 11.05 | A mixed Hausa and Yoruba speech run records BOTH speech models (and not the LLM) in run.json and the report header | - | PASS | model='NCAIR1/Hausa-ASR, NCAIR1/Yoruba-ASR' |

## How to reproduce

```bash
cd evidence/release-verification-0.1.0a3-2026-10-12/harness  # then: ./run_release.sh  (installs from PyPI in fresh venvs and runs everything)
#
# core: PyPI build
venv-pypi/bin/python t01_core.py; venv-pypi/bin/python t02_offline.py   # and t03, t04, t07, t08, t10
# speech + real LLM
AF_VENV=$PWD/venv-pypi-asr AF_LABEL=pypi-asr-scipyfix venv-pypi-asr/bin/python t05_asr.py
venv-pypi/bin/python t06_llm.py
# unreleased HEAD: same scripts with AF_VENV=$PWD/venv-head AF_LABEL=head
python report.py
```

