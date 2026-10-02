# Use the Python API

The command line is a thin layer over a Python library. Anything the CLI does, you can do from your own code, which is useful for notebooks, CI checks and custom pipelines.

!!! info "API stability"
    AtlasForge is pre-alpha (`0.1.0.dev0`). Names and signatures may change before a release. Importing `atlasforge` loads no ML framework; `torch` and `transformers` are imported only when you use the `local` backend.

Every example below was run against a stub server before being included. The full signatures are in the [API reference](../reference/python-api.md).

## Talk to a model

```python
from atlasforge.backends.openai import OpenAIBackend
from atlasforge.types import GenParams

with OpenAIBackend(base_url="http://127.0.0.1:11434/v1", model="natlas-local-chat") as backend:
    result = backend.generate(
        [{"role": "user", "content": "What is the capital of Nigeria?"}],
        GenParams(temperature=0.0, max_new_tokens=200),
    )
    print(result.text)           # the answer
    print(result.latency_ms)     # milliseconds
    print(result.finish_reason)  # "stop" or "length"
    print(result.usage)          # Usage(prompt_tokens=..., completion_tokens=...) if reported
    print(backend.info())        # BackendInfo(backend='openai', model=..., capabilities=...)
```

`generate` takes a list of `{"role", "content"}` messages (roles: `system`, `user`, `assistant`). Passing `params=None` uses the model-card defaults: `temperature=0.1`, `repetition_penalty=1.12`, `max_new_tokens=1000`.

`OpenAIBackend` options worth knowing:

| Argument | Default | Meaning |
|---|---|---|
| `base_url` | required | the server URL, normally ending in `/v1` |
| `model` | required | the model name the server serves |
| `api_key` / `api_key_env` | env `ATLASFORGE_API_KEY` | the bearer token, or the name of the variable to read it from |
| `asr_model` | same as `model` | the speech model name for `/audio/transcriptions` |
| `asr` | `True` | set `False` to refuse `transcribe` calls |
| `timeout` | `120.0` | seconds per request |
| `max_retries` | `2` | retries on `429`, `500`, `502`, `503`, `504` and connection errors |
| `send_repetition_penalty` | `True` | set `False` for servers that reject that field |
| `allow_insecure_http` | `False` | permit plain `http://` to a non-local host |

To build a backend from CLI-style options instead, use `build_backend("openai", base_url=..., model=...)` or `build_backend("local", model=...)` from `atlasforge.backends.factory`.

## Run and score a dataset

```python
from atlasforge.backends.openai import OpenAIBackend
from atlasforge.eval.dataset import load_dataset
from atlasforge.eval.runner import RunConfig, read_results, run
from atlasforge.eval.score import score_run, write_report_json

dataset = load_dataset("data.jsonl", "generation")   # raises DatasetError with a line number

with OpenAIBackend(base_url="http://127.0.0.1:8000/v1", model="NCAIR1/N-ATLaS") as backend:
    summary = run(backend, dataset, "runs/base", config=RunConfig(concurrency=4))

print(summary)   # RunSummary(out_dir=..., total=40, skipped=0, ok=40, failed=0)

report = score_run(dataset, read_results("runs/base/results.jsonl"), metrics=["exact_match", "chrf"])
print(report.metric("exact_match@tone_aware").mean)     # 0.5
print(report.n_failed)                                  # failures count as wrong
write_report_json(report, "runs/base/report.json")
```

`run` is **resumable**: calling it again with the same output directory skips finished examples and checks the manifest, exactly like `atlasforge eval`. `RunConfig` holds `gen_params`, `lang` (default ASR language), `concurrency`, `retry_errors` and `max_consecutive_failures`.

`run` writes `run.json` and `results.jsonl` only. Scoring is a separate call, so you can re-score with different metrics as often as you like. `report.per_example` maps each example id to its scores, keyed like `exact_match@tone_aware`.

You can also pass an `on_result` callback to `run` to watch records as they are written (the CLI uses it for the progress bar).

## Compare two runs

```python
from atlasforge.compare import compare_runs, to_markdown

comparison = compare_runs(
    dataset, "runs/base", "runs/tuned",
    slice_fields=["lang", "length", "domain"],   # built-ins plus any meta key
    n_boot=1000, seed=0,
)
for metric in comparison.metrics:
    print(metric.key, metric.verdict, metric.delta, (metric.low, metric.high))
print(comparison.regressed_slices)               # tuple of SliceResult
open("cmp.md", "w", encoding="utf-8").write(to_markdown(comparison))
```

`compare_runs` verifies that both runs were produced from this dataset (by hash) before pairing. If you have already scored the runs, `compare_scores(dataset, base_report, candidate_report)` skips the re-scoring. The statistics are also available on their own: `paired_bootstrap(base_values, candidate_values)` and `mcnemar_exact(base_values, candidate_values)` from `atlasforge.compare`.

## Validate a dataset

```python
from atlasforge.eval.validate import validate_dataset

report = validate_dataset("test.jsonl", "generation", against="train.jsonl")
if not report.ok:
    for issue in report.errors:
        print(issue.code, issue.line, issue.message)
print(report.stats["langs"])
```

## Normalise text

```python
from atlasforge.eval.normalize import normalize, tone_aware, tone_insensitive

normalize("Kọ́pà àtijọ́!", tone_insensitive("yo"))   # 'kọpa atijọ'
normalize("Kọ́pà àtijọ́!", tone_aware("yo"))         # 'kọ́pà àtijọ́'
```

## Transcribe long audio

```python
from atlasforge.asr import transcribe_long

transcript = transcribe_long(backend, "voice-note.ogg", "ha")   # any length, any backend
print(transcript.text)
for chunk in transcript.chunks:        # our windows, not model timestamps
    print(chunk.start_s, chunk.end_s, chunk.text)
```

Lower-level pieces: `decode_audio` (any file to 16 kHz mono PCM), `plan_windows` (the sample ranges), and `merge_texts` (join transcripts and drop words duplicated across overlaps):

```python
from atlasforge.asr import merge_texts, plan_windows

plan_windows(16_000 * 70)                       # [(0, 448000), (416000, 864000), (832000, 1120000)]
merge_texts(["a b c d e", "d e f g"])           # 'a b c d e f g'
```

To make *any* backend accept long audio, wrap it: `LongAudioBackend(backend)` from `atlasforge.asr.chunking`. The `eval` command does this for ASR datasets.

## Write your own backend

Anything with these four methods is a backend, and works with `run`, `compare` and the rest. No inheritance required:

```python
from atlasforge.backends.base import Backend
from atlasforge.errors import UnsupportedFeatureError
from atlasforge.types import BackendInfo, Generation

class MyBackend:
    def generate(self, messages, params=None):
        return Generation(text=my_model(messages[-1]["content"]), latency_ms=0.0)

    def transcribe(self, audio, lang):
        raise UnsupportedFeatureError("This backend does not do speech.")

    def info(self):
        return BackendInfo(backend="mine", model="my-model-1", capabilities=frozenset({"generate"}))

    def close(self):
        pass

assert isinstance(MyBackend(), Backend)     # the protocol is runtime-checkable
summary = run(MyBackend(), dataset, "runs/mine")
```

Raise `UnsupportedFeatureError` for capabilities you lack, and leave them out of `capabilities`. Whatever `info()` returns is recorded in the run manifest, so be honest about the model name and revision.

## Handle errors

All deliberate errors derive from `AtlasForgeError` and carry a `hint` with the next step:

```python
from atlasforge.errors import AtlasForgeError

try:
    load_dataset("missing.jsonl", "generation")
except AtlasForgeError as exc:
    print(exc.message)     # cannot read missing.jsonl: No such file or directory
    print(exc.hint)        # Check the path.
    print(exc.format())    # both, as the CLI shows them
```

The full hierarchy is in [Errors and exit codes](../reference/errors.md).
