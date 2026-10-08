# Run directories

`atlasforge eval --out runs/base` creates a **run directory**: a folder that holds everything about one model's pass over one dataset.

```text
runs/base/
├── run.json          what was run: dataset fingerprint, model, revision, settings
├── results.jsonl     one line per example: the raw answer, latency, or the error
├── report.json       scored metrics (written by eval and report)
├── report.md         the same, as a readable report
└── report.html       the same again, as one self-contained page for a browser
```

## `run.json`: the manifest

It records exactly what produced the results, so they can be trusted and compared later. This is a real manifest from the demo (trimmed):

{{ file_example("run") }}

## `results.jsonl`: the raw answers

Each line is one example's outcome. Exactly one of `prediction` or `error` is set. The first two lines of the demo's base run:

{{ file_example("results") }}

Errors are stored as `ExceptionName: message`. For anything unexpected only the exception **type** is stored, never its text, so a prompt can never leak into your results through an error message.

## The report, three times

Scoring writes the same result in three shapes, so you can pick the one that fits:

| File | For |
|---|---|
| `report.json` | A script, or a submission checker. Every number, no prose. |
| `report.md` | Reading and pasting into a README or a submission. |
| `report.html` | Looking at, or sending to someone. One self-contained file — no server, no JavaScript, no network. |

They are the same numbers. The two readable ones are rendered from one `ScoreReport`, so they cannot drift apart. `atlasforge report runs/base --dataset data.jsonl` rewrites all three from `results.jsonl`, without calling the model again.

## Resuming

Run the same command again with the same `--out`, and finished examples are skipped:

```bash
atlasforge eval data.jsonl --out runs/base ...   # interrupted at example 600 of 1000
atlasforge eval data.jsonl --out runs/base ...   # picks up at 601
```

- Failed examples are **retried** by default (`--no-retry-errors` to keep them as they are).
- Results are written one line at a time, so an interruption loses at most the example in flight. If the last line was cut off, it is dropped and that example runs again.
- A corrupt line **anywhere else** stops the run with a clear error, because skipping it silently would hide data loss.

## Runs are never mixed

Before resuming, AtlasForge compares the new settings with `run.json`. It refuses, with a message saying what changed, if any of these differ:

| Checked | Why |
|---|---|
| Dataset fingerprint and task | Results from different data are not comparable |
| Model name and revision | A different model is a different run |
| Generation settings | Temperature and similar change the answers |
| Default language | It changes which ASR model is used |

Use a new `--out` directory to start a different run. A LoRA adapter is part of the model name (`NCAIR1/N-ATLaS+my-adapter`), so a base run and an adapter run can never be confused.

## The circuit breaker

If a server dies mid-run, retrying every remaining example would take hours and achieve nothing. After **20 failures in a row** (`--max-consecutive-failures`, `0` turns it off) the run stops with:

```text
error: Stopped after 20 failures in a row. Last error: BackendConnectionError: ...
  -> Fix the cause, then re-run with the same --out to resume; everything already finished is kept.
```

Fix the server and run the same command again. Nothing is lost.

## Comparing runs

`compare` takes two run directories and the dataset they were both made from, and refuses if either run came from a different dataset file. Because runs hold raw answers, you can compare them without any model running.

With `--out`, it writes the comparison in the same three shapes as a run's report: `comparison.json` for a script, `comparison.md` to read, and `comparison.html` — the same page treatment, with each metric's change drawn against its confidence interval.
