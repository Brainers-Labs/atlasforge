# Quickstart

In about ten minutes you will send a prompt to N-ATLaS, evaluate it on a small dataset, and compare two models. There are two ways to reach a model. Pick one.

| | Path A: an endpoint | Path B: your own machine |
|---|---|---|
| You have | the URL of an OpenAI-compatible server already serving N-ATLaS (vLLM, a team gateway, a hosted endpoint) | a laptop or workstation, and you have accepted the licence on Hugging Face |
| Backend | `--backend openai --base-url ...` | `--backend openai` pointing at a local Ollama/llama.cpp, or `--backend local` |
| Good for | quick evaluation, teams, CI | offline work, small machines, trying things out |

Whichever you choose, the commands from step 3 onward are identical.

## 1. Install and check

```bash
git clone https://github.com/im-aderm/atlasforge && cd atlasforge
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
atlasforge doctor
```

See [Installation](installation.md) for what each check means.

## 2. Get a model to talk to

=== "Path A: an existing endpoint"

    Set the URL once so you do not repeat it:

    ```bash
    export ATLASFORGE_BASE_URL=http://127.0.0.1:8000/v1
    export ATLASFORGE_API_KEY=...        # only if the server requires a key
    ```

    The server must speak the OpenAI chat API (`POST /v1/chat/completions`). The `--model` name must match what the server serves. The default is `NCAIR1/N-ATLaS`.

    !!! tip "Plain `http://` is only allowed to your own machine"
        To protect your key, AtlasForge refuses plain `http://` to anything other than `localhost`, `127.0.0.1` or `::1`. Use `https://`, or pass `--allow-insecure-http` if you trust the network.

=== "Path B: your own machine"

    The fp16 LLM is 16 GB, which does not fit on most laptops. The practical route is to serve a **quantised** copy locally with [Ollama](https://ollama.com) or llama.cpp and point AtlasForge at it. The full recipe, including the one gotcha (Ollama drops the chat template), is in [Serve the models](../guides/serve-models.md#ollama-quantised-local-serve). Once served:

    ```bash
    export ATLASFORGE_BASE_URL=http://127.0.0.1:11434/v1
    ```

    Then use `--model` with the name you gave the model, for example `--model natlas-local-chat`.

    For the **speech** models, which are small, you can load them directly instead: see [Transcribe and evaluate speech](../guides/asr.md).

## 3. Send one prompt

```bash
atlasforge run "What is the capital of Nigeria? Answer in one sentence." --model NCAIR1/N-ATLaS
```

```text
The capital of Nigeria is Abuja.
```

Add `--json` to also get the model name, latency, finish reason and token usage, and `--system "..."` to set a system prompt. Use `-` as the prompt to read it from standard input.

## 4. Make a tiny dataset

A dataset is a JSONL file: one JSON object per line. Save this as `data.jsonl`:

```json
{"id": "g1", "input": "Translate to English: Sannu", "reference": "Hello", "lang": "ha", "meta": {"domain": "greetings"}}
{"id": "g2", "input": "Translate to English: Ẹ káàárọ̀", "reference": "Good morning", "lang": "yo", "meta": {"domain": "greetings"}}
{"id": "q1", "input": "What is the capital of Nigeria?", "reference": "Abuja", "lang": "en", "meta": {"domain": "facts"}}
```

!!! warning "This is a format example, not a benchmark"
    Three rows tell you nothing about a model. Real evaluation needs hundreds of examples from your own task. The [dataset format reference](../reference/dataset-format.md) lists every field.

Check it before you spend any compute:

```bash
atlasforge dataset validate data.jsonl --task generation
```

`validate` reports every problem at once (malformed lines, duplicates, broken Unicode, stripped diacritics) and exits `1` if there are errors. See [Validate a dataset](../guides/validate-dataset.md).

## 5. Evaluate

```bash
atlasforge eval data.jsonl --out runs/base --model NCAIR1/N-ATLaS
```

AtlasForge sends each example to the model, saves the answers, scores them, and prints a table. It writes four files into `runs/base/`:

```text
runs/base/
  run.json         what was run (dataset hash, model, parameters)
  results.jsonl    one answer per example, appended as they finish
  report.md        a readable report
  report.json      the same numbers, machine-readable
```

If the run is interrupted, run the same command again: finished examples are skipped. The default metrics for a `generation` task are `exact_match` and `chrf`. Every text metric appears twice, once **tone-aware** and once **tone-insensitive**; see [Tone-aware scoring](../concepts/tone-aware-scoring.md).

## 6. Compare two models

Evaluate a second model (your fine-tune, a different quantisation, a different prompt) on the **same** dataset into a different directory:

```bash
atlasforge eval data.jsonl --out runs/tuned --base-url http://127.0.0.1:8001/v1 --model my-finetune
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned --out cmp --slice domain
```

You get `cmp/comparison.md` and `cmp/comparison.json`. For each metric the report shows the difference, a 95% confidence interval, and a verdict: *improved*, *regressed* or *no clear change*. The `--slice domain` option breaks the result down by the `domain` field in your `meta`, so you can see **where** it got worse, not only whether it did. See [Compare two models](../guides/compare.md).

## 7. Transcribe audio

The speech models are small enough to run directly on most machines:

```bash
pip install -e ".[asr]"
atlasforge transcribe note.ogg --lang ha --backend local
```

Long recordings are split into overlapping windows automatically. Details in [Transcribe and evaluate speech](../guides/asr.md).

## Where to go next

- [Evaluate a model](../guides/evaluate.md): every option, resuming, concurrency, failure handling.
- [Dataset format](../reference/dataset-format.md): classification and ASR datasets, multi-turn chats.
- [Honest statistics](../concepts/honest-statistics.md): how to read a comparison report without fooling yourself.
- [Troubleshooting](../operations/troubleshooting.md): when something goes wrong.
