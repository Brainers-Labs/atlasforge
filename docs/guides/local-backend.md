# Run locally with transformers

`--backend local` loads the N-ATLaS models **inside the AtlasForge process** with Hugging Face `transformers`. There is no server to run. It is the simplest option for the small speech models and for the LLM on a machine with a large GPU.

```bash
pip install -e ".[local]"      # LLM
pip install -e ".[asr]"        # speech models
```

## Verification status

| Part | Status |
|---|---|
| **Speech models** (`transcribe`, ASR `eval`) | **VERIFIED**: run for real on a Mac M1 (Apple MPS and CPU) |
| **LLM** (`run`, `eval`, fp16) | **NOT VERIFIED**: the code is written against the model card and tested with stand-in modules only. The 16.08 GB of fp16 weights cannot fit a 16 GB machine, so it needs a GPU of 24 GB or more |

Do not rely on the local LLM path until it has been run on suitable hardware.

## Speech models

```bash
atlasforge transcribe note.ogg --lang ha --backend local
atlasforge eval speech.jsonl --task asr --backend local --lang ha --out runs/asr
```

The ASR model for the language is downloaded on first use (gated; see [Models and access](../getting-started/models-and-access.md)) and kept in memory for the rest of the process. One model is loaded **per language** actually used.

Device selection (`--device`):

| Value | Meaning |
|---|---|
| `auto` (default) | an NVIDIA GPU if available, else Apple MPS if available, else CPU |
| `cuda` / `mps` / `cpu` | force one |

Audio longer than 30 seconds must go through `transcribe` or `eval` (which chunk it). Calling the backend directly with longer audio raises an `AudioError` that points you to `transcribe_long`.

## The LLM

```bash
atlasforge run "Ina kwana?" --backend local --model NCAIR1/N-ATLaS
atlasforge eval data.jsonl --out runs/local --backend local --model NCAIR1/N-ATLaS
```

What the backend does, per request:

1. Loads the tokenizer and model on first use (`float16`, `device_map="auto"` when `--device auto`).
2. Builds the prompt with the **tokenizer's own chat template**. If a tokenizer has none, it raises an error rather than guessing a format.
3. Caps `max_new_tokens` so prompt plus output stays within the model's `max_position_embeddings`; if the prompt alone fills the context, it raises a `ResourceError`.
4. Generates with `repetition_penalty` from your settings, sampling when `temperature > 0` (and greedy at `0`), and seeds `torch` when you pass a seed.
5. Reports token usage and a `finish_reason` of `length` (hit the cap) or `stop`.

### Quantisation

`--quantize 4bit` or `8bit` uses `bitsandbytes`, which **requires an NVIDIA GPU**. On a Mac or a CPU-only machine AtlasForge refuses with an explanation and suggests serving a quantised model through Ollama or llama.cpp instead (see [Serve the models](serve-models.md)). `bitsandbytes` is not installed on macOS at all.

### Memory

| Setting | Weights size | Notes |
|---|---|---|
| fp16 / bf16 | 16.08 GB on disk | needs a GPU with comfortably more than that, plus room for the KV cache |
| 4-bit | roughly a quarter | an estimate from the model card (about 6 GB); not measured by us |

`atlasforge doctor` flags GPUs under about 16 GB. When loading fails for lack of memory you get a `ResourceError` with the same advice.

## Errors you may see

| Message | Meaning |
|---|---|
| `The local backend needs 'torch', which is not installed.` | install the `local` or `asr` extra |
| `Cannot access NCAIR1/...` | licence not accepted on that repo, or no/wrong token |
| `Not enough memory to load ...` | the model does not fit; quantise or serve elsewhere |
| `Could not load NCAIR1/... (ImportError)` | a broken dependency; on macOS usually the `scipy` wheel, see [Troubleshooting](../operations/troubleshooting.md) |

## Always close backends

The backend holds model weights. The CLI closes it for you. In your own code, call `close()` (in a `try`/`finally`), which frees accelerator memory and is safe to call twice. `OpenAIBackend` additionally works as a context manager (`with` block); `LocalBackend` does not. See the [Python API guide](python-api.md).
