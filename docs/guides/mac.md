# Run on a Mac (Apple Silicon)

This page records what actually works on a MacBook with an M1 chip and 16 GB of unified memory, because that is the machine AtlasForge's model-side verification was done on. Larger Macs will do better; the same approach applies.

## What fits, what does not

| Task | On a 16 GB M1 | How |
|---|---|---|
| Install, validate datasets, re-score runs, run the tests | works | no model needed |
| Speech: Hausa, Yoruba, Igbo, English | **works** | `--backend local`, runs on Apple's GPU (MPS) or CPU |
| LLM at fp16 | **does not fit** | the weights alone are 16.08 GB, about 93% of total RAM |
| LLM, int4 through Ollama | **works** | a 6.6 GB model; see [Serve the models](serve-models.md#ollama-quantised-local-serve) |
| LLM 4-bit through `--quantize 4bit` | **not available** | needs `bitsandbytes`, which needs an NVIDIA GPU |
| Fine-tuning the LLM | not practical | needs a GPU box |

## Setup that was verified

```bash
brew install ffmpeg
git clone https://github.com/im-aderm/atlasforge && cd atlasforge
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,asr]"
pytest && ruff check . && ruff format --check . && mypy      # 559 tests pass
atlasforge doctor
```

On this machine `doctor` shows a `gpu` warning ("no NVIDIA GPU detected") and warns about the `local` extra missing `accelerate`. Both are expected: the check looks only for NVIDIA GPUs, and `accelerate` is only needed for the LLM path.

## Speech on the Apple GPU

```bash
atlasforge transcribe clip.wav --lang ha --backend local
```

With `--device auto`, AtlasForge uses MPS when PyTorch reports it available. A 19-second Hausa clip transcribed in about 22 seconds wall time including the first model load. Pass `--device cpu` to force the CPU.

## The LLM through Ollama

1. Install [Ollama](https://ollama.com) and make sure it is running.
2. Download the official weights with `HF_HUB_DISABLE_XET=1` (about 16 GB; free disk space needed).
3. `ollama create ... --quantize int4`, then add the chat template.
4. Point AtlasForge at `http://127.0.0.1:11434/v1`.

Every step, with the exact files, is in [Serve the models](serve-models.md#ollama-quantised-local-serve). Remember the two warnings there: the result is a degraded copy, and quantised weights must stay on your machine.

## Known Mac issues

### `Could not load NCAIR1/... (ImportError)`

A `scipy` wheel problem on recent macOS, not an AtlasForge bug. The `asr` and `local` extras pin `scipy<1.15` on macOS; if you installed packages another way, `pip install "scipy<1.15"`. Details in [Troubleshooting](../operations/troubleshooting.md#could-not-load-ncair1-importerror-on-macos).

### `ModuleNotFoundError: No module named '_lzma'`

Python built by `pyenv` without the `xz` library lacks the `lzma` module, which breaks the `datasets` package (an optional dependency of the `finetune` extra). Install `xz` with Homebrew and reinstall that Python (`brew install xz`, then `pyenv install <version>` again). Not needed for anything else.

### Downloads failing with `CAS Client Error`

Set `HF_HUB_DISABLE_XET=1`.

## Memory etiquette

Apple Silicon shares one memory pool between the system, apps and the GPU. Do not try to load the fp16 LLM with `--backend local` on a 16 GB machine: it risks swapping hard or freezing the system. AtlasForge's own estimate checks (`doctor`) are advisory only.
