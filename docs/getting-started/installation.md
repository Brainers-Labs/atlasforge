# Installation

## Requirements

| Requirement | Needed for | Notes |
|---|---|---|
| Python 3.10, 3.11, 3.12 or 3.13 | everything | CI tests Linux, macOS and Windows on 3.10 and 3.13 |
| `ffmpeg` on your `PATH` | audio (`transcribe`, ASR evaluation) | a system tool, not a Python package |
| A Hugging Face account and token | downloading the models locally | the models are gated; see [Models and access](models-and-access.md) |
| A GPU, **or** a remote/quantised endpoint | the 8B LLM | the fp16 weights are 16 GB; see [Serve the models](../guides/serve-models.md) |

You do **not** need a GPU, a token or any model to install AtlasForge, validate datasets, re-score finished runs or run the test suite.

!!! note "Not on PyPI yet"
    AtlasForge has not been published to PyPI. Install it from the repository as shown below. This page will change when a release is published.

## Install from source

```bash
git clone https://github.com/im-aderm/atlasforge
cd atlasforge
python3 -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

The core install is deliberately light. It pulls in only `typer`, `rich`, `httpx`, `numpy`, `jiwer` and `sacrebleu`, and **never** `torch` or `transformers`.

## Optional extras

Install an extra only when you need what it enables.

| Extra | Command | Adds | Enables |
|---|---|---|---|
| `local` | `pip install -e ".[local]"` | `torch`, `transformers`, `accelerate`, `safetensors`, `huggingface-hub`, plus `bitsandbytes` on non-macOS | `--backend local` for the LLM |
| `asr` | `pip install -e ".[asr]"` | `torch`, `transformers`, `huggingface-hub`, `librosa`, `soundfile` | `--backend local` for speech models |
| `finetune` | `pip install -e ".[finetune]"` | everything in `local`, plus `peft`, `trl`, `datasets` | reserved for the fine-tuning recipes (not written yet) |
| `dev` | `pip install -e ".[dev]"` | `pytest`, `ruff`, `mypy`, `pip-audit`, `pre-commit` | running the checks; see [Development](../project/development.md) |
| `docs` | `pip install -e ".[docs]"` | `mkdocs-material`, `mkdocstrings` | building this site |

Extras can be combined: `pip install -e ".[asr,local]"`.

### Install ffmpeg

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Linux (Debian/Ubuntu)"

    ```bash
    sudo apt install ffmpeg
    ```

=== "Windows"

    ```powershell
    winget install Gyan.FFmpeg
    ```

ffmpeg lets AtlasForge read any common audio format, including WhatsApp voice notes (`.ogg`/opus), `.m4a`, `.mp3` and `.wav`, and convert it to the 16 kHz mono audio the models expect.

## Check your setup

```bash
atlasforge doctor
```

`doctor` inspects your environment and prints one row per check. It exits with code `1` only if a check **fails**; warnings never fail it.

| Check | What it tests | Status rules |
|---|---|---|
| `python` | Python version | fails below 3.10 |
| `gpu` | NVIDIA GPUs via `nvidia-smi` | warns if none; warns if under about 16 GB (fp16 estimate) and mentions 4-bit at about 6 GB |
| `disk` | free space in your home directory | warns under 25 GB (the LLM weights are about 16 GB) |
| `ffmpeg` | `ffmpeg` on `PATH` | warns if missing, with the install command for your OS |
| `hf-token` | `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, or a cached login | warns if absent; a found token is shown masked, like `hf_****abcd` |
| `extra:local`, `extra:asr`, `extra:finetune` | whether each extra's packages are importable | warns and prints the `pip install` command for what is missing |

Example on a Mac with no NVIDIA GPU and nothing optional installed yet:

```text
┏━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Status ┃ Check          ┃ Detail                                             ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ OK     │ python         │ 3.10.12                                            │
│ WARN   │ gpu            │ no NVIDIA GPU detected                             │
│ OK     │ disk           │ 161 GB free                                        │
│ OK     │ ffmpeg         │ /opt/homebrew/bin/ffmpeg                           │
│ WARN   │ hf-token       │ no token found (needed for the gated NCAIR1        │
│        │                │ models)                                            │
│ WARN   │ extra:local    │ missing torch, transformers, accelerate            │
│ WARN   │ extra:asr      │ missing transformers, librosa, soundfile           │
│ WARN   │ extra:finetune │ missing peft, trl, datasets                        │
└────────┴────────────────┴────────────────────────────────────────────────────┘
```

For scripts and CI, `atlasforge doctor --json` prints the same checks as JSON.

!!! warning "The GPU check only recognises NVIDIA"
    `doctor` queries `nvidia-smi`. On Apple Silicon it will say "no NVIDIA GPU detected" even though the Apple GPU (MPS) is usable for the speech models. That is expected. See [Run on a Mac](../guides/mac.md).

## Platform notes

### macOS and the `scipy` pin

`transformers` imports `scipy` when it loads. On recent macOS releases on Apple Silicon, the default `scipy` 1.15 wheel fails to load one of its extensions (`_propack`) with a Mach-O `zero-fill section` error, which surfaces as `Could not load NCAIR1/... (ImportError)`. The `asr` and `local` extras therefore pin `scipy<1.15` on macOS. If you installed packages some other way, run `pip install "scipy<1.15"`. See [Troubleshooting](../operations/troubleshooting.md#could-not-load-ncair1-importerror-on-macos).

### Windows

Everything works, including the full test suite. Use `.venv\Scripts\activate` to activate the environment. `bitsandbytes` (4-bit/8-bit quantisation) is installed by the `local` extra on Windows and Linux, but it needs an NVIDIA GPU to be useful.

## Verify the install

```bash
atlasforge --version
atlasforge --help
```

To run the project's own test suite (no GPU, token or model needed):

```bash
pip install -e ".[dev]"
pytest
```

## Next step

Continue to the [Quickstart](quickstart.md).
