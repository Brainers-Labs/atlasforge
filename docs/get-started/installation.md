# Installation

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10, 3.11, 3.12 or 3.13 |
| Operating system | Linux, macOS or Windows. CI tests all three. |
| ffmpeg | Only for speech: it decodes `.ogg`, `.m4a`, `.mp3` and `.wav` |
| A GPU | **Not** needed for evaluation, comparison or the demo. Needed for the 8B model locally and for fine-tuning. See [Choose your setup](choose-your-setup.md). |

## Install from source

AtlasForge is not on PyPI yet. Install it from a clone of the repository:

```bash
git clone https://github.com/im-aderm/atlasforge
cd atlasforge
python -m venv .venv
```

=== "macOS / Linux"

    ```bash
    . .venv/bin/activate
    pip install -e .
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\Activate.ps1
    pip install -e .
    ```

Check it works:

```bash
atlasforge --version
atlasforge doctor
```

## Optional extras

The base install is deliberately light: it pulls in no PyTorch and no Transformers. Add only what you need.

| Install | Adds | Use it for |
|---|---|---|
| `pip install -e .` | Typer, Rich, httpx, NumPy, jiwer, sacrebleu | Everything that talks to a server, plus scoring, comparing and the demo |
| `pip install -e ".[local]"` | PyTorch, Transformers, Accelerate, bitsandbytes | Running the models in-process (`--backend local`) |
| `pip install -e ".[asr]"` | PyTorch, Transformers, librosa, soundfile | The official speech models |
| `pip install -e ".[finetune]"` | the local extras, plus PEFT, TRL, datasets, PyYAML | `atlasforge finetune` |
| `pip install -e ".[docs]"` | MkDocs Material and plugins | Building this documentation |
| `pip install -e ".[dev]"` | test, lint, type-check tools, and the docs extra | Contributing |

!!! note "Quote the brackets in zsh"
    In zsh (the macOS default) write `pip install -e ".[local]"` with the quotes, or the shell tries to expand the brackets.

## Install ffmpeg

ffmpeg is a system tool, not a Python package. You only need it for audio.

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Windows"

    ```powershell
    winget install Gyan.FFmpeg
    ```

    Open a new terminal afterwards so the updated `PATH` is picked up.

=== "Linux (Debian/Ubuntu)"

    ```bash
    sudo apt install ffmpeg
    ```

## Verify with `atlasforge doctor`

```bash
atlasforge doctor
```

`doctor` checks your Python version, GPU and memory, free disk, ffmpeg, Hugging Face token and which extras are installed. Each line is `OK`, `WARN` or `FAIL`, and every non-OK line tells you the exact next step.

- A `WARN` is information, not an error. For example, "no NVIDIA GPU detected" is fine if you only use a server.
- Only a `FAIL` (such as an unsupported Python version) makes the command exit with code 1.
- Add `--json` for machine-readable output.

## Run from Python without installing a command

```bash
python -m atlasforge --help
```

## Uninstall

```bash
pip uninstall atlasforge
```

Your datasets, run directories and reports are ordinary files in your own folders; AtlasForge never stores anything elsewhere.
