# Installation

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10, 3.11, 3.12 or 3.13 |
| Operating system | Linux, macOS or Windows. CI tests all three. |
| ffmpeg | Only for speech: it decodes `.ogg`, `.m4a`, `.mp3` and `.wav` |
| A GPU | **Not** needed for evaluation, comparison or the demo. Needed for the 8B model locally and for fine-tuning. See [Choose your setup](choose-your-setup.md). |

## Install

The distribution is `brainers-atlasforge`, and the current version is a pre-release. `--pre` is
shown below but is not required while the alpha is the only version published — pip falls back to
pre-releases when a project has no stable release to prefer, so the plain command installs it too.
It becomes necessary once a stable version exists, because pip will then choose that one:

```bash
pip install --pre brainers-atlasforge
atlasforge --version
```

The import name and the command are both `atlasforge`. The distribution name differs because the
plain name on PyPI belongs to an unrelated bioinformatics project.

## Install from source

To work on AtlasForge itself, or to run the tip of `main` rather than a release, install from a clone:

```bash
git clone https://github.com/Brainers-Labs/atlasforge
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

!!! warning "If PowerShell refuses to run the activation script"
    A default Windows PowerShell blocks `.ps1` files, and says so: *"Activate.ps1 cannot be
    loaded because running scripts is disabled on this system."* That is Windows' default
    policy, not a fault in the virtual environment. Allow it for this session only:

    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .venv\Scripts\Activate.ps1
    ```

    Or skip activation altogether and call the environment's interpreter directly. Every
    command works the same way, and nothing needs to be unblocked:

    ```powershell
    .venv\Scripts\python.exe -m pip install -e .
    .venv\Scripts\atlasforge.exe --version
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
| `pip install "brainers-atlasforge"` | Typer, Rich, httpx, NumPy, jiwer, sacrebleu | Everything that talks to a server, plus scoring, comparing and the demo |
| `pip install "brainers-atlasforge[local]"` | PyTorch, Transformers, Accelerate, bitsandbytes | Running the models in-process (`--backend local`) |
| `pip install "brainers-atlasforge[asr]"` | PyTorch, Transformers, librosa, soundfile | The official speech models |
| `pip install "brainers-atlasforge[finetune]"` | the local extras, plus PEFT, TRL, datasets, PyYAML | `atlasforge finetune` |
| `pip install "brainers-atlasforge[bench]"` | lm-evaluation-harness, PyTorch, Transformers, Accelerate | `atlasforge bench afrobench` — the published AfroBench-LITE suite |
| `pip install "brainers-atlasforge[docs]"` | MkDocs Material and plugins | Building this documentation |
| `pip install "brainers-atlasforge[dev]"` | test, lint, type-check tools, and the docs extra | Contributing |

In a clone the same extras are written `pip install -e ".[local]"`, `-e ".[asr]"` and so on.
`--pre` may be added to any of these; it is not required yet, for the reason given above.

!!! note "Quote the brackets in zsh"
    In zsh (the macOS default) write `pip install "brainers-atlasforge[local]"` with the quotes, or the shell tries to expand the brackets.

!!! warning "The PyPI name is `brainers-atlasforge`, not `atlasforge`"
    `atlasforge` on PyPI is an unrelated bioinformatics project, so the published distribution is
    **`brainers-atlasforge`** — but the import name and the command are both still `atlasforge`,
    and so is every example on this site. See [Releasing](../project/releasing.md).

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
pip uninstall brainers-atlasforge
```

Your datasets, run directories and reports are ordinary files in your own folders; AtlasForge never stores anything elsewhere.
