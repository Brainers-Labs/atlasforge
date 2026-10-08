# Use it in Colab or a notebook

Everything AtlasForge does in a terminal it does in a cell. The difference is not the tool, it is
the three things a notebook changes: **how you install**, **where your files live**, and **what the
runtime gives you**. This page covers all of them, and is written for a free Colab notebook first —
the same rules apply in JupyterLab, VS Code, Kaggle or a Workbench instance.

!!! quote "The short version"
    ```text
    %pip install -q brainers-atlasforge     # a cell magic, not a shell command
    !atlasforge doctor                      # a shell command, with a leading !
    ```
    A free runtime is enough for the demo, the reports, the comparison and the data checks. The
    rows that touch a real model need a GPU runtime, a token and the accepted licence.

## 1. Start from the finished notebook

The repository ships a notebook that runs the whole workflow — validate, score, compare, render —
on synthetic demo data. It needs no model, no token and no GPU, so it runs top to bottom on a free
runtime in about a minute.

[Open it in Colab](https://colab.research.google.com/github/Brainers-Labs/atlasforge/blob/main/notebooks/atlasforge-quickstart.ipynb){ .md-button .md-button--primary }
[Read it on GitHub](https://github.com/Brainers-Labs/atlasforge/blob/main/notebooks/atlasforge-quickstart.ipynb){ .md-button }

| Cell | What it shows |
|---|---|
| `install` | Installing the published distribution with `%pip` |
| `demo` | Building the 92-example dataset and two finished runs |
| `report` | Scoring a run under both tone views, failures counted as wrong |
| `compare` | Pairing two runs example by example, including the `has_number` regression |
| `html` | Rendering the self-contained HTML report inline |
| `real` | The same workflow against N-ATLaS, commented out — it needs a GPU and a token |

The notebook is a test artefact, not a screenshot: its portable cells are executed on every commit.
The conventions that keep it honest are described in
[`notebooks/README.md`](https://github.com/Brainers-Labs/atlasforge/blob/main/notebooks/README.md);
the one that matters if you fork it is that a cell whose first line is `# notebook: colab-only` is
skipped by the tests — so a cell you make notebook-specific has to say so.

## 2. What works on which runtime

| Task | Free CPU runtime | GPU runtime |
|---|---|---|
| `demo`, `report`, `compare`, `dataset validate`, `card` | Yes | Yes |
| `eval`, `run`, `transcribe` against a server you point at | Yes, if the endpoint is reachable | Yes |
| `eval --backend local` (AtlasForge loads the 8B weights) | No | Yes, with `--quantize 4bit` |
| `transcribe` with the official ASR models | Yes, slowly | Yes |
| `finetune` (QLoRA) | No — it needs CUDA | Yes |
| `bench afrobench` | No | Yes, and it is a large install |

!!! warning "The model rows are still unverified"
    Everything in the CPU column is covered by the test suite. The **local backend, the official
    ASR models, fine-tuning and benchmarking have never been run against the real N-ATLaS
    weights** — in a notebook or anywhere else. [Project status](../help/status.md) lists exactly
    what has been exercised. Treat the GPU column as a plan to confirm, and report what happens.

A notebook is not magic hardware. A free runtime is a CPU with roughly 12 GB of RAM and about 100 GB
of scratch disk; a GPU runtime is an NVIDIA card — a T4 on the free tier, larger on paid tiers.
Colab's limits change, so check its own page, but the split above is the same split as
[Choose your setup](../get-started/choose-your-setup.md).

## 3. Cell syntax: three prefixes

A notebook cell is not a shell. It is Python, plus a small set of line prefixes that a plain Python
interpreter would reject:

| Prefix | Meaning | Example |
|---|---|---|
| `%` | An IPython *magic*, run by the kernel | `%pip install -q brainers-atlasforge` |
| `!` | The rest of the line is a shell command | `!atlasforge doctor` |
| `%%` | The magic applies to the whole cell | `%%writefile finetune.yaml` |

**Use `%pip`, never `!pip`.** `%pip` installs into the interpreter that is running the notebook, so
the next cell can import what you installed. `!pip` runs whatever `pip` is first on `PATH`, which is
sometimes a different environment — the classic *"I installed it and it still says
`ModuleNotFoundError`"*. If you do install into the wrong place, or install a second time in one
session after an import, restart the runtime.

!!! note "Blocks on this page are written as plain shell"
    Every command below is shown the way you would type it in a terminal, so it pastes into one. In
    a notebook cell, add the `!` — `!atlasforge demo`. Nothing else changes.

## 4. Install

The distribution is `brainers-atlasforge`; the import name and the command are both `atlasforge`.
In a cell:

```text
%pip install -q brainers-atlasforge
```

That is the whole install for the demo, the reports, the comparison and the data checks. Add an
extra when you need one, quoting the brackets so the shell does not try to expand them:

```text
%pip install -q "brainers-atlasforge[local]"
%pip install -q "brainers-atlasforge[asr]"
%pip install -q "brainers-atlasforge[finetune]"
```

The same extras in a terminal are `pip install "brainers-atlasforge[local]"` and so on — the
[installation page](../get-started/installation.md) lists what each one brings.

The base install pulls in no PyTorch and no Transformers, which matters more in a notebook than
anywhere else: `[local]`, `[finetune]` and `[bench]` are each a multi-gigabyte download, and a
hosted runtime is a fresh, empty machine every session, so they are pulled down again every time.
Put the install in the first cell and accept the wait, or mount Drive and keep a wheel cache there.

Check what you got:

```bash
atlasforge --version
atlasforge doctor
```

`doctor` is worth running in a notebook for the same reason as anywhere else: it reports the Python
version, the GPU and its memory, free disk, ffmpeg, whether an `HF_TOKEN` is visible — masked, never
in full — and which extras are installed. If a row is wrong, this is the cheapest place to find out.

```bash
atlasforge doctor --json
```

`--json` is the notebook-friendly form: a cell's output is stored inside the `.ipynb`, and one line
of JSON is far less to scroll past than a full table.

## 5. The walkthrough, in cells

These are the [quickstart](../get-started/quickstart.md) commands. In Colab, each one is prefixed
with `!`:

```bash
atlasforge demo atlasforge-demo
atlasforge dataset validate atlasforge-demo/toy_qa.jsonl
atlasforge report atlasforge-demo/runs/base --dataset atlasforge-demo/toy_qa.jsonl
atlasforge compare atlasforge-demo/toy_qa.jsonl \
  --base atlasforge-demo/runs/base \
  --candidate atlasforge-demo/runs/tuned \
  --slice domain
```

{{ transcript("demo") }}

The same steps through the Python API, which is usually what you want in a notebook — the return
values are objects, so the next cell can compute on them instead of parsing printed text:

```python
from pathlib import Path

from atlasforge import write_reports
from atlasforge.demo import build_demo

demo = Path("atlasforge-demo")
if not demo.exists():
    build_demo(demo)  # refuses a folder that already holds other files

base = demo / "runs" / "base"
write_reports(base, demo / "toy_qa.jsonl")
print((base / "report.md").read_text(encoding="utf-8")[:400])
```

`build_demo` refuses a non-empty folder rather than mixing its files with yours — that is why the
guard is there, and why the notebook gives its demo its own directory. Every entry point is listed
in the [Python API reference](../reference/python-api.md); the CLI and the API are the same code, so
`atlasforge report` calls `write_reports` and `atlasforge compare` calls `compare_runs`.

## 6. Show a report in the notebook

A cell's output is HTML, so a report can be displayed rather than printed. AtlasForge's HTML reports
are **one self-contained file** — inline SVG, no JavaScript, nothing fetched from the network —
which is exactly what a notebook wants: it renders the same offline, and it makes no request the
runtime might block.

```python
# docs:run
from pathlib import Path

from atlasforge import write_reports

run = Path("atlasforge-demo/runs/base")
write_reports(run, "atlasforge-demo/toy_qa.jsonl")

page = run / "report.html"
html = page.read_text(encoding="utf-8")

try:
    from IPython.display import HTML, display
except ImportError:  # running as a plain script
    print(f"{len(html):,} characters of self-contained HTML in {page.name}")
else:
    display(HTML(html))
```

The `try`/`except` is not decoration. The same cell has to work when the notebook is exported to a
script and run headless — which is how the repository tests it — so the display step degrades into a
line of text instead of an exception.

For a report too tall to read inline, download it and open it in a browser:

```python
from google.colab import files

files.download("comparison/comparison.html")
```

Markdown and JSON need no work at all: `report.md` and `report.json` sit beside the HTML, and
`print(Path("comparison/comparison.md").read_text())` is often clearer than a rendered page.

## 7. Keep the token out of the notebook

A notebook is a file that gets shared, emailed and committed with its outputs attached. A token
typed into a cell is a token in the file, and a token printed by a cell is a token in the output.

Use the platform's secret store, which keeps it out of the `.ipynb` entirely. In Colab, open the
key icon in the left sidebar, add a secret named `HF_TOKEN`, switch on notebook access, and read it
into the environment:

```python
import os

from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

On Kaggle the same idea with a different API — Add-ons → Secrets, then
`from kaggle_secrets import UserSecretsClient` and `UserSecretsClient().get_secret("HF_TOKEN")`.
Anywhere else, set the variable in the environment that starts the kernel. The name AtlasForge reads
is on the [environment page](../reference/environment.md), and it is in the tool's own redaction
list, so it cannot reach a report or a log.

!!! danger "Two ways to leak a token that look harmless"
    - `%env HF_TOKEN=hf_...` **echoes the value into the cell output**, which is saved in the
      notebook file. Use the secret store.
    - `!export HF_TOKEN=hf_...` also echoes it, and the variable is gone by the next cell anyway —
      every `!` line is a separate shell process.

AtlasForge itself never prints a token: `doctor` shows `hf_****abcd`, and the redaction list is
tested. The leak to worry about is your own shell, not the tool. If a token does end up in a
notebook you have shared, revoke it at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and make a new one.

The same care applies to data. Outputs are stored in the file, so a cell that prints examples prints
them into the notebook. Clear the outputs before sharing one, and see the
[security policy](../project/security.md).

## 8. Where your files go, and what survives a disconnect

A hosted runtime is a fresh machine every session, and the disk it gives you is temporary: when the
session ends, everything in it is gone.

| Location | Survives a disconnect? |
|---|---|
| The working directory (`/content` in Colab, `/kaggle/working` on Kaggle) | No |
| Mounted cloud storage (Drive, GCS, S3) | Yes |
| A file you downloaded | Yes — it is on your machine |

This is a smaller problem for AtlasForge than for most tools, because **the run directory is the
unit of work, and it resumes**. `eval` writes `run.json` as a manifest and appends each answer to
`results.jsonl` as it arrives; run the same command again against the same `--out` and the examples
already finished are skipped. A disconnect in the middle of a long evaluation on a paid GPU is time
lost, not work lost — provided the run directory was somewhere that survived.

```python
from google.colab import drive

drive.mount("/content/drive")
```

Then put runs and reports under `/content/drive/MyDrive/...` and point `--out` there.
[Run directories](../concepts/run-directories.md) explains what each file is.

!!! tip "Read from local disk, write to Drive"
    Drive is a network filesystem, and a large `results.jsonl` read from it is slow. The usual
    pattern is to evaluate into `/content/runs/...` and copy the finished directory — or just the
    reports — to Drive at the end. Nothing in AtlasForge requires either layout.

Two habits that pay for themselves:

- **Do not re-run a finished evaluation.** `report` and `compare` need no model and no GPU. If the
  session died after the answers were recorded, re-open the run directory and score it: seconds of
  CPU work, not another pass over the model.
- **Download the report before the session ends.** The HTML is one file and it travels well.

## 9. Evaluate a model from a notebook

There are two ways to get answers out of N-ATLaS in a notebook, and they differ in where the model
runs.

### Point at an endpoint you already have

This is the `openai` backend, and it is the default. It works from a CPU-only notebook, because the
model is running somewhere else:

```bash
atlasforge eval my_data.jsonl --out runs/base \
  --base-url https://your-server:8000/v1 \
  --model NCAIR1/N-ATLaS
```

Plain `http://` is only accepted for `localhost` and `127.0.0.1`. A remote server must use `https`,
or you must opt in knowingly with `--allow-insecure-http`. See
[Serve N-ATLaS](serve-n-atlas.md) for the server side.

### Serve the model inside the notebook

On a GPU runtime you can start a server in one cell and talk to it from the next, over the loopback
address — which is why the localhost rule above matters here: no tunnel, no certificate, no exposed
port.

```text
%pip install -q vllm
!vllm serve NCAIR1/N-ATLaS
```

`vllm serve` blocks, so in a real notebook you start it in the background (`!nohup vllm serve ... &`,
or a `subprocess.Popen` in a Python cell) and give it time to load before the evaluation cell runs.
The endpoint it creates is `http://127.0.0.1:8000/v1`, which is the default `--base-url`, so the
evaluation is just:

```bash
atlasforge eval my_data.jsonl --out runs/natlas --model NCAIR1/N-ATLaS
```

!!! danger "This has not been run against the real weights"
    The commands follow vLLM's own documentation, and AtlasForge's `openai` backend is tested end to
    end against a real HTTP server with a stand-in model — but **the two have never been put
    together on N-ATLaS**. Whether vLLM serves this model's chat template unmodified is one of the
    open questions on the [status page](../help/status.md). Expect to adjust on first contact.

The alternative is `--backend local`, which loads the weights in the notebook process: no server,
but the model and the notebook compete for the same memory, and an 8B model in fp16 is about 16 GB
before activations. On a GPU runtime, quantise:

```bash
atlasforge eval my_data.jsonl --out runs/natlas --backend local --quantize 4bit
```

## 10. Fine-tune in a notebook

Fine-tuning is where a notebook is genuinely the right tool: it needs a GPU, it runs for a long
time, and you want the data checks and the loss in front of you.

Start from a GPU runtime and install the extra:

```text
%pip install -q "brainers-atlasforge[finetune]"
```

Write the config with a cell magic, so the file lands on the runtime's disk: make the cell's first
line `%%writefile finetune.yaml`, and the rest of the cell is the config.

```yaml
train_file: train.jsonl
eval_file: test.jsonl        # used only to refuse leakage; never trained on
output_dir: adapters/hausa-agri

lora_r: 16
lora_alpha: 32
learning_rate: 0.0002
num_epochs: 1
max_seq_len: 2048
```

Then dry-run before spending GPU time. It needs no GPU and no model: it validates the data, refuses
train/test leakage and prints the plan.

```bash
atlasforge finetune finetune.yaml --dry-run
```

After that, the run itself and the evaluation that decides whether it was worth anything — the
sequence is set out in full on [Fine-tune with QLoRA](fine-tune-with-qlora.md), and the base model
is evaluated first so there is something to compare against:

```bash
atlasforge finetune finetune.yaml
atlasforge eval test.jsonl --out runs/tuned --backend local --adapter adapters/hausa-agri
atlasforge compare test.jsonl --base runs/base --candidate runs/tuned --slice domain
```

!!! danger "The training run has never been executed"
    The recipe, its data checks and its planning are tested; **the training itself has not been run
    on a GPU anywhere**. Memory use, speed and results are unknown. Read
    [Fine-tune with QLoRA](fine-tune-with-qlora.md) before relying on this.

Two notebook-specific notes. **A free-tier T4 is short of the 24 GB the defaults are written for** —
if it runs out of memory, lower `max_seq_len` or `lora_r` and raise `grad_accum` to keep the
effective batch size the same. And **a disconnect does not resume a training run** the way it
resumes an evaluation: watch the runtime's idle timeout, and copy the finished adapter out before
the session ends.

## 11. Speech in a notebook

Decoding audio needs **ffmpeg**, which is a system package and is not on every runtime. Colab has
it; if it is missing, one cell adds it:

```text
!apt-get -qq install -y ffmpeg
```

Then upload the recordings — the folder icon in the sidebar, or `files.upload()` — and transcribe:

```bash
atlasforge transcribe clip.ogg --lang ha --out transcripts
```

The official speech models are Whisper-Small fine-tunes, so they run on a CPU, slowly. Clips longer
than the models' 30-second limit are split automatically, and `--silence-aware` moves each cut to a
nearby pause instead of a fixed grid. See [Transcribe speech](transcribe-speech.md).

A transcript run produces a run directory like any other, so the evaluation half works in a notebook
too: score the transcripts under both tone views and compare two recognisers exactly as you would
compare two language models.

## 12. Other notebook platforms

The rules do not change; the two knobs do. What you need to find on any platform is **how to install
into the running kernel** and **where secrets go**.

| Platform | Install | Secret | GPU |
|---|---|---|---|
| Colab | `%pip install` | Secrets sidebar, `google.colab.userdata` | T4 on the free tier, larger on paid |
| Kaggle | `%pip install` | Add-ons → Secrets, `kaggle_secrets` | Free GPU sessions, with a weekly quota |
| JupyterLab / VS Code, on your machine | `%pip install`, or the kernel's own environment | Your shell environment | Whatever the machine has |
| Vertex AI Workbench, SageMaker Studio | `%pip install` | The platform's secret manager | Configurable |
| Binder, Deepnote, Lightning | `%pip install` | The platform's environment settings | Usually none |

Two generic traps. On a hosted platform **the internet may be off by default** — Kaggle requires you
to switch it on per notebook, and without it every `pip install` and every model download fails in a
way that reads like a package error. And **the session is idle-timed out**; a long evaluation should
be writing to a run directory on persistent storage, never to scratch disk alone.

## 13. A notebook is a record, not a result

A number printed in a cell is not evidence of anything — it is a screenshot, and a screenshot cannot
be re-scored. What AtlasForge produces instead is a **run directory**: the manifest, every answer
with the model's raw output, and the report built from them. That directory is the artefact worth
keeping, attaching or committing. `report` and `compare` can rebuild every figure from it later, on
any machine, with no model.

Three habits make a notebook's output trustworthy:

- **Pass `--seed`** to `eval` and `compare`, so the sampling and the bootstrap are reproducible.
- **Keep the run directory.** Keep only the notebook and you keep the conclusion while losing the
  evidence for it.
- **Say what the numbers came from.** The quickstart notebook's data is **synthetic** — fixed rules,
  labelled `synthetic-demo/...` everywhere. It demonstrates the tool; it is not a measurement of
  N-ATLaS, and it must never be presented as one. The reports committed in
  [`examples/reports/`](https://github.com/Brainers-Labs/atlasforge/blob/main/examples/README.md)
  are that same synthetic run, for the same reason.

The verdicts themselves are built not to flatter: failed calls count as wrong, a slice under 30
examples is reported as *insufficient data* rather than given an answer, and every text metric
appears under both tone views. Reading the report is the subject of
[Compare two models](compare-two-models.md#3-read-it).

## 14. Troubleshooting in a notebook

| What you see | What it is | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'atlasforge'` right after installing | The install went to another interpreter, or a stale module is cached | Use `%pip`, not `!pip`; if it persists, restart the runtime |
| `atlasforge: command not found` in a cell | The kernel's environment is not first on `PATH` | `!python -m atlasforge doctor` — the same commands, run through the interpreter |
| A cell seems to do nothing | The line was Python, not a shell command | Prefix it: `!atlasforge ...` |
| `SyntaxError` on a line that works in a terminal | Same cause | Add the `!`, and keep `%` for `pip` only |
| `pip` re-downloads everything every session | The runtime is a new machine each time | Expected. Cache wheels on Drive, or install only the extra you need |
| `No space left on device` | The 8B weights are about 16 GB before caches | Free the scratch disk, or use a runtime with more of it |
| The session disconnected mid-evaluation | Idle timeout, or the runtime was recycled | Re-run the same `eval` command against the same `--out`; finished examples are skipped. Do not delete the run directory |
| `plain http is only allowed for localhost` | A remote server over an unencrypted connection | Use `https`, or `--allow-insecure-http` if you know the network is private |
| A model will not download | The Awarri licence is not accepted for *that* repository | Accept it on each model page; `atlasforge doctor` names the ones missing |
| A token appeared in the cell output | `%env` or `!export` echoed it | Revoke it, and use the platform's secret store |

Anything not in this table is on the [troubleshooting](../help/troubleshooting.md) page, which is
written for a terminal but applies unchanged — the errors come from the tool, not from the notebook.

## Where to go next

<div class="grid cards" markdown>

- **The full walkthrough**

    [Quickstart](../get-started/quickstart.md) — every step, with the real output and what to
    notice in it.

- **Your own data**

    [Datasets](../concepts/datasets.md) — the JSONL format, and the checks that catch a bad file
    before a model is involved.

- **A model of your own**

    [Evaluate a model](evaluate-a-model.md), then [Compare two models](compare-two-models.md).

- **A real GPU**

    [Serve N-ATLaS](serve-n-atlas.md) — vLLM, llama.cpp or Ollama behind one endpoint.

</div>
