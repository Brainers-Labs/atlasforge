"""Command-line interface. Thin over the Python API.

No ``from __future__ import annotations`` here on purpose: Typer reads the
annotations at runtime to build the CLI.
"""

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from atlasforge import __version__, api, bench, config
from atlasforge.asr.chunking import transcribe_long
from atlasforge.asr.models import asr_model_for
from atlasforge.backends.base import Backend
from atlasforge.backends.factory import DEFAULT_MODEL, build_backend
from atlasforge.cards import CardInfo, check_card, render_card
from atlasforge.compare.slices import BUILTIN_FIELDS, MIN_SLICE_N
from atlasforge.demo import BASE_NAME, DATASET_NAME, TUNED_NAME, build_demo
from atlasforge.doctor import Check, exit_code, run_checks
from atlasforge.errors import AtlasForgeError, ConfigError, ResourceError
from atlasforge.eval.dataset import TASKS, Task, load_dataset
from atlasforge.eval.runner import RESULTS_NAME, RunConfig
from atlasforge.eval.validate import validate_dataset
from atlasforge.finetune import describe, load_config, prepare_data, train
from atlasforge.render import print_comparison, print_score, print_validation
from atlasforge.types import GenParams, Message, parse_lang

app = typer.Typer(
    name="atlasforge",
    help="Run, evaluate, compare and fine-tune the official N-ATLaS models.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
dataset_app = typer.Typer(help="Dataset tools.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
bench_app = typer.Typer(help="Benchmark suite runners.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")

_STYLE = {"ok": "green", "warn": "yellow", "fail": "bold red"}
_LABEL = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}

# --- options shared by every command that talks to a model -------------------------------

BackendOpt = Annotated[
    str,
    typer.Option(
        "--backend",
        "-b",
        help="openai: any OpenAI-compatible server (vLLM, llama.cpp, Ollama). local: transformers.",
    ),
]
BaseUrlOpt = Annotated[
    str | None,
    typer.Option(
        "--base-url", envvar="ATLASFORGE_BASE_URL", help="Server URL, e.g. http://127.0.0.1:8000/v1"
    ),
]
ModelOpt = Annotated[str, typer.Option("--model", help="Model name or Hugging Face repo id.")]
QuantizeOpt = Annotated[
    str, typer.Option("--quantize", help="local backend only: none, 4bit or 8bit (NVIDIA GPU).")
]
AdapterOpt = Annotated[
    str | None,
    typer.Option(
        "--adapter",
        help="local backend only: a LoRA adapter (path or repo) applied on the base model.",
    ),
]
PenaltyOpt = Annotated[
    bool,
    typer.Option(
        "--send-repetition-penalty/--no-send-repetition-penalty",
        help="openai backend: send the model-card repetition_penalty. Disable if the server rejects it.",
    ),
]
DeviceOpt = Annotated[
    str, typer.Option("--device", help="local backend only: auto, cpu, cuda, mps.")
]
TimeoutOpt = Annotated[float, typer.Option("--timeout", help="Seconds to wait per request.")]
RetriesOpt = Annotated[
    int, typer.Option("--retries", help="Retries on 429/5xx/connection errors (openai backend).")
]
InsecureOpt = Annotated[
    bool, typer.Option("--allow-insecure-http", help="Permit plain http to a non-local server.")
]
TaskOpt = Annotated[str, typer.Option("--task", "-t", help=f"One of: {', '.join(TASKS)}.")]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"atlasforge {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
        ),
    ] = False,
) -> None:
    """AtlasForge command-line interface."""


def _task(value: str) -> Task:
    if value not in TASKS:
        raise ConfigError(f"Unknown task {value!r}.", hint=f"Use one of: {', '.join(TASKS)}.")
    return value


def _backend(
    name: str,
    *,
    base_url: str | None,
    model: str,
    quantize: str,
    device: str,
    adapter: str | None = None,
    send_repetition_penalty: bool = True,
    timeout: float,
    retries: int = 2,
    insecure: bool,
    asr_model: str | None = None,
    asr_models: Mapping[str, str] | None = None,
) -> Backend:
    return build_backend(
        name,
        base_url=base_url,
        model=model,
        quantize=quantize,
        device=device,
        adapter=adapter,
        send_repetition_penalty=send_repetition_penalty,
        timeout=timeout,
        retries=retries,
        allow_insecure_http=insecure,
        asr_model=asr_model,
        asr_models=asr_models,
    )


def _left_to_us(ctx: typer.Context, name: str) -> bool:
    """Whether the user left this option alone, so a project file may decide it.

    By name, not by identity: Typer vendors its own copy of Click, and its ``ParameterSource``
    is a *different* enum class from ``click.core.ParameterSource`` — a member of one is never
    the member of the other, so ``is`` against the installed Click's enum is quietly always
    false. The member's name is the same either way.
    """
    source = ctx.get_parameter_source(name)
    return source is not None and source.name == "DEFAULT"


def _config_values(ctx: typer.Context, names: tuple[str, ...]) -> dict[str, Any]:
    """What ``atlasforge.toml`` may supply for options the user did not set.

    The documented precedence is flag → environment variable → project file → default. Click
    records where each value came from, so this reads the file only where the answer is
    "nowhere": a value the user typed, and ``--base-url`` from ``ATLASFORGE_BASE_URL``, both
    arrive as something other than the default and are left alone.
    """
    # Not `load_config`: the fine-tuning YAML loader of that name is already imported here.
    project = config.load_config()
    return {
        name: project.values[name]
        for name in names
        if name in project.values and _left_to_us(ctx, name)
    }


def _err(message: str) -> None:
    typer.echo(message, err=True)


# --- doctor -------------------------------------------------------------------------------


def _render_checks(checks: list[Check]) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Detail")
    for check in checks:
        table.add_row(
            Text(_LABEL[check.status], style=_STYLE[check.status]),
            Text(check.name),
            Text(check.detail),
        )
    console.print(table)
    for check in checks:
        if check.hint and check.status != "ok":
            console.print(Text(f"{check.name}: {check.hint}", style="dim"), soft_wrap=True)


@app.command()
def doctor(
    json_output: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON.")
    ] = False,
) -> None:
    """Check Python, GPU, disk, ffmpeg, Hugging Face token and optional extras."""
    checks = run_checks()
    if json_output:
        typer.echo(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        _render_checks(checks)
    raise typer.Exit(exit_code(checks))


# --- run ----------------------------------------------------------------------------------


@app.command()
def run(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="The prompt, or - to read it from stdin.")],
    backend: BackendOpt = "openai",
    base_url: BaseUrlOpt = None,
    model: ModelOpt = DEFAULT_MODEL,
    quantize: QuantizeOpt = "none",
    adapter: AdapterOpt = None,
    send_repetition_penalty: PenaltyOpt = True,
    device: DeviceOpt = "auto",
    timeout: TimeoutOpt = 120.0,
    retries: RetriesOpt = 2,
    insecure: InsecureOpt = False,
    system: Annotated[str | None, typer.Option("--system", help="Optional system prompt.")] = None,
    temperature: Annotated[float, typer.Option(help="0 for greedy decoding.")] = 0.1,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate.")] = 1000,
    seed: Annotated[int | None, typer.Option(help="Sampling seed.")] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSON, not just the text.")
    ] = False,
) -> None:
    """Send one prompt to N-ATLaS and print the answer."""
    chosen = _config_values(
        ctx, ("backend", "base_url", "model", "quantize", "device", "timeout", "retries")
    )
    backend = chosen.get("backend", backend)
    base_url = chosen.get("base_url", base_url)
    model = chosen.get("model", model)
    quantize = chosen.get("quantize", quantize)
    device = chosen.get("device", device)
    timeout = chosen.get("timeout", timeout)
    retries = chosen.get("retries", retries)
    text = sys.stdin.read() if prompt == "-" else prompt
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    params = GenParams(temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)

    engine = _backend(
        backend,
        base_url=base_url,
        model=model,
        quantize=quantize,
        adapter=adapter,
        send_repetition_penalty=send_repetition_penalty,
        device=device,
        timeout=timeout,
        retries=retries,
        insecure=insecure,
    )
    try:
        generation = engine.generate(messages, params)
        info = engine.info()
    finally:
        engine.close()
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "text": generation.text,
                    "model": info.model,
                    "revision": info.revision,
                    "latency_ms": round(generation.latency_ms, 1),
                    "finish_reason": generation.finish_reason,
                    "request_id": generation.request_id,
                    "usage": asdict(generation.usage) if generation.usage else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        typer.echo(generation.text)


# --- transcribe ---------------------------------------------------------------------------


@app.command()
def transcribe(
    ctx: typer.Context,
    files: Annotated[
        list[Path], typer.Argument(help="Audio file(s): wav, mp3, m4a, ogg/opus, ...")
    ],
    lang: Annotated[str, typer.Option("--lang", "-l", help="ha, yo, ig or en.")],
    backend: BackendOpt = "openai",
    base_url: BaseUrlOpt = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="ASR model. Default: the official NCAIR1 model for --lang (e.g. NCAIR1/Hausa-ASR).",
        ),
    ] = None,
    device: DeviceOpt = "auto",
    timeout: TimeoutOpt = 120.0,
    retries: RetriesOpt = 2,
    insecure: InsecureOpt = False,
    silence_aware: Annotated[
        bool,
        typer.Option(
            "--silence-aware",
            help="Move each window boundary to the nearest pause instead of a fixed grid.",
        ),
    ] = False,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write JSONL here.")] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSONL, not plain text.")
    ] = False,
) -> None:
    """Transcribe audio with the official N-ATLaS ASR model. Long audio is split automatically."""
    chosen = _config_values(ctx, ("backend", "base_url", "model", "device", "timeout", "retries"))
    backend = chosen.get("backend", backend)
    base_url = chosen.get("base_url", base_url)
    model = chosen.get("model", model)
    device = chosen.get("device", device)
    timeout = chosen.get("timeout", timeout)
    retries = chosen.get("retries", retries)
    language = parse_lang(lang)
    asr_model = model or asr_model_for(language)
    engine = _backend(
        backend,
        base_url=base_url,
        model=asr_model,
        quantize="none",
        device=device,
        timeout=timeout,
        retries=retries,
        insecure=insecure,
        asr_model=asr_model,
        # A local backend routes by language from the official map; only an explicit
        # --model needs overriding there.
        asr_models={language: asr_model} if model else None,
    )
    records: list[dict[str, Any]] = []
    failed = 0
    try:
        for file in files:
            try:
                result = transcribe_long(engine, file, language, silence_aware=silence_aware)
            except AtlasForgeError as exc:
                failed += 1
                _err(f"error: {file}: {exc.format()}")
                continue
            records.append(
                {
                    "file": str(file),
                    "lang": language,
                    "text": result.text,
                    "latency_ms": round(result.latency_ms, 1),
                    "chunks": [
                        {"start_s": c.start_s, "end_s": c.end_s, "text": c.text}
                        for c in result.chunks
                    ],
                }
            )
    finally:
        engine.close()

    if out is not None:
        out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    elif json_output or len(files) > 1:
        for record in records:
            typer.echo(json.dumps(record, ensure_ascii=False))
    else:
        for record in records:
            typer.echo(record["text"])
    raise typer.Exit(1 if failed else 0)


# --- eval / report / compare --------------------------------------------------------------


def _write_reports(out: Path, dataset_path: Path, task: Task, metrics: list[str] | None) -> None:
    report = api.write_reports(out, dataset_path, task, metrics or None)
    print_score(Console(), report)
    typer.echo(f"Wrote {out / api.REPORT_MD}, {out / api.REPORT_JSON} and {out / api.REPORT_HTML}")


@app.command("eval")
def eval_command(
    ctx: typer.Context,
    dataset: Annotated[Path, typer.Argument(help="JSONL dataset.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Run directory (created; resumable).")],
    task: TaskOpt = "generation",
    backend: BackendOpt = "openai",
    base_url: BaseUrlOpt = None,
    model: ModelOpt = DEFAULT_MODEL,
    quantize: QuantizeOpt = "none",
    adapter: AdapterOpt = None,
    send_repetition_penalty: PenaltyOpt = True,
    device: DeviceOpt = "auto",
    timeout: TimeoutOpt = 120.0,
    retries: RetriesOpt = 2,
    insecure: InsecureOpt = False,
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            "-m",
            help="Metric name, or module:function for one of your own. Repeatable.",
        ),
    ] = None,
    concurrency: Annotated[int, typer.Option(help="Parallel requests (HTTP backends).")] = 1,
    lang: Annotated[
        str | None, typer.Option("--lang", "-l", help="Default language for ASR.")
    ] = None,
    temperature: Annotated[float, typer.Option(help="0 for greedy decoding.")] = 0.1,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate.")] = 1000,
    seed: Annotated[
        int | None, typer.Option(help="Sampling seed. Recorded in the run and its report.")
    ] = None,
    retry_errors: Annotated[bool, typer.Option(help="Re-run failed examples on resume.")] = True,
    max_consecutive_failures: Annotated[
        int,
        typer.Option(help="Stop after this many failures in a row (0 = never). Progress is kept."),
    ] = 20,
) -> None:
    """Run a dataset through a model, score it, and write report.md / report.json.

    Re-running with the same --out resumes: finished examples are skipped.
    """
    chosen = _config_values(
        ctx, ("backend", "base_url", "model", "quantize", "device", "timeout", "retries")
    )
    backend = chosen.get("backend", backend)
    base_url = chosen.get("base_url", base_url)
    model = chosen.get("model", model)
    quantize = chosen.get("quantize", quantize)
    device = chosen.get("device", device)
    timeout = chosen.get("timeout", timeout)
    retries = chosen.get("retries", retries)
    kind = _task(task)
    examples = load_dataset(dataset, kind)
    language = parse_lang(lang) if lang else None
    if kind == "asr" and language is not None and model == DEFAULT_MODEL:
        # An ASR task served over an endpoint should not be asked for the LLM by default.
        model = asr_model_for(language)
    engine = _backend(
        backend,
        base_url=base_url,
        model=model,
        quantize=quantize,
        adapter=adapter,
        send_repetition_penalty=send_repetition_penalty,
        device=device,
        timeout=timeout,
        retries=retries,
        insecure=insecure,
        asr_model=model if kind == "asr" else None,
    )
    config = RunConfig(
        gen_params=GenParams(temperature=temperature, max_new_tokens=max_new_tokens, seed=seed),
        lang=language,
        concurrency=concurrency,
        retry_errors=retry_errors,
        max_consecutive_failures=max_consecutive_failures,
    )
    columns = (
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    try:
        with Progress(*columns, console=Console(stderr=True), transient=True) as progress:
            bar = progress.add_task(f"{model}", total=len(examples))
            # The reports are written below, once the bar is gone; the run here is silent.
            evaluation = api.evaluate(
                examples,
                out_dir=out,
                task=kind,
                backend=engine,
                config=config,
                on_result=lambda _record: progress.advance(bar),
                write_report=False,
            )
    finally:
        engine.close()
    summary = evaluation.run

    if summary.skipped:
        typer.echo(f"Resumed: {summary.skipped} example(s) were already finished.")
    _write_reports(out, dataset, kind, metric)
    if summary.failed:
        _err(
            f"warning: {summary.failed} of {summary.total} example(s) failed; see {out / RESULTS_NAME}"
        )
    raise typer.Exit(1 if summary.failed == summary.total else 0)


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="A run directory made by `atlasforge eval`.")],
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="The dataset it was run on.")],
    task: TaskOpt = "generation",
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            "-m",
            help="Metric name, or module:function for one of your own. Repeatable.",
        ),
    ] = None,
) -> None:
    """Re-score a finished run (no model needed) and rewrite its report."""
    _write_reports(run_dir, dataset, _task(task), metric)


@app.command()
def compare(
    dataset: Annotated[Path, typer.Argument(help="The dataset both runs used.")],
    base: Annotated[Path, typer.Option("--base", help="Run directory of the base model.")],
    candidate: Annotated[Path, typer.Option("--candidate", help="Run directory of the candidate.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the report.")] = Path(
        "comparison"
    ),
    task: TaskOpt = "generation",
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            "-m",
            help="Metric name, or module:function for one of your own. Repeatable.",
        ),
    ] = None,
    primary: Annotated[
        str | None, typer.Option(help="Metric for slices, e.g. chrf@tone_aware.")
    ] = None,
    slice_by: Annotated[
        list[str] | None,
        typer.Option(
            "--slice",
            help=f"Slice field (repeatable). Built in: {', '.join(BUILTIN_FIELDS)}; or any meta key.",
        ),
    ] = None,
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 1000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
    min_slice_n: Annotated[
        int, typer.Option(help="Smaller slices are 'insufficient data'.")
    ] = MIN_SLICE_N,
) -> None:
    """Did the candidate actually beat the base, and where did it get worse?"""
    result = api.compare_runs(
        dataset,
        base,
        candidate,
        task=_task(task),
        metrics=metric or None,
        slice_fields=[*BUILTIN_FIELDS, *(slice_by or [])],
        primary=primary,
        n_boot=n_boot,
        seed=seed,
        min_slice_n=min_slice_n,
    )
    api.write_comparison(result, out)
    print_comparison(Console(), result)
    typer.echo(
        f"Wrote {out / api.COMPARISON_MD}, {out / api.COMPARISON_JSON} "
        f"and {out / api.COMPARISON_HTML}"
    )


# --- dataset validate ---------------------------------------------------------------------


@dataset_app.command("validate")
def dataset_validate(
    file: Annotated[Path, typer.Argument(help="JSONL dataset to check.")],
    task: TaskOpt = "generation",
    against: Annotated[
        Path | None,
        typer.Option("--against", help="Another split, to check for train/test leakage."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON.")
    ] = False,
) -> None:
    """Find malformed lines, duplicates, leakage, broken Unicode and stripped diacritics."""
    result = validate_dataset(file, _task(task), against=against)
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_validation(Console(), result)
    raise typer.Exit(0 if result.ok else 1)


# --- demo ---------------------------------------------------------------------------------


@app.command()
def demo(
    directory: Annotated[Path, typer.Argument(help="Folder to create.")] = Path("atlasforge-demo"),
) -> None:
    """Write synthetic demo data and sample runs, to try AtlasForge with no model at all.

    The answers come from fixed rules, not from N-ATLaS or any model.
    """
    out = build_demo(directory)
    data = out / DATASET_NAME
    base, tuned = out / "runs" / BASE_NAME, out / "runs" / TUNED_NAME
    typer.echo(f"Wrote synthetic demo data to {out} (NOT real N-ATLaS output).")
    typer.echo("Try these, in order:")
    typer.echo(f"  atlasforge dataset validate {data}")
    typer.echo(f"  atlasforge report {base} --dataset {data}")
    typer.echo(f"  atlasforge compare {data} --base {base} --candidate {tuned} --slice domain")


# --- finetune / card ----------------------------------------------------------------------


@app.command()
def finetune(
    config: Annotated[Path, typer.Argument(help="Fine-tuning config (.yaml or .json).")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate the data and show the plan; train nothing.")
    ] = False,
) -> None:
    """Fine-tune N-ATLaS with QLoRA and save a LoRA adapter (needs an NVIDIA GPU).

    --dry-run needs no GPU: it checks the data (including train/test leakage) and prints the plan.
    """
    settings = load_config(config)
    if dry_run:
        typer.echo(describe(settings, prepare_data(settings)))
        typer.echo("Dry run: nothing was trained.")
        raise typer.Exit(0)
    result = train(settings)
    typer.echo(f"Adapter saved to {result.output_dir} ({result.n_train_examples} examples).")
    typer.echo("Next:")
    typer.echo(
        f"  atlasforge eval DATA --backend local --adapter {result.output_dir} --out runs/tuned"
    )
    typer.echo("  atlasforge compare DATA --base runs/base --candidate runs/tuned")
    typer.echo(
        f"  atlasforge card --training-run {Path(result.output_dir) / 'training_run.json'} ..."
    )


def _text_or_file(value: str) -> str:
    """``@path`` reads the text from a file; anything else is used as given."""
    if not value.startswith("@"):
        return value
    path = Path(value[1:])
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc.strerror or exc}") from exc


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read {path} as JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object.")
    return data


@app.command()
def card(
    name: Annotated[
        str, typer.Option("--name", help="Model name. 'Powered by Awarri' is added if missing.")
    ],
    description: Annotated[str, typer.Option("--description", help="What the adapter does.")],
    training_data: Annotated[
        str,
        typer.Option(
            "--training-data",
            help="Where the data came from AND its licence. Start with @ to read a file.",
        ),
    ],
    lang: Annotated[
        list[str] | None, typer.Option("--lang", "-l", help="ha, yo, ig, en. Repeatable.")
    ] = None,
    domain: Annotated[str | None, typer.Option(help="e.g. agriculture")] = None,
    author: Annotated[str | None, typer.Option(help="Who made this.")] = None,
    adapter_repo: Annotated[
        str | None, typer.Option(help="Hugging Face repo of the adapter.")
    ] = None,
    intended_use: Annotated[str | None, typer.Option(help="What it is meant for.")] = None,
    comparison: Annotated[
        Path | None, typer.Option(help="comparison.json from `atlasforge compare`.")
    ] = None,
    training_run: Annotated[
        Path | None, typer.Option(help="training_run.json from `atlasforge finetune`.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the card.")] = Path(
        "MODEL_CARD.md"
    ),
) -> None:
    """Write a licence-aware model card (attribution, 'Powered by Awarri', user cap) for an adapter."""
    info = CardInfo(
        name=name,
        description=_text_or_file(description),
        training_data=_text_or_file(training_data),
        languages=tuple(parse_lang(item) for item in (lang or [])),
        domain=domain,
        author=author,
        adapter_repo=adapter_repo,
        intended_use=intended_use,
        comparison=_read_json(comparison),
        training_run=_read_json(training_run),
    )
    out.write_text(render_card(info), encoding="utf-8")
    for warning in check_card(info):
        _err(f"warning: {warning}")
    typer.echo(f"Wrote {out}")


# --- bench afrobench ------------------------------------------------------------------------


@bench_app.command("afrobench")
def bench_afrobench(
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Directory for bench.json and bench.md.")
    ] = Path("benches/afrobench"),
    model: ModelOpt = DEFAULT_MODEL,
    revision: Annotated[
        str | None, typer.Option("--revision", help="Model revision (commit SHA) to pin.")
    ] = None,
    tasks: Annotated[
        list[str] | None,
        typer.Option("--tasks", help="Exact harness task names. Repeatable. Overrides discovery."),
    ] = None,
    list_tasks: Annotated[
        bool,
        typer.Option("--list", help="Print the harness tasks matching the suite; run nothing."),
    ] = False,
    batch_size: Annotated[
        str, typer.Option("--batch-size", help="lm-eval batch size, e.g. 8 or auto.")
    ] = "auto",
    few_shot: Annotated[
        int | None, typer.Option("--few-shot", help="Number of few-shot examples per task.")
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="lm-eval device, e.g. cuda:0 or cpu.")
    ] = None,
    chat_template: Annotated[
        bool, typer.Option("--chat-template", help="Apply the tokenizer's chat template.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the harness command; run nothing.")
    ] = False,
) -> None:
    """Run the published AfroBench-LITE suite through lm-evaluation-harness.

    Tasks come from the installed harness, not from AtlasForge, and the study's published
    figures are never copied into the report. Needs the `bench` extra.
    """
    if list_tasks:
        found = bench.discover_tasks(_harness_task_names())
        for name in found:
            typer.echo(name)
        for family in bench.missing_families(found):
            _err(f"no harness task matches the {family!r} family")
        raise typer.Exit(0 if found else 1)

    chosen = tuple(tasks) if tasks else bench.discover_tasks(_harness_task_names())
    if not chosen:
        raise ResourceError(
            f"The installed harness has no task matching the {bench.SUITE} families.",
            hint="Run `atlasforge bench afrobench --list`, or pass --tasks yourself.",
        )
    command = bench.build_command(
        model=model,
        tasks=chosen,
        output_path=out / bench.RAW_NAME,
        revision=revision,
        batch_size=batch_size,
        few_shot=few_shot,
        device=device,
        apply_chat_template=chat_template,
    )
    if dry_run:
        typer.echo(" ".join(command))
        typer.echo("Dry run: nothing was run.")
        raise typer.Exit(0)

    report = bench.run_afrobench(
        out,
        model=model,
        revision=revision,
        tasks=chosen,
        batch_size=batch_size,
        few_shot=few_shot,
        device=device,
        apply_chat_template=chat_template,
    )
    for task in report.tasks:
        shown = bench.fmt_metric(task.primary_metric, task.primary)
        typer.echo(f"{task.task}: {shown} ({task.primary_metric or 'no metric'})")
    for family in report.missing_families:
        _err(f"warning: no harness task matched {family!r}, so it is not in this report")
    typer.echo(f"Wrote {out / bench.RESULTS_NAME} and {out / bench.REPORT_NAME}")


def _harness_task_names() -> tuple[str, ...]:
    """Every task the installed harness offers, or a hint to install it."""
    try:
        return bench.harness_tasks()
    except ImportError as exc:
        raise ResourceError(
            "lm-evaluation-harness is not installed, so there are no tasks to run.",
            hint=f"Install it with: {bench.INSTALL_HINT}",
        ) from exc


def main() -> None:
    """Console-script entry point: turn AtlasForge errors into clean output."""
    try:
        app()
    except AtlasForgeError as exc:
        typer.echo(f"error: {exc.format()}", err=True)
        raise SystemExit(2) from exc
