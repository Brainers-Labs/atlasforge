"""Command-line interface. Thin over the Python API.

No ``from __future__ import annotations`` here on purpose: Typer reads the
annotations at runtime to build the CLI.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from atlasforge import __version__
from atlasforge.asr.chunking import LongAudioBackend, transcribe_long
from atlasforge.backends.base import Backend
from atlasforge.backends.factory import DEFAULT_MODEL, build_backend
from atlasforge.compare import compare_runs
from atlasforge.compare import to_markdown as comparison_markdown
from atlasforge.compare.slices import BUILTIN_FIELDS, MIN_SLICE_N
from atlasforge.doctor import Check, exit_code, run_checks
from atlasforge.errors import AtlasForgeError, ConfigError
from atlasforge.eval.dataset import TASKS, Task, load_dataset
from atlasforge.eval.report import to_markdown as score_markdown
from atlasforge.eval.runner import (
    RESULTS_NAME,
    RunConfig,
    read_manifest,
    read_results,
)
from atlasforge.eval.runner import run as run_dataset
from atlasforge.eval.score import score_run, write_report_json
from atlasforge.eval.validate import validate_dataset
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
    timeout: float,
    retries: int = 2,
    insecure: bool,
) -> Backend:
    return build_backend(
        name,
        base_url=base_url,
        model=model,
        quantize=quantize,
        device=device,
        timeout=timeout,
        retries=retries,
        allow_insecure_http=insecure,
    )


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
    prompt: Annotated[str, typer.Argument(help="The prompt, or - to read it from stdin.")],
    backend: BackendOpt = "openai",
    base_url: BaseUrlOpt = None,
    model: ModelOpt = DEFAULT_MODEL,
    quantize: QuantizeOpt = "none",
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
    files: Annotated[
        list[Path], typer.Argument(help="Audio file(s): wav, mp3, m4a, ogg/opus, ...")
    ],
    lang: Annotated[str, typer.Option("--lang", "-l", help="ha, yo, ig or en.")],
    backend: BackendOpt = "openai",
    base_url: BaseUrlOpt = None,
    model: ModelOpt = DEFAULT_MODEL,
    device: DeviceOpt = "auto",
    timeout: TimeoutOpt = 120.0,
    retries: RetriesOpt = 2,
    insecure: InsecureOpt = False,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write JSONL here.")] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSONL, not plain text.")
    ] = False,
) -> None:
    """Transcribe audio with the official N-ATLaS ASR model. Long audio is split automatically."""
    language = parse_lang(lang)
    engine = _backend(
        backend,
        base_url=base_url,
        model=model,
        quantize="none",
        device=device,
        timeout=timeout,
        retries=retries,
        insecure=insecure,
    )
    records: list[dict[str, Any]] = []
    failed = 0
    try:
        for file in files:
            try:
                result = transcribe_long(engine, file, language)
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
    dataset = load_dataset(dataset_path, task)
    manifest = read_manifest(out)
    report = score_run(dataset, read_results(out / RESULTS_NAME), metrics=metrics or None)
    write_report_json(report, out / "report.json")
    (out / "report.md").write_text(score_markdown(report, manifest), encoding="utf-8")
    print_score(Console(), report)
    typer.echo(f"Wrote {out / 'report.md'} and {out / 'report.json'}")


@app.command("eval")
def eval_command(
    dataset: Annotated[Path, typer.Argument(help="JSONL dataset.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Run directory (created; resumable).")],
    task: TaskOpt = "generation",
    backend: BackendOpt = "openai",
    base_url: BaseUrlOpt = None,
    model: ModelOpt = DEFAULT_MODEL,
    quantize: QuantizeOpt = "none",
    device: DeviceOpt = "auto",
    timeout: TimeoutOpt = 120.0,
    retries: RetriesOpt = 2,
    insecure: InsecureOpt = False,
    metric: Annotated[list[str] | None, typer.Option("--metric", "-m", help="Repeatable.")] = None,
    concurrency: Annotated[int, typer.Option(help="Parallel requests (HTTP backends).")] = 1,
    lang: Annotated[
        str | None, typer.Option("--lang", "-l", help="Default language for ASR.")
    ] = None,
    temperature: Annotated[float, typer.Option(help="0 for greedy decoding.")] = 0.1,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate.")] = 1000,
    retry_errors: Annotated[bool, typer.Option(help="Re-run failed examples on resume.")] = True,
    max_consecutive_failures: Annotated[
        int,
        typer.Option(help="Stop after this many failures in a row (0 = never). Progress is kept."),
    ] = 20,
) -> None:
    """Run a dataset through a model, score it, and write report.md / report.json.

    Re-running with the same --out resumes: finished examples are skipped.
    """
    kind = _task(task)
    examples = load_dataset(dataset, kind)
    engine = _backend(
        backend,
        base_url=base_url,
        model=model,
        quantize=quantize,
        device=device,
        timeout=timeout,
        retries=retries,
        insecure=insecure,
    )
    if kind == "asr":
        engine = LongAudioBackend(engine)
    config = RunConfig(
        gen_params=GenParams(temperature=temperature, max_new_tokens=max_new_tokens),
        lang=parse_lang(lang) if lang else None,
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
            summary = run_dataset(
                engine,
                examples,
                out,
                config=config,
                on_result=lambda _record: progress.advance(bar),
            )
    finally:
        engine.close()

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
    metric: Annotated[list[str] | None, typer.Option("--metric", "-m", help="Repeatable.")] = None,
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
    metric: Annotated[list[str] | None, typer.Option("--metric", "-m", help="Repeatable.")] = None,
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
    kind = _task(task)
    examples = load_dataset(dataset, kind)
    fields = [*BUILTIN_FIELDS, *(slice_by or [])]
    result = compare_runs(
        examples,
        base,
        candidate,
        metrics=metric or None,
        slice_fields=fields,
        primary=primary,
        n_boot=n_boot,
        seed=seed,
        min_slice_n=min_slice_n,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(comparison_markdown(result), encoding="utf-8")
    (out / "comparison.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_comparison(Console(), result)
    typer.echo(f"Wrote {out / 'comparison.md'} and {out / 'comparison.json'}")


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


def main() -> None:
    """Console-script entry point: turn AtlasForge errors into clean output."""
    try:
        app()
    except AtlasForgeError as exc:
        typer.echo(f"error: {exc.format()}", err=True)
        raise SystemExit(2) from exc
