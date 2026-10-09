"""MkDocs macros: documentation fragments computed from the real code.

Reference material (CLI flags, metrics, config defaults, errors, file formats) and every
terminal transcript in the docs are produced here from the running program, so the docs
cannot disagree with the tool. If a metric or setting is added without a description below,
the build fails on purpose.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import typer.main
from typer.testing import CliRunner

import atlasforge.errors as errors_module
from atlasforge import __version__, config
from atlasforge.cli import app
from atlasforge.eval.flags import DESCRIPTIONS, FLAG_NAMES
from atlasforge.eval.normalize import normalize, tone_aware, tone_insensitive
from atlasforge.eval.score import KNOWN_METRICS, LOWER_IS_BETTER, TASK_DEFAULTS
from atlasforge.finetune.config import QLoRAConfig

_METRIC_DOCS = {
    "exact_match": (
        "fraction",
        "1 if the normalised answer equals the normalised reference, else 0.",
    ),
    "accuracy": (
        "fraction",
        "Classification only. Which label the answer names (earliest whole-word match) equals the reference label.",
    ),
    "macro_f1": (
        "fraction",
        "Classification only. F1 averaged over the classes in the references. Pooled: no per-example value, so no confidence interval.",
    ),
    "accuracy_strict": (
        "fraction",
        "Classification only. Like `accuracy`, but the whole answer must *be* a label -- `not positive` matches nothing instead of `positive`.",
    ),
    "macro_f1_strict": (
        "fraction",
        "Classification only. `macro_f1` with the whole-answer matching of `accuracy_strict`. Pooled only.",
    ),
    "chrf": (
        "0-100",
        "Character n-gram F-score (sacrebleu chrF). Good for generation and translation.",
    ),
    "chrf++": ("0-100", "chrF plus word n-grams (word order 2)."),
    "wer": (
        "fraction",
        "Word error rate. Lower is better. Pooled WER divides total word errors by total reference words.",
    ),
    "cer": ("fraction", "Character error rate. Lower is better."),
}

_CONFIG_DOCS = {
    "backend": "`openai` for any OpenAI-compatible server, or `local` for transformers.",
    "base_url": "Default for `--base-url`, e.g. `http://127.0.0.1:8000/v1`.",
    "model": "Default for `--model`: the NCAIR1 model to run.",
    "quantize": "`local` backend only: `none`, `4bit` or `8bit`.",
    "device": "`local` backend only: `auto`, `cpu`, `cuda` or `mps`.",
    "timeout": "Seconds to wait per request.",
    "retries": "Retries on 429/5xx/connection errors.",
}

_SETTING_DOCS = {
    "train_file": "JSONL training data (the same format as evaluation; every example needs a `reference`). **Required.**",
    "output_dir": "Where the LoRA adapter and `training_run.json` are written. **Required.**",
    "eval_file": "A held-out file. Used only to refuse train/test leakage; never trained on.",
    "base_model": "Hugging Face repo of the model to fine-tune.",
    "revision": "Pin the base model to a specific commit for reproducibility.",
    "lora_r": "LoRA rank. Higher means more capacity and more memory.",
    "lora_alpha": "LoRA scaling factor.",
    "lora_dropout": "Dropout on the LoRA layers.",
    "target_modules": "Which projection layers get adapters (Llama-style names).",
    "learning_rate": "Peak learning rate.",
    "num_epochs": "Passes over the training data.",
    "batch_size": "Examples per device per step.",
    "grad_accum": "Steps accumulated before an update. Effective batch = batch_size x grad_accum.",
    "max_seq_len": "Longest training sequence in tokens. Sequences over it are cut off; the run is refused if more than 10% would be.",
    "warmup_ratio": "Fraction of steps used for learning-rate warm-up.",
    "logging_steps": "Log the loss every N steps.",
    "seed": "Random seed.",
    "quantize": "`4bit` (QLoRA, needs an NVIDIA GPU) or `none` (plain LoRA).",
    "min_examples": "Refuse to train on fewer examples than this.",
}

_runner = CliRunner()


# ---------------------------------------------------------------- the demo, run for real


@lru_cache(maxsize=1)
def _demo_transcripts() -> dict[str, str]:
    """Run the quickstart commands for real and keep their output and files."""
    previous_cwd, previous_columns = Path.cwd(), os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "104"
    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            data = "atlasforge-demo/toy_qa.jsonl"
            base, tuned = "atlasforge-demo/runs/base", "atlasforge-demo/runs/tuned"
            commands = {
                "demo": ["demo", "atlasforge-demo"],
                "validate": ["dataset", "validate", data],
                "report": ["report", base, "--dataset", data],
                "compare": [
                    "compare",
                    data,
                    "--base",
                    base,
                    "--candidate",
                    tuned,
                    "--slice",
                    "domain",
                ],
            }
            for name, argv in commands.items():
                outcome = _runner.invoke(app, argv)
                if outcome.exit_code != 0:
                    raise RuntimeError(f"docs demo command failed: {argv}\n{outcome.output}")
                results[name] = _clean(outcome.output)
            results["report_md"] = Path(base, "report.md").read_text(encoding="utf-8")
            results["comparison_md"] = Path("comparison", "comparison.md").read_text(
                encoding="utf-8"
            )
            results["results_head"] = "".join(
                Path(base, "results.jsonl").read_text(encoding="utf-8").splitlines(True)[:2]
            )
            results["run_json"] = Path(base, "run.json").read_text(encoding="utf-8")
            results["report_json"] = Path(base, "report.json").read_text(encoding="utf-8")
            results["comparison_json"] = Path("comparison", "comparison.json").read_text(
                encoding="utf-8"
            )
            results["report_html"] = Path(base, "report.html").read_text(encoding="utf-8")
            results["dataset_head"] = "".join(
                Path(data).read_text(encoding="utf-8").splitlines(True)[:2]
            )

            # A real fine-tune dry run, on a disjoint train/test split of the demo data.
            rows = Path(data).read_text(encoding="utf-8").splitlines(True)
            Path("train.jsonl").write_text("".join(rows[:70]), encoding="utf-8")
            Path("test.jsonl").write_text("".join(rows[70:]), encoding="utf-8")
            # JSON, not YAML, so building the docs needs no PyYAML. The output is identical.
            Path("finetune.json").write_text(
                json.dumps(
                    {
                        "train_file": "train.jsonl",
                        "eval_file": "test.jsonl",
                        "output_dir": "adapters/demo",
                    }
                ),
                encoding="utf-8",
            )
            outcome = _runner.invoke(app, ["finetune", "finetune.json", "--dry-run"])
            if outcome.exit_code != 0:
                raise RuntimeError(f"docs dry-run failed:\n{outcome.output}")
            results["dry_run"] = _clean(outcome.output)
        finally:
            os.chdir(previous_cwd)
            if previous_columns is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = previous_columns
    return results


def _clean(output: str) -> str:
    """Trailing spaces off, and forward slashes, so a transcript looks the same on every OS."""
    return "\n".join(line.rstrip() for line in output.splitlines()).strip().replace("\\", "/")


def _demote(markdown: str, levels: int = 3) -> str:
    """Push every heading down so an embedded report fits under a page's own headings."""
    return re.sub(
        r"^(#+) ", lambda m: "#" * (len(m.group(1)) + levels) + " ", markdown, flags=re.MULTILINE
    )


def _json_excerpt(text: str) -> str:
    """Pretty JSON, with long lists cut to a few items so a page stays readable."""
    data = json.loads(text)

    def trim(value: Any, depth: int = 0) -> Any:
        if isinstance(value, list):
            shown = [trim(v, depth + 1) for v in value[:2]]
            return shown + (["..."] if len(value) > 2 else [])
        if isinstance(value, dict):
            items = list(value.items())
            limit = 6 if depth >= 1 else 40
            trimmed = {k: trim(v, depth + 1) for k, v in items[:limit]}
            if len(items) > limit:
                trimmed["..."] = f"({len(items) - limit} more)"
            return trimmed
        return value

    return json.dumps(trim(data), indent=2, ensure_ascii=False)


# ------------------------------------------------------------------- CLI reference


_FLAG = re.compile(r"(?<![`\w-])(--[A-Za-z][\w-]*)")


def _code_flags(text: str) -> str:
    """Wrap --flag in backticks so Markdown typography cannot turn the hyphens into a dash."""
    return _FLAG.sub(r"`\1`", text)


def _option_row(param: Any) -> str:
    names = " / ".join(f"`{n}`" for n in [*param.opts, *getattr(param, "secondary_opts", [])])
    kind = "flag" if getattr(param, "is_flag", False) else param.type.name.lower()
    if getattr(param, "multiple", False):
        kind += " (repeatable)"
    default = param.default
    shown = (
        "required" if param.required else ("" if default in (None, False, ()) else f"`{default}`")
    )
    envvar = f" Env: `{param.envvar}`." if param.envvar else ""
    raw_help = (getattr(param, "help", "") or "").replace("\n", " ")
    help_text = _code_flags(raw_help).replace("|", "\\|")
    return f"| {names} | {kind} | {shown} | {help_text}{envvar} |"


def _command_markdown(command: Any, path: str) -> str:
    lines = [f"### `{path}`", ""]
    if command.help:
        lines += [_code_flags(inspect.cleandoc(command.help)), ""]
    arguments = [p for p in command.params if p.param_type_name == "argument"]
    options = [
        p for p in command.params if p.param_type_name == "option" and "--help" not in p.opts
    ]
    usage = " ".join(
        [path, *(f"<{a.name}>" for a in arguments), "[OPTIONS]" if options else ""]
    ).strip()
    lines += ["```bash", usage, "```", ""]
    if arguments:
        lines += ["| Argument | Required | Description |", "|---|---|---|"]
        for a in arguments:
            lines.append(
                f"| `{a.name}` | {'yes' if a.required else 'no'} | {(getattr(a, 'help', '') or '').replace('|', chr(92) + '|')} |"
            )
        lines.append("")
    if options:
        lines += ["| Option | Type | Default | Description |", "|---|---|---|---|"]
        lines += [_option_row(o) for o in options]
        lines.append("")
    return "\n".join(lines)


def _walk(command: Any, path: str) -> list[str]:
    if getattr(command, "commands", None):
        out: list[str] = []
        for name in command.commands:
            out += _walk(command.commands[name], f"{path} {name}")
        return out
    return [_command_markdown(command, path)]


# ----------------------------------------------------------------------- the macros


def define_env(env: Any) -> None:
    """Entry point called by mkdocs-macros."""
    env.variables["version"] = __version__

    @env.macro
    def transcript(name: str) -> str:
        """A real terminal transcript from the demo, as a fenced block."""
        return f"```text\n{_demo_transcripts()[name]}\n```"

    @env.macro
    def sample_report(kind: str) -> str:
        """A real generated Markdown report, with headings pushed down."""
        key = {"report": "report_md", "comparison": "comparison_md"}[kind]
        return _demote(_demo_transcripts()[key])

    @env.macro
    def file_example(name: str) -> str:
        """A real file written by the tool, trimmed for reading."""
        t = _demo_transcripts()
        if name == "dataset":
            return f"```json\n{t['dataset_head'].rstrip()}\n```"
        if name == "results":
            return f"```json\n{t['results_head'].rstrip()}\n```"
        if name == "run":
            return f"```json\n{_json_excerpt(t['run_json'])}\n```"
        if name == "report":
            return f"```json\n{_json_excerpt(t['report_json'])}\n```"
        if name == "comparison":
            return f"```json\n{_json_excerpt(t['comparison_json'])}\n```"
        if name == "report_html":
            # The head of the real page: the whole file is a stylesheet plus inline SVG, which
            # would not read as an example, so the docs show the part that states what it is.
            head = t["report_html"].split("<style>", 1)[0].rstrip()
            return f"```html\n{head}\n```"
        raise KeyError(name)

    @env.macro
    def normalize_demo() -> str:
        """A table of real normaliser output under both tone views."""
        samples = (
            ("Ẹ káàárọ̀", None),
            ("Ṣé o wà dáadáa?", "yo"),
            ("Don’t stop!", "en"),
            ("’Yar ɓarawo", "ha"),
            ("’Yar", "en"),
            ("  Well-known,   facts.  ", None),
        )
        rows = ["| Input | Language | Tone-aware | Tone-insensitive |", "|---|---|---|---|"]
        for text, lang in samples:
            aware = normalize(text, tone_aware(lang))  # type: ignore[arg-type]
            insensitive = normalize(text, tone_insensitive(lang))  # type: ignore[arg-type]
            rows.append(f"| `{text}` | {lang or '-'} | `{aware}` | `{insensitive}` |")
        return "\n".join(rows)

    @env.macro
    def cli_reference() -> str:
        """Every command, argument and option, read from the Typer app."""
        root = typer.main.get_command(app)
        return "\n".join(_walk(root, "atlasforge"))

    @env.macro
    def metrics_table() -> str:
        """All metrics, checked against the registry so none is left undocumented."""
        missing = KNOWN_METRICS - set(_METRIC_DOCS)
        extra = set(_METRIC_DOCS) - KNOWN_METRICS
        if missing or extra:
            raise RuntimeError(
                f"metric docs out of sync: missing={sorted(missing)} extra={sorted(extra)}"
            )
        rows = ["| Metric | Scale | Better | Description |", "|---|---|---|---|"]
        for name in sorted(_METRIC_DOCS):
            scale, text = _METRIC_DOCS[name]
            better = "lower" if name in LOWER_IS_BETTER else "higher"
            rows.append(f"| `{name}` | {scale} | {better} | {text} |")
        defaults = ["", "Defaults when `--metric` is not given:", ""]
        defaults += [
            f"- `{task}`: {', '.join(f'`{m}`' for m in metrics)}"
            for task, metrics in TASK_DEFAULTS.items()
        ]
        return "\n".join([*rows, *defaults])

    @env.macro
    def config_table() -> str:
        """Every `atlasforge.toml` key, checked against what the loader actually accepts."""
        if set(config.KEYS) != set(_CONFIG_DOCS):
            raise RuntimeError(
                "config docs out of sync: "
                f"missing={sorted(set(config.KEYS) - set(_CONFIG_DOCS))} "
                f"extra={sorted(set(_CONFIG_DOCS) - set(config.KEYS))}"
            )
        rows = ["| Key | Type | Meaning |", "|---|---|---|"]
        rows += [
            f"| `{key}` | {config.type_name(key)} | {_CONFIG_DOCS[key]} |"
            for key in sorted(config.KEYS)
        ]
        return "\n".join(rows)

    @env.macro
    def flags_table() -> str:
        """Every failure-mode flag, read from the registry so the list cannot drift."""
        if set(FLAG_NAMES) != set(DESCRIPTIONS):
            raise RuntimeError("a failure-mode flag has no description")
        rows = ["| Flag | What it means |", "|---|---|"]
        rows += [f"| `{name}` | {DESCRIPTIONS[name]} |" for name in FLAG_NAMES]
        return "\n".join(rows)

    @env.macro
    def fine_tune_settings() -> str:
        """Every fine-tuning setting with its real default."""
        fields = {f.name: f for f in dataclasses.fields(QLoRAConfig)}
        missing = set(fields) - set(_SETTING_DOCS)
        extra = set(_SETTING_DOCS) - set(fields)
        if missing or extra:
            raise RuntimeError(
                f"setting docs out of sync: missing={sorted(missing)} extra={sorted(extra)}"
            )
        rows = ["| Setting | Default | Description |", "|---|---|---|"]
        for name, f in fields.items():
            if f.default is not dataclasses.MISSING:
                default = (
                    f"`{f.default}`"
                    if not isinstance(f.default, tuple)
                    else "`" + ", ".join(f.default) + "`"
                )
            else:
                default = "*(required)*"
            rows.append(f"| `{name}` | {default} | {_SETTING_DOCS[name]} |")
        return "\n".join(rows)

    @env.macro
    def errors_table() -> str:
        """Every AtlasForge exception, with its parent and meaning."""
        rows = ["| Exception | Extends | Meaning |", "|---|---|---|"]
        for name, cls in inspect.getmembers(errors_module, inspect.isclass):
            if (
                issubclass(cls, errors_module.AtlasForgeError)
                and cls.__module__ == errors_module.__name__
            ):
                parent = (
                    cls.__bases__[0].__name__
                    if cls is not errors_module.AtlasForgeError
                    else "Exception"
                )
                summary = (inspect.getdoc(cls) or "").splitlines()[0]
                rows.append(f"| `{name}` | `{parent}` | {summary} |")
        return "\n".join(rows)
