"""Run the published AfroBench-LITE suite through ``lm-evaluation-harness``.

The suite is a research benchmark, and it already has a runner: EleutherAI's
``lm-evaluation-harness``. AtlasForge does not reimplement the tasks — it builds the
harness command line for an official ``NCAIR1`` model, records what came back, and
writes it down in the same shape as every other AtlasForge report, so the numbers sit
beside the ones from ``eval`` and ``compare``.

Two things this module deliberately does not do:

* **It does not hard-code task identifiers.** A harness task name carries a language
  suffix (``afrixnli_yo``, ``belebele_hau``) and the set changes between harness
  versions, so the names are *discovered* from the installed harness and filtered by
  family (:func:`discover_tasks`). A family that matches nothing is reported as a gap
  rather than silently skipped.
* **It does not carry the published figures.** The AfroBench-LITE study's numbers are
  someone else's measurements of a specific harness revision on a specific dataset
  revision; copying them here would make them look like ours. The report records what
  this machine measured, and ``planning/21_NATLAS_DISCOVERY.md`` keeps the study's
  relative gains with their source.

Like the local backend, this path is **tested against stand-ins and has never been run
against the harness or the real weights** — see ``docs/help/status.md``.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Final

from atlasforge.backends.factory import DEFAULT_MODEL
from atlasforge.errors import ConfigError, ResourceError
from atlasforge.eval.format import FRACTION_METRICS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SUITE: Final = "afrobench-lite"
MODULE: Final = "lm_eval"
INSTALL_HINT: Final = 'pip install "brainers-atlasforge[bench]"'

#: The families the published AfroBench-LITE study reports (``planning/21``). These are
#: matched as substrings of the harness's own task names, case-insensitively, because
#: the harness adds the language itself -- ``afrixnli`` covers ``afrixnli_yo`` and its
#: seven siblings at once.
FAMILIES: Final[tuple[str, ...]] = (
    "afrixnli",
    "belebele",
    "afrimmlu",
    "flores",
    "sib",
    "injongo",
    "afrimgsm",
)

#: Metric keys preferred as a task's headline figure, in this order. Matched against the
#: key with lm-eval's aggregation suffix removed, so ``acc,none`` and a bare ``acc`` both
#: land on ``acc``. The suffix ``none`` means "no grouping": the mean over examples.
PREFERRED_METRICS: Final[tuple[str, ...]] = (
    "acc_norm",
    "acc",
    "exact_match",
    "f1",
    "chrf",
    "bleu",
)

#: Ours. The raw harness output stays beside it under ``harness/``: the report is a
#: reading of that file, and a reading that cannot be checked against its source is a
#: claim rather than a record.
RESULTS_NAME: Final = "bench.json"
REPORT_NAME: Final = "bench.md"
RAW_NAME: Final = "harness/results.json"

#: Harness metric keys that are 0-1 fractions and so print as percentages; the suffix a
#: harness key carries (``acc,none``) is stripped before the lookup. ``chrf`` and ``bleu``
#: are deliberately absent: they are already on a 0-100 scale, and scaling them again is
#: exactly the bug the first run through a stand-in harness produced -- chrF 38.2 rendered
#: as ``3820.00``.
FRACTION_KEYS: Final = FRACTION_METRICS | {"acc", "acc_norm", "f1"}


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskResult:
    """One harness task: every figure it reported, and the one used as the headline."""

    task: str
    metrics: dict[str, float]
    primary_metric: str | None
    primary: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchReport:
    """What a harness run reported, plus enough about the run to make it checkable."""

    model: str
    revision: str | None
    tasks: tuple[TaskResult, ...]
    harness_version: str | None = None
    missing_families: tuple[str, ...] = ()
    suite: str = SUITE
    #: The ``--num_fewshot`` the run was asked for, or ``None`` when it was not overridden. The
    #: distinction matters and is not pedantry: without the flag each task runs with whatever the
    #: harness defaults to for it, which is not necessarily zero-shot, so a report that printed
    #: "0" here would be inventing a setting. Few-shot count changes what the figure means.
    few_shot: int | None = None
    #: Where the harness's file actually landed, relative to the report. Not always the path
    #: it was given -- some harness versions treat ``--output_path`` as a directory and invent
    #: a name under it -- so the report records the path it found rather than the one it asked
    #: for, which would point at a file that is not there.
    raw_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "model": self.model,
            "revision": self.revision,
            "harness_version": self.harness_version,
            "missing_families": list(self.missing_families),
            "few_shot": self.few_shot,
            "raw_path": self.raw_path,
            "tasks": [
                {
                    "task": t.task,
                    "primary_metric": t.primary_metric,
                    "primary": t.primary,
                    "metrics": t.metrics,
                }
                for t in self.tasks
            ],
        }


def discover_tasks(
    available: Sequence[str], *, families: Sequence[str] = FAMILIES
) -> tuple[str, ...]:
    """The harness's own task names that belong to the suite's families.

    Matching is a case-insensitive substring, so a harness that renames ``afrixnli_yo``
    to ``afrixnli-yo`` is still found, and one that adds ``afrimgsm_ha_xn`` finds it too.
    Sorted, so the report does not depend on the order the harness listed them in.
    """
    wanted = [f.lower() for f in families]
    return tuple(sorted({name for name in available if any(f in name.lower() for f in wanted)}))


def missing_families(
    found: Sequence[str], *, families: Sequence[str] = FAMILIES
) -> tuple[str, ...]:
    """Which families the harness has nothing for, so a gap is visible rather than implied."""
    lowered = [name.lower() for name in found]
    return tuple(f for f in families if not any(f.lower() in name for name in lowered))


def build_command(
    *,
    model: str,
    tasks: Sequence[str],
    output_path: Path,
    revision: str | None = None,
    batch_size: str = "auto",
    few_shot: int | None = None,
    device: str | None = None,
    apply_chat_template: bool = False,
    trust_remote_code: bool = True,
) -> list[str]:
    """The ``lm_eval`` argv for one run. No shell, no quoting, no guessing.

    ``-m lm_eval`` rather than the console script, so this works when the harness is
    installed into an environment whose ``bin`` is not on ``PATH`` — which is exactly
    where a script pip-installed next to AtlasForge ends up.
    """
    if not tasks:
        raise ConfigError(
            "No benchmark tasks were given.",
            hint="Pass --tasks with the harness's own task names.",
        )
    if revision is not None and not revision.strip():
        raise ConfigError("--revision was empty. Leave it out to use the default revision.")
    model_args = [f"pretrained={model}"]
    if revision:
        model_args.append(f"revision={revision}")
    if trust_remote_code:
        model_args.append("trust_remote_code=True")
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--model",
        "hf",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_path),
    ]
    if few_shot is not None:
        command.extend(["--num_fewshot", str(few_shot)])
    if device is not None:
        command.extend(["--device", device])
    if apply_chat_template:
        command.append("--apply_chat_template")
    return command


def parse_results(text: str, *, model: str, revision: str | None = None) -> BenchReport:
    """Read the JSON the harness wrote into a :class:`BenchReport`.

    Only numeric entries are kept: the harness mixes figures with strings such as
    ``"alias": "afrixnli_yo"`` in the same object, and a string is not a result.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"The harness output is not JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), Mapping):
        raise ConfigError(
            "The harness output has no 'results' object.",
            hint=(
                "This must be the file lm_eval writes under --output_path (results.json by "
                "default), not its log output."
            ),
        )
    results: Mapping[str, object] = payload["results"]
    tasks: list[TaskResult] = []
    for task, block in sorted(results.items()):
        metrics = {
            key: float(value)
            for key, value in (block if isinstance(block, Mapping) else {}).items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        metric, value = _primary(metrics)
        tasks.append(TaskResult(task=task, metrics=metrics, primary_metric=metric, primary=value))
    version = payload.get("lm_eval_version")
    return BenchReport(
        model=model,
        revision=revision,
        tasks=tuple(tasks),
        harness_version=version if isinstance(version, str) else None,
    )


def _primary(metrics: Mapping[str, float]) -> tuple[str | None, float | None]:
    """The headline figure for a task: the first preferred metric it actually reported.

    Falls back to the first key in sorted order, so a task whose only figure is something
    unanticipated still shows a number instead of a dash.
    """
    for wanted in PREFERRED_METRICS:
        for key in metrics:
            if key.split(",")[0].lower() == wanted:
                return key, metrics[key]
    if not metrics:
        return None, None
    first = min(metrics)
    return first, metrics[first]


def fmt_metric(metric: str | None, value: float | None) -> str:
    """One harness figure: ``33.61`` for accuracy-like metrics, ``38.2`` for chrF, ``-`` for none.

    The harness's own keys decide the scale, so a translation task and a multiple-choice task
    can sit in the same table without either being misread.
    """
    if value is None:
        return "-"
    name = (metric or "").split(",")[0].lower()
    return f"{value * 100:.2f}" if name in FRACTION_KEYS else f"{value:.2f}"


def report_markdown(report: BenchReport) -> str:
    """The human-readable half of the result, beside the JSON."""
    shots = (
        f"`{report.few_shot}`"
        if report.few_shot is not None
        else "the harness's own default for each task (not overridden)"
    )
    lines = [
        f"# {report.suite}",
        "",
        f"Model: `{report.model}`",
        f"Revision: `{report.revision or 'default'}`",
        f"Harness: `lm_evaluation_harness {report.harness_version or 'unknown'}`",
        f"Few-shot: {shots}",
        "",
        "Measured on this machine, with the task set the installed harness exposes. These are",
        "AtlasForge's own observations; the study's published figures are in `planning/21`, and",
        f"the harness's raw output is beside this file at `{report.raw_path or RAW_NAME}`.",
        "",
    ]
    if report.missing_families:
        lines += [
            f"**Not covered:** no harness task matched {', '.join(f'`{f}`' for f in report.missing_families)}.",
            "Those numbers are absent, not zero.",
            "",
        ]
    if not report.tasks:
        lines += ["No task reported a result.", ""]
        return "\n".join(lines) + "\n"
    lines += ["| Task | Metric | Value |", "|---|---|---|"]
    for task in report.tasks:
        lines.append(
            f"| `{task.task}` | `{task.primary_metric or '-'}` | "
            f"{fmt_metric(task.primary_metric, task.primary)} |"
        )
    lines += [
        "",
        "Accuracy-like metrics are shown as percentages; chrF and BLEU stay on the harness's own",
        "0-100 scale.",
        "",
    ]
    other = [
        (t.task, key, value)
        for t in report.tasks
        for key, value in sorted(t.metrics.items())
        if key != t.primary_metric
    ]
    if other:
        lines += [
            "## Every metric the harness reported, unscaled",
            "",
            "The headline table rescales the ones that are fractions; this one is the harness's",
            "own numbers, so a figure can be checked against the raw output without arithmetic.",
            "",
            "| Task | Metric | Value |",
            "|---|---|---|",
        ]
        lines += [f"| `{task}` | `{key}` | {value:.6g} |" for task, key, value in other]
        lines.append("")
    return "\n".join(lines) + "\n"


def write_bench(report: BenchReport, out_dir: str | Path) -> Path:
    """Write ``bench.json`` and ``bench.md`` into ``out_dir``, creating it if needed."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULTS_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    path = directory / REPORT_NAME
    path.write_text(report_markdown(report), encoding="utf-8")
    return path


def harness_version() -> str | None:  # pragma: no cover - needs lm-eval installed
    """The installed harness version, or ``None`` if it cannot be read."""
    try:
        return metadata.version("lm-eval")
    except metadata.PackageNotFoundError:
        return None


def harness_tasks() -> tuple[str, ...]:  # pragma: no cover - needs lm-eval installed
    """Every task name the installed harness offers.

    Imported through ``importlib`` rather than a ``from`` statement, so the optional extra
    is loaded when it is asked for — the same way the local backend reaches ``torch``. A
    missing harness raises ``ImportError``, which the CLI turns into an install hint.
    """
    manager = importlib.import_module("lm_eval.tasks").TaskManager()
    return tuple(manager.all_tasks)


def find_results(directory: Path, *, preferred: Path) -> Path | None:
    """The file the harness wrote, which is not always the path it was given.

    ``--output_path`` is treated as a directory by some harness versions, which then
    invent their own file name underneath it. Rather than depend on which, the exact
    path is tried first and the directory is searched after; nothing is guessed.
    """
    if preferred.is_file():
        return preferred
    found = sorted(path for path in directory.rglob("results*.json") if path.is_file())
    return found[-1] if found else None


def _run(command: Sequence[str]) -> int:  # pragma: no cover - starts a real harness
    completed = subprocess.run(command, check=False)  # noqa: S603 - argv built here, no shell
    return completed.returncode


def run_afrobench(
    out_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    revision: str | None = None,
    tasks: Sequence[str] | None = None,
    available: Callable[[], Sequence[str]] = harness_tasks,
    run: Callable[[Sequence[str]], int] = _run,
    version: Callable[[], str | None] = harness_version,
    batch_size: str = "auto",
    few_shot: int | None = None,
    device: str | None = None,
    apply_chat_template: bool = False,
    trust_remote_code: bool = True,
) -> BenchReport:
    """Run the suite and write the report. Returns what was written.

    ``tasks`` overrides the discovery: pass the exact harness task names when the
    installed harness spells them differently from the families in ``planning/21``.
    """
    directory = Path(out_dir)
    preferred = directory / RAW_NAME
    known = tuple(available())
    chosen = tuple(tasks) if tasks else discover_tasks(known)
    if not chosen:
        raise ResourceError(
            f"The installed harness has no task matching the {SUITE} families.",
            hint=(
                f"Check `python -m {MODULE} --tasks list`, then pass the names with --tasks. "
                f"Harness installed? {INSTALL_HINT}"
            ),
        )
    unknown = [name for name in chosen if tasks and name not in known]
    if tasks and unknown:
        raise ConfigError(
            f"The harness does not have {', '.join(repr(n) for n in unknown)}.",
            hint="`atlasforge bench afrobench --list` prints what it does have.",
        )
    command = build_command(
        model=model,
        tasks=chosen,
        output_path=preferred,
        revision=revision,
        batch_size=batch_size,
        few_shot=few_shot,
        device=device,
        apply_chat_template=apply_chat_template,
        trust_remote_code=trust_remote_code,
    )
    status = run(command)
    if status != 0:
        raise ResourceError(
            f"The harness exited with status {status}.",
            hint="Its own output above says why; nothing was written.",
        )
    written = find_results(directory, preferred=preferred)
    if written is None:
        raise ResourceError(
            f"The harness exited cleanly but wrote no results under {directory}.",
            hint="Look for --output_path in its output; it may have written elsewhere.",
        )
    report = parse_results(written.read_text(encoding="utf-8"), model=model, revision=revision)
    found = {t.task for t in report.tasks}
    report = BenchReport(
        model=report.model,
        revision=report.revision,
        tasks=report.tasks,
        harness_version=report.harness_version or version(),
        missing_families=missing_families(sorted(found)),
        few_shot=few_shot,
        raw_path=_relative(written, directory),
    )
    write_bench(report, directory)
    return report


def _relative(path: Path, directory: Path) -> str:
    """``path`` as the report should name it: relative to the report, POSIX separators.

    :func:`find_results` only ever returns a path built from ``directory`` itself, so this is a
    prefix strip and not a search. The separators are normalised because a report written on
    Windows is read on whatever machine the next person has, and ``harness\\results.json`` is not
    a path there.
    """
    return path.relative_to(directory).as_posix()


__all__ = [
    "FAMILIES",
    "RAW_NAME",
    "REPORT_NAME",
    "RESULTS_NAME",
    "SUITE",
    "BenchReport",
    "TaskResult",
    "build_command",
    "discover_tasks",
    "find_results",
    "fmt_metric",
    "missing_families",
    "parse_results",
    "report_markdown",
    "run_afrobench",
    "write_bench",
]
