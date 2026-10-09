"""Licence-aware model cards for fine-tuned N-ATLaS adapters.

The N-ATLaS licence attaches obligations to derivatives. This module writes them into the
card so they are not forgotten, and refuses to invent anything: training data, evaluation
numbers and hyper-parameters come only from what the author supplies or from files that
AtlasForge itself produced (``comparison.json``, ``training_run.json``).

The licence terms below were read from the base model's Hugging Face card (see
``planning/21_NATLAS_DISCOVERY.md``). That is a summary: **the licence text governs**, and the
card says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final

from atlasforge import __version__
from atlasforge.backends.factory import DEFAULT_MODEL
from atlasforge.eval.format import fmt_bound, fmt_delta, fmt_value

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

POWERED_BY: Final = "Powered by Awarri"
BASE_MODEL_URL: Final = "https://huggingface.co/NCAIR1/N-ATLaS"
_POOLED_ONLY: Final = "not tested (pooled metric)"

# Base-model limitations, as stated on its model card (VERIFIED, planning/21).
_BASE_LIMITATIONS: Final = (
    "bias, including dialect and accent bias",
    "limited handling of code-switching",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CardInfo:
    """Everything a card is built from. Nothing here is guessed."""

    name: str
    description: str
    training_data: str  # required: provenance AND licence of the data, in the author's words
    languages: Sequence[str] = ()
    domain: str | None = None
    base_model: str = DEFAULT_MODEL
    author: str | None = None
    adapter_repo: str | None = None
    intended_use: str | None = None
    comparison: Mapping[str, Any] | None = None  # parsed comparison.json
    training_run: Mapping[str, Any] | None = None  # parsed training_run.json
    generated_on: str | None = field(default=None)  # ISO date; today when omitted


def suggested_name(name: str) -> str:
    """``name`` with the required "Powered by Awarri" suffix, unless already present."""
    if POWERED_BY.lower() in name.lower():
        return name
    return f"{name} - {POWERED_BY}"


def check_card(info: CardInfo) -> list[str]:
    """Problems worth fixing before publishing. Never blocks; the author decides."""
    warnings: list[str] = []
    if POWERED_BY.lower() not in info.name.lower():
        warnings.append(
            f'The name does not include "{POWERED_BY}", which the base licence asks derivative '
            f'works to carry. Suggested: "{suggested_name(info.name)}".'
        )
    if not info.languages:
        warnings.append("No languages given; add --lang so users know what this covers.")
    if info.comparison is None:
        warnings.append(
            "No evaluation results: run `atlasforge compare` and pass --comparison, or the card "
            "will say none were recorded."
        )
    else:
        regressions = _regressions(info.comparison)
        if regressions:
            warnings.append(
                f"The comparison shows {len(regressions)} regression(s); they are listed in the "
                "card. Make sure you are comfortable publishing them (you should be)."
            )
    if not info.training_data.strip():
        warnings.append("training_data is empty; state where the data came from and its licence.")
    return warnings


def render_card(info: CardInfo) -> str:
    """The full model card as Markdown with Hugging Face front matter."""
    title = suggested_name(info.name)
    date = info.generated_on or datetime.now(timezone.utc).date().isoformat()
    sections = [
        _front_matter(info),
        f"# {title}",
        "",
        info.description.strip(),
        "",
        *_summary(info),
        *_licence(info),
        *_usage(info),
        "## Training data",
        "",
        info.training_data.strip(),
        "",
        *_training_details(info.training_run),
        *_evaluation(info.comparison),
        *_limitations(),
        (
            f"---\n*Card generated with AtlasForge {__version__} on {date}. "
            "Review it before publishing: it is only as accurate as what you supplied.*"
        ),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _front_matter(info: CardInfo) -> str:
    lines = [
        "---",
        "license: other",
        f"base_model: {info.base_model}",
        "library_name: peft",
        "tags:",
        "  - n-atlas",
        "  - nigerian-languages",
        "  - lora",
        "  - atlasforge",
    ]
    if info.languages:
        lines.append("language:")
        lines.extend(f"  - {lang}" for lang in info.languages)
    lines.append("---")
    return "\n".join(lines) + "\n"


def _summary(info: CardInfo) -> list[str]:
    rows = [f"- **Base model:** [{info.base_model}]({BASE_MODEL_URL})"]
    if info.languages:
        rows.append(f"- **Languages:** {', '.join(info.languages)}")
    if info.domain:
        rows.append(f"- **Domain:** {info.domain}")
    if info.author:
        rows.append(f"- **Author:** {info.author}")
    if info.intended_use:
        rows.append(f"- **Intended use:** {info.intended_use.strip()}")
    return [*rows, ""]


def _licence(info: CardInfo) -> list[str]:
    return [
        "## Licence and attribution",
        "",
        (
            f"This is a derivative of [{info.base_model}]({BASE_MODEL_URL}), developed by Awarri "
            "Technologies in partnership with the National Centre for Artificial Intelligence and "
            "Robotics (NCAIR) and NITDA, under the Federal Ministry of Communications, Innovation "
            "and Digital Economy of Nigeria."
        ),
        "",
        (
            "It is governed by the base model's custom *Open-Source Research and Innovation "
            "License*. As summarised from the base model card (**the licence text governs; read it "
            "before use**):"
        ),
        "",
        "- use is limited to organisations with no more than 1,000 active end-users;",
        "- commercial use requires a separate licensing agreement;",
        "- attribution to Awarri Technologies and the Federal Ministry of Communications is required;",
        f'- derivative works must carry the suffix "{POWERED_BY}".',
        "",
        f"**{POWERED_BY}.**",
        "",
        (
            "This repository contains only a LoRA adapter, not the base weights. Obtain the base "
            "model from Hugging Face and accept its licence yourself."
        ),
        "",
    ]


def _usage(info: CardInfo) -> list[str]:
    adapter = info.adapter_repo or "<path-or-repo-of-this-adapter>"
    return [
        "## Usage",
        "",
        "Load the base model, then apply the adapter (requires access to the gated base model):",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'tokenizer = AutoTokenizer.from_pretrained("{info.base_model}")',
        f'base = AutoModelForCausalLM.from_pretrained("{info.base_model}", device_map="auto")',
        f'model = PeftModel.from_pretrained(base, "{adapter}")',
        "```",
        "",
        "To evaluate it against the base model with AtlasForge:",
        "",
        "```bash",
        "atlasforge eval data.jsonl --backend local --adapter <adapter> --out runs/tuned",
        "atlasforge compare data.jsonl --base runs/base --candidate runs/tuned",
        "```",
        "",
    ]


def _training_details(run: Mapping[str, Any] | None) -> list[str]:
    if not run:
        return []
    labels = (
        ("method", "Method"),
        ("base_model", "Base model"),
        ("base_revision", "Base revision"),
        ("n_train_examples", "Training examples"),
        ("dataset_sha256", "Training file fingerprint (sha256)"),
        ("atlasforge_version", "AtlasForge version"),
    )
    rows = ["## Training details", "", "| Setting | Value |", "|---|---|"]
    for key, label in labels:
        value = run.get(key)
        if value not in (None, ""):
            text = f"`{str(value)[:16]}`" if key == "dataset_sha256" else str(value)
            rows.append(f"| {label} | {text} |")
    hyper = run.get("hyperparameters")
    if isinstance(hyper, dict):
        rows.extend(f"| {k} | {v} |" for k, v in sorted(hyper.items()))
    return [*rows, ""]


def _regressions(comparison: Mapping[str, Any]) -> list[str]:
    primary = str(comparison.get("primary") or "").split("@")[0]
    found = [
        f"{m['name']} ({m['view'].replace('_', '-')}) got worse overall"
        for m in comparison.get("metrics", [])
        if m.get("verdict") == "regressed"
    ]
    found.extend(
        f"{s['field']} = {s['value']} (n={s['n']}): {fmt_delta(primary, s.get('delta'))}"
        for s in comparison.get("slices", [])
        if s.get("status") == "regressed"
    )
    return found


def _evaluation(comparison: Mapping[str, Any] | None) -> list[str]:
    if comparison is None:
        return [
            "## Evaluation",
            "",
            "No evaluation results were recorded for this adapter.",
            "",
        ]
    base, cand = comparison.get("base", {}), comparison.get("candidate", {})
    lines = [
        "## Evaluation",
        "",
        (
            f"Compared with the base model on {comparison.get('n_total')} examples "
            f"(dataset fingerprint `{str(comparison.get('dataset_sha256', ''))[:16]}`). "
            f"Base: {base.get('model') or '?'}. Candidate: {cand.get('model') or '?'}."
        ),
        "",
        "| Metric | View | Base | Adapter | Delta | 95% CI | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in comparison.get("metrics", []):
        if m.get("verdict") == _POOLED_ONLY:
            continue
        name = m["name"]
        interval = (
            "-"
            if m.get("low") is None
            else f"[{fmt_bound(name, m['low'])}, {fmt_bound(name, m['high'])}]"
        )
        lines.append(
            f"| {name} | {m['view'].replace('_', '-')} | {fmt_value(name, m.get('base_mean'))} "
            f"| {fmt_value(name, m.get('candidate_mean'))} | {fmt_delta(name, m.get('delta'))} "
            f"| {interval} | {m['verdict']} |"
        )
    regressions = _regressions(comparison)
    lines += ["", "**Regressions**", ""]
    lines += [f"- {r}" for r in regressions] if regressions else ["None found."]
    lines += [
        "",
        (
            "These numbers describe this dataset only. They do not establish performance on other "
            "data, and small slices are reported as insufficient data rather than judged."
        ),
        "",
    ]
    return lines


def _limitations() -> list[str]:
    return [
        "## Limitations",
        "",
        "Inherited from the base model's card:",
        "",
        *(f"- {item}" for item in _BASE_LIMITATIONS),
        "",
        "The behaviour of this fine-tune outside the evaluation above has not been characterised.",
        "",
    ]
