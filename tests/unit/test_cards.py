import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from atlasforge.cards import POWERED_BY, CardInfo, check_card, render_card, suggested_name
from atlasforge.cli import app
from atlasforge.errors import ConfigError

COMPARISON: dict[str, Any] = {
    "task": "generation",
    "dataset_sha256": "abcdef0123456789" * 4,
    "n_total": 500,
    "primary": "exact_match@tone_aware",
    "base": {"model": "NCAIR1/N-ATLaS", "revision": "aaaa"},
    "candidate": {"model": "NCAIR1/N-ATLaS+my-adapter", "revision": "aaaa"},
    "metrics": [
        {
            "name": "exact_match",
            "view": "tone_aware",
            "base_mean": 0.412,
            "candidate_mean": 0.538,
            "delta": 0.126,
            "low": 0.08,
            "high": 0.17,
            "verdict": "improved",
        },
        {
            "name": "macro_f1",
            "view": "tone_aware",
            "verdict": "not tested (pooled metric)",
            "base_mean": None,
            "candidate_mean": None,
            "delta": None,
            "low": None,
            "high": None,
        },
    ],
    "slices": [
        {"field": "has_number", "value": "yes", "n": 60, "delta": -0.062, "status": "regressed"},
        {"field": "lang", "value": "ha", "n": 400, "delta": 0.15, "status": "improved"},
    ],
}


def info(**overrides: Any) -> CardInfo:
    base: dict[str, Any] = {
        "name": f"Hausa Agri {POWERED_BY}",
        "description": "A LoRA adapter for Hausa agriculture questions.",
        "training_data": "1,200 questions written by our team; released CC-BY-4.0.",
        "languages": ("ha",),
        "generated_on": "2026-10-05",
    }
    base.update(overrides)
    return CardInfo(**base)


class TestName:
    def test_suffix_added_when_missing(self) -> None:
        assert suggested_name("Hausa Agri") == f"Hausa Agri - {POWERED_BY}"

    def test_suffix_not_duplicated_and_case_insensitive(self) -> None:
        assert suggested_name("X powered by awarri") == "X powered by awarri"
        assert suggested_name(f"X - {POWERED_BY}") == f"X - {POWERED_BY}"

    def test_card_title_always_carries_the_suffix(self) -> None:
        text = render_card(info(name="Plain name"))
        assert f"# Plain name - {POWERED_BY}" in text


class TestLicenceSection:
    def test_all_four_obligations_are_stated(self) -> None:
        text = render_card(info())
        assert "1,000 active end-users" in text
        assert "separate licensing agreement" in text
        assert "Awarri Technologies" in text
        assert "Federal Ministry of Communications" in text
        assert f'"{POWERED_BY}"' in text

    def test_says_the_licence_text_governs(self) -> None:
        assert "licence text governs" in render_card(info())

    def test_says_no_base_weights_are_included(self) -> None:
        assert "only a LoRA adapter, not the base weights" in render_card(info())

    def test_links_the_base_model(self) -> None:
        assert "https://huggingface.co/NCAIR1/N-ATLaS" in render_card(info())


class TestFrontMatter:
    def test_hugging_face_metadata(self) -> None:
        text = render_card(info(languages=("ha", "yo")))
        head = text.split("---")[1]
        assert "license: other" in head
        assert "base_model: NCAIR1/N-ATLaS" in head
        assert "library_name: peft" in head
        assert "  - ha" in head
        assert "  - yo" in head

    def test_no_language_block_without_languages(self) -> None:
        assert "language:" not in render_card(info(languages=())).split("---")[1]


class TestContentComesOnlyFromInputs:
    def test_training_data_is_quoted_verbatim(self) -> None:
        text = render_card(info(training_data="Scraped from X, licence unknown."))
        assert "Scraped from X, licence unknown." in text

    def test_optional_sections_absent_when_not_given(self) -> None:
        text = render_card(info())
        assert "## Training details" not in text
        assert "**Domain:**" not in text
        assert "**Author:**" not in text
        assert "No evaluation results were recorded" in text

    def test_summary_fields_appear_when_given(self) -> None:
        text = render_card(info(domain="agriculture", author="Brainers Labs", intended_use="Q&A"))
        assert "**Domain:** agriculture" in text
        assert "**Author:** Brainers Labs" in text
        assert "**Intended use:** Q&A" in text

    def test_usage_snippet_uses_the_given_adapter_repo(self) -> None:
        assert 'PeftModel.from_pretrained(base, "me/hausa-agri")' in render_card(
            info(adapter_repo="me/hausa-agri")
        )

    def test_usage_snippet_has_a_placeholder_otherwise(self) -> None:
        assert "<path-or-repo-of-this-adapter>" in render_card(info())

    def test_generation_date_is_recorded(self) -> None:
        assert "on 2026-10-05" in render_card(info())

    def test_default_date_is_today(self) -> None:
        text = render_card(info(generated_on=None))
        assert f"on {datetime.now(timezone.utc).date().isoformat()}." in text


class TestTrainingDetails:
    def test_only_present_fields_are_listed(self) -> None:
        run = {
            "method": "qlora",
            "base_model": "NCAIR1/N-ATLaS",
            "n_train_examples": 1200,
            "dataset_sha256": "f" * 64,
            "hyperparameters": {"lora_r": 16, "learning_rate": 0.0002},
        }
        text = render_card(info(training_run=run))
        assert "| Method | qlora |" in text
        assert "| Training examples | 1200 |" in text
        assert f"`{'f' * 16}`" in text
        assert "| lora_r | 16 |" in text
        assert "Base revision" not in text

    def test_empty_run_adds_nothing(self) -> None:
        assert "## Training details" not in render_card(info(training_run={}))


class TestEvaluation:
    def test_table_from_a_comparison(self) -> None:
        text = render_card(info(comparison=COMPARISON))
        assert "500 examples" in text
        assert "`abcdef0123456789`" in text
        assert (
            "| exact_match | tone-aware | 41.2% | 53.8% | +12.6 pts | [+8.0, +17.0] | improved |"
            in text
        )
        assert "NCAIR1/N-ATLaS+my-adapter" in text

    def test_pooled_only_metrics_are_left_out_of_the_table(self) -> None:
        assert "macro_f1" not in render_card(info(comparison=COMPARISON))

    def test_regressions_are_published_not_hidden(self) -> None:
        text = render_card(info(comparison=COMPARISON))
        assert "**Regressions**" in text
        assert "has_number = yes (n=60): -6.2 pts" in text

    def test_no_regressions_says_so(self) -> None:
        clean = {**COMPARISON, "slices": [], "metrics": COMPARISON["metrics"][:1]}
        assert "None found." in render_card(info(comparison=clean))

    def test_limits_of_the_numbers_are_stated(self) -> None:
        assert "describe this dataset only" in render_card(info(comparison=COMPARISON))

    def test_metric_without_interval_renders_a_dash(self) -> None:
        odd = {
            **COMPARISON,
            "metrics": [{**COMPARISON["metrics"][0], "low": None, "high": None}],
            "slices": [],
        }
        assert "| - | improved |" in render_card(info(comparison=odd))


class TestLimitations:
    def test_inherited_limitations_listed(self) -> None:
        text = render_card(info())
        assert "dialect and accent bias" in text
        assert "code-switching" in text

    def test_does_not_claim_more_than_is_known(self) -> None:
        assert "has not been characterised" in render_card(info())


class TestChecks:
    def test_clean_card_has_no_warnings(self) -> None:
        assert check_card(info(comparison={**COMPARISON, "slices": []})) == []

    def test_missing_suffix_is_flagged_with_a_suggestion(self) -> None:
        (warning, *_rest) = check_card(info(name="Plain"))
        assert POWERED_BY in warning
        assert "Plain - " in warning

    def test_missing_evaluation_flagged(self) -> None:
        assert any("No evaluation results" in w for w in check_card(info()))

    def test_regressions_flagged(self) -> None:
        assert any("regression" in w for w in check_card(info(comparison=COMPARISON)))

    def test_missing_languages_flagged(self) -> None:
        assert any("No languages" in w for w in check_card(info(languages=())))

    def test_empty_training_data_flagged(self) -> None:
        assert any("training_data is empty" in w for w in check_card(info(training_data="  ")))


@pytest.mark.parametrize("missing", ["name", "description", "training_data"])
def test_required_fields_exist(missing: str) -> None:
    fields: dict[str, Any] = {"name": "n", "description": "d", "training_data": "t"}
    del fields[missing]
    with pytest.raises(TypeError):
        CardInfo(**fields)


runner = CliRunner()


class TestCardCommand:
    def args(self, out: Path, *extra: str) -> list[str]:
        return [
            "card", "--name", "Hausa Agri", "--description", "Answers farming questions.",
            "--training-data", "Our own 1,200 questions, CC-BY-4.0.", "--out", str(out), *extra,
        ]  # fmt: skip

    def test_writes_a_card_and_warns_about_the_missing_suffix(self, tmp_path: Path) -> None:
        out = tmp_path / "CARD.md"
        result = runner.invoke(app, self.args(out, "--lang", "ha"))
        assert result.exit_code == 0, result.output
        assert f"# Hausa Agri - {POWERED_BY}" in out.read_text(encoding="utf-8")
        assert "warning:" in result.output
        assert POWERED_BY in result.output
        assert f"Wrote {out}" in result.output

    def test_training_data_can_come_from_a_file(self, tmp_path: Path) -> None:
        notes = tmp_path / "data.txt"
        notes.write_text("Collected by volunteers; licence CC0.", encoding="utf-8")
        out = tmp_path / "CARD.md"
        args = [a if not a.startswith("Our own") else f"@{notes}" for a in self.args(out)]
        assert runner.invoke(app, args).exit_code == 0
        assert "licence CC0" in out.read_text(encoding="utf-8")

    def test_comparison_and_training_run_files_are_used(self, tmp_path: Path) -> None:
        comparison = tmp_path / "comparison.json"
        comparison.write_text(json.dumps(COMPARISON), encoding="utf-8")
        run = tmp_path / "training_run.json"
        run.write_text(json.dumps({"method": "qlora", "n_train_examples": 1200}), encoding="utf-8")
        out = tmp_path / "CARD.md"
        result = runner.invoke(
            app,
            self.args(
                out,
                "--comparison",
                str(comparison),
                "--training-run",
                str(run),
                "-l",
                "ha",
                "-l",
                "yo",
            ),
        )
        assert result.exit_code == 0, result.output
        text = out.read_text(encoding="utf-8")
        assert "| exact_match | tone-aware |" in text
        assert "| Training examples | 1200 |" in text
        assert "  - yo" in text
        assert "regression" in result.output  # the comparison has one: the CLI says so

    def test_bad_language_is_rejected(self, tmp_path: Path) -> None:
        result = runner.invoke(app, self.args(tmp_path / "c.md", "--lang", "fr"))
        assert isinstance(result.exception, ConfigError)

    def test_unreadable_or_non_object_json_is_a_config_error(self, tmp_path: Path) -> None:
        missing = runner.invoke(
            app, self.args(tmp_path / "c.md", "--comparison", str(tmp_path / "nope.json"))
        )
        assert isinstance(missing.exception, ConfigError)
        listy = tmp_path / "list.json"
        listy.write_text("[1]", encoding="utf-8")
        wrong = runner.invoke(app, self.args(tmp_path / "c.md", "--comparison", str(listy)))
        assert isinstance(wrong.exception, ConfigError)

    def test_missing_training_data_file(self, tmp_path: Path) -> None:
        args = [
            a if not a.startswith("Our own") else "@nope.txt" for a in self.args(tmp_path / "c.md")
        ]
        assert isinstance(runner.invoke(app, args).exception, ConfigError)
