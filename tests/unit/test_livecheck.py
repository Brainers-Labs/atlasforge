"""The evidence file is a submission artefact, so what it claims is tested like anything else.

The probes that need a GPU are not exercised here — nothing on this machine can. What is tested is
the part that decides what the file *says*: that a check which fails is recorded as failed rather
than as a pass, that a check which cannot run says so, and that a credential cannot reach the page.

Nothing here may run a probe. ``live_steps()`` builds the eight steps without executing them, and
the script is driven with its globals replaced, so a test can never start a model download.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from atlasforge.livecheck import (
    CHECKS,
    PROBES,
    REPOS,
    Evidence,
    Outcome,
    Row,
    Sample,
    Step,
    _access_step,
    _asr_step,
    _audio_for,
    _server_step,
    _timestamps_step,
    _vram_used,
    collect,
    environment_facts,
    environment_secrets,
    evidence_filename,
    gpu_facts,
    hub_revisions,
    live_steps,
    redact,
    render,
)
from atlasforge.types import LANGS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    from atlasforge.doctor import Access

FIXED = datetime(2026, 10, 7, 12, 30, tzinfo=timezone.utc)
FAKE_TOKEN = "hf_" + "a" * 30
ROOT = Path(__file__).resolve().parents[2]


def build(
    *steps: Step, shas: Mapping[str, str] | None = None, secrets: Sequence[str] = ()
) -> Evidence:
    """Collect a report from steps the test chose, with the clock and the Hub pinned."""
    revisions = dict.fromkeys(REPOS, "abc123") if shas is None else shas
    return collect(
        steps,
        shas=revisions,
        facts={"python": "3.12.0"},
        secrets=secrets,
        now=lambda: FIXED,
    )


def passing(check: str, result: str = "fine") -> Step:
    return Step(check, lambda: Outcome(result))


def run_step(step: Step) -> Row:
    """Run one built step and hand back its row, so a guard can be read without a whole report."""
    return collect([step], shas={"repo": "abc"}).rows[0]


def load_script() -> ModuleType:
    """Import ``scripts/live_smoke.py`` without running ``main``."""
    spec = importlib.util.spec_from_file_location("live_smoke", ROOT / "scripts" / "live_smoke.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script with every probe replaced, so calling ``main`` touches nothing."""
    module = load_script()
    monkeypatch.setattr(module, "live_steps", lambda **_kwargs: [passing(CHECKS[0], "all 5")])
    monkeypatch.setattr(module, "hub_revisions", lambda: {"repo": "abc123"})
    monkeypatch.setattr(module, "environment_facts", lambda: {"python": "3.12.0"})
    monkeypatch.setattr(module, "gpu_facts", lambda: {"gpu": "none detected"})
    return module


class TestRedaction:
    def test_a_known_secret_is_masked_by_its_value(self) -> None:
        masked = redact(f"Authorization: Bearer {FAKE_TOKEN}", [FAKE_TOKEN])
        assert masked == "Authorization: Bearer hf_****aaaa"

    def test_a_token_shaped_value_is_masked_even_when_nobody_named_it(self) -> None:
        """The backstop. An exception can echo a header from a variable we never read."""
        assert "a" * 30 not in redact(f"token {FAKE_TOKEN} rejected")

    def test_an_hf_prefix_too_short_to_be_a_token_is_left_alone(self) -> None:
        assert redact("hf_hub is the client") == "hf_hub is the client"

    def test_ordinary_text_is_untouched(self) -> None:
        assert redact("VRAM 15.2 GB used") == "VRAM 15.2 GB used"

    def test_an_empty_secret_is_ignored_rather_than_replaced_everywhere(self) -> None:
        assert redact("abc", [""]) == "abc"


class TestEnvironment:
    def test_it_reads_the_credentials_this_process_holds(self) -> None:
        env = {"HF_TOKEN": FAKE_TOKEN, "ATLASFORGE_API_KEY": "sk-abc", "PATH": "/usr/bin"}
        assert environment_secrets(env) == [FAKE_TOKEN, "sk-abc"]

    def test_a_missing_variable_is_not_an_empty_secret(self) -> None:
        assert environment_secrets({"PATH": "/usr/bin"}) == []

    def test_the_facts_name_the_packages_and_never_import_them(self) -> None:
        facts = environment_facts(version=lambda name: f"{name}-1.2", which=lambda _: "/usr/bin/x")
        assert facts["torch"] == "torch-1.2"
        assert facts["ffmpeg"] == "/usr/bin/x"
        assert facts["atlasforge"]

    def test_a_missing_binary_says_not_found_rather_than_blank(self) -> None:
        assert environment_facts(which=lambda _: None)["ffmpeg"] == "not found"

    def test_gpus_are_named_with_their_memory(self) -> None:
        assert gpu_facts(lambda: [("NVIDIA A100", 79.3)]) == {"gpu": "NVIDIA A100 (79 GB)"}

    def test_no_gpu_is_stated_rather_than_left_empty(self) -> None:
        assert gpu_facts(list) == {"gpu": "none detected"}

    def test_no_torch_says_not_checked_instead_of_none_detected(self) -> None:
        def missing() -> list[tuple[str, float]]:
            raise ImportError("No module named 'torch'")

        assert gpu_facts(missing)["gpu"].startswith("not checked")

    def test_a_driver_failure_is_recorded_with_its_exception(self) -> None:
        def broken() -> list[tuple[str, float]]:
            raise RuntimeError("driver mismatch")

        assert gpu_facts(broken) == {"gpu": "unknown (RuntimeError: driver mismatch)"}


class TestTheGuards:
    """The parts of the probe section that decide *what a row says*, without touching a model.

    Every probe below ``live_steps`` is split in two for exactly this: a guard that is testable
    here, and a ``# pragma: no cover`` body that needs the hardware. These drive the guards.
    """

    def test_no_base_url_says_so_rather_than_connecting_to_nothing(self) -> None:
        row = run_step(_server_step(None))
        assert row.status == "not run"
        assert "--base-url" in row.result

    def test_no_audio_directory_says_so_rather_than_reporting_an_empty_transcript(self) -> None:
        for build in (_asr_step, _timestamps_step):
            row = run_step(build(None))
            assert row.status == "not run"
            assert "--audio-dir" in row.result

    def test_a_directory_that_is_not_there_is_treated_as_no_audio(self, tmp_path: Path) -> None:
        missing = tmp_path / "not-created"
        assert run_step(_asr_step(missing)).status == "not run"
        assert run_step(_timestamps_step(missing)).status == "not run"

    def test_a_directory_with_no_clips_names_the_missing_thing(self, tmp_path: Path) -> None:
        row = run_step(_timestamps_step(tmp_path))
        assert row.status == "not run"
        assert "no clip found" in row.result

    def test_a_clip_is_found_whatever_its_container(self, tmp_path: Path) -> None:
        for lang, suffix in (("ha", ".wav"), ("yo", ".ogg"), ("ig", ".m4a"), ("en", ".mp3")):
            (tmp_path / f"{lang}{suffix}").write_bytes(b"RIFF")
        clip = _audio_for(tmp_path, "ha")
        assert clip is not None
        assert clip.name == "ha.wav"
        end = _audio_for(tmp_path, "en")
        assert end is not None
        assert end.name == "en.mp3"

    def test_an_absent_language_is_none_rather_than_a_guess_at_a_filename(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "ha.wav").write_bytes(b"RIFF")
        assert _audio_for(tmp_path, "yo") is None

    def test_a_file_named_like_a_directory_is_not_a_clip(self, tmp_path: Path) -> None:
        """The guard reads ``is_file``, so a directory called ``ha.wav`` cannot be handed to ASR."""
        (tmp_path / "ha.wav").mkdir()
        assert _audio_for(tmp_path, "ha") is None


class TestTheGuardsDelegate:
    """When the guard *does* have what it needs, it hands it to the probe unchanged.

    The probe itself is replaced here — it needs a server or the weights — but the argument it is
    given is the guard's whole job, so that is what these check.
    """

    def test_a_base_url_is_passed_to_the_server_probe_with_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, str]] = []

        def probe(base_url: str, model: str) -> Outcome:
            seen.append((base_url, model))
            return Outcome(f"{base_url} answered")

        monkeypatch.setattr("atlasforge.livecheck._probe_server", probe)
        row = run_step(_server_step("http://127.0.0.1:8000/v1", "some/model"))
        assert seen == [("http://127.0.0.1:8000/v1", "some/model")]
        assert row.result == "http://127.0.0.1:8000/v1 answered"

    def test_an_audio_directory_is_passed_to_the_asr_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Path] = []

        def probe(directory: Path) -> Outcome:
            seen.append(directory)
            return Outcome("transcribed")

        monkeypatch.setattr("atlasforge.livecheck._probe_asr", probe)
        assert run_step(_asr_step(tmp_path)).result == "transcribed"
        assert seen == [tmp_path]

    def test_the_first_clip_found_is_the_one_the_timestamp_probe_gets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "yo.ogg").write_bytes(b"OggS")
        seen: list[Path] = []

        def probe(clip: Path) -> Outcome:
            seen.append(clip)
            return Outcome("chunks")

        monkeypatch.setattr("atlasforge.livecheck._probe_timestamps", probe)
        assert run_step(_timestamps_step(tmp_path)).result == "chunks"
        assert [clip.name for clip in seen] == ["yo.ogg"]


class TestAccessGuard:
    def test_all_five_granted_verifies_the_row(self) -> None:
        row = run_step(_access_step(["a", "b", "c", "d", "e"], access=lambda _: "granted"))
        assert row == Row(CHECKS[0], "all 5 accessible", "verified")

    def test_a_single_refusal_fails_the_row_and_names_the_repository(self) -> None:
        def access(model: str) -> Access:
            return "granted" if model != "b" else "denied"

        row = run_step(_access_step(["a", "b", "c"], access=access))
        assert row.status == "failed"
        assert row.result == "licence not accepted on b"

    def test_cannot_tell_is_not_run_and_is_never_reported_as_success(self) -> None:
        def access(model: str) -> Access:
            return "granted" if model != "b" else "unknown"

        row = run_step(_access_step(["a", "b", "c"], access=access))
        assert row.status == "not run"
        assert row.result == "could not check 1 of 3"

    def test_a_refusal_outranks_an_unknown_when_both_are_present(self) -> None:
        """A definite no is the more useful row: it says which page to open."""
        answers: dict[str, Access] = {"a": "denied", "b": "unknown"}
        row = run_step(_access_step(["a", "b"], access=lambda m: answers[m]))
        assert row.status == "failed"
        assert "a" in row.result

    def test_it_asks_about_every_repository_it_was_given(self) -> None:
        asked: list[str] = []

        def access(model: str) -> Access:
            asked.append(model)
            return "granted"

        _access_step(["x", "y"], access=access).run()
        assert asked == ["x", "y"]


class TestRevisions:
    class Info:
        def __init__(self, sha: str) -> None:
            self.sha = sha

    def test_a_repository_resolves_to_its_commit(self) -> None:
        class Api:
            def model_info(self, model: str) -> object:
                return TestRevisions.Info("d34db33f")

        assert hub_revisions(["a"], api=Api) == {"a": "d34db33f"}

    def test_a_commit_that_could_not_be_read_names_the_exception(self) -> None:
        class Api:
            def model_info(self, model: str) -> object:
                raise ConnectionError("offline")

        assert hub_revisions(["a"], api=Api)["a"] == "unknown (ConnectionError)"

    def test_an_empty_commit_is_an_empty_string_rather_than_none(self) -> None:
        class Api:
            def model_info(self, model: str) -> object:
                return TestRevisions.Info("")

        assert hub_revisions(["a"], api=Api)["a"] == ""

    def test_one_unreachable_repository_does_not_lose_the_others(self) -> None:
        class Api:
            def model_info(self, model: str) -> object:
                if model == "b":
                    raise TimeoutError("slow")
                return TestRevisions.Info("abc")

        assert hub_revisions(["a", "b", "c"], api=Api) == {
            "a": "abc",
            "b": "unknown (TimeoutError)",
            "c": "abc",
        }


class TestVram:
    class Cuda:
        def __init__(self, *, available: bool, peak: int) -> None:
            self._available = available
            self._peak = peak

        def is_available(self) -> bool:
            return self._available

        def max_memory_allocated(self) -> int:
            return self._peak

    def test_peak_is_reported_in_gibibytes_not_bytes(self) -> None:
        torch = SimpleNamespace(cuda=TestVram.Cuda(available=True, peak=16 * 1024**3))
        assert _vram_used(torch) == 16.0

    def test_a_cpu_only_machine_returns_none_so_the_row_claims_no_number(self) -> None:
        torch = SimpleNamespace(cuda=TestVram.Cuda(available=False, peak=1))
        assert _vram_used(torch) is None

    def test_torch_without_cuda_at_all_is_none_rather_than_an_attribute_error(self) -> None:
        assert _vram_used(SimpleNamespace()) is None


class TestTheLogItself:
    def test_the_checks_are_the_eight_the_planning_log_lists(self) -> None:
        assert len(CHECKS) == 8
        assert CHECKS[0].startswith("Gated access")
        assert CHECKS[-1].startswith("Commit SHAs")

    def test_every_repository_the_licence_must_be_accepted_on_is_covered(self) -> None:
        assert len(REPOS) == 5
        assert len(set(REPOS)) == 5

    def test_there_is_a_probe_sentence_for_every_language_and_no_others(self) -> None:
        assert tuple(PROBES) == LANGS

    def test_building_the_steps_does_not_run_them(self) -> None:
        """The seven model checks are constructed, never executed: no GPU, no download, no token."""
        assert [step.check for step in live_steps()] == list(CHECKS[:7])

    def test_the_sixth_check_is_the_one_that_needs_audio(self) -> None:
        assert live_steps()[5].check == CHECKS[5]


class TestCollect:
    def test_a_step_that_returns_is_verified(self) -> None:
        evidence = build(passing(CHECKS[0]))
        assert evidence.rows[0] == Row(CHECKS[0], "fine", "verified")
        assert evidence.verified == 2  # the step, and the derived SHA row

    def test_a_step_that_raises_becomes_a_failed_row_with_the_exception_words(self) -> None:
        def explode() -> Outcome:
            raise ConnectionError("hub unreachable")

        evidence = build(Step(CHECKS[0], explode))
        assert evidence.rows[0].status == "failed"
        assert evidence.rows[0].result == "ConnectionError: hub unreachable"

    def test_one_broken_step_does_not_stop_the_others(self) -> None:
        def explode() -> Outcome:
            raise TimeoutError("slow")

        evidence = build(Step(CHECKS[0], explode), passing(CHECKS[1]))
        assert [row.status for row in evidence.rows] == ["failed", "verified", "verified"]

    def test_a_step_that_could_not_run_keeps_its_own_status(self) -> None:
        step = Step(CHECKS[4], lambda: Outcome("no --base-url given", "not run"))
        assert build(step).rows[0].status == "not run"

    def test_a_secret_in_a_failure_message_never_reaches_the_page(self) -> None:
        def explode() -> Outcome:
            raise ConnectionError(f"401 from the hub with token {FAKE_TOKEN}")

        page = render(build(Step(CHECKS[0], explode), secrets=(FAKE_TOKEN,)))
        assert FAKE_TOKEN not in page
        assert "hf_****" in page

    def test_a_secret_in_model_output_never_reaches_the_page(self) -> None:
        sample = Sample("Hausa (ha)", "prompt", f"the key is {FAKE_TOKEN}", 1.0)
        step = Step(CHECKS[2], lambda: Outcome("ok", sample=sample))
        assert FAKE_TOKEN not in render(build(step, secrets=(FAKE_TOKEN,)))

    def test_samples_are_kept_in_order(self) -> None:
        first = Sample("Hausa", "menene?", "Abuja", 2.5)
        second = Sample("Yoruba", "kini?", "Abuja", 3.5)
        evidence = build(
            Step(CHECKS[2], lambda: Outcome("two", sample=first)),
            Step(CHECKS[5], lambda: Outcome("one", sample=second)),
        )
        assert [sample.what for sample in evidence.samples] == ["Hausa", "Yoruba"]

    def test_an_unresolved_commit_sha_marks_the_row_not_run(self) -> None:
        evidence = build(shas={"a": "", "b": "abc"})
        assert evidence.rows[-1].status == "not run"
        assert "a" in evidence.rows[-1].result

    def test_a_revision_that_could_not_be_read_counts_as_unknown(self) -> None:
        assert build(shas={"a": "unknown (ConnectionError)"}).rows[-1].status == "not run"

    def test_all_revisions_resolved_verifies_the_row(self) -> None:
        assert build(shas={"a": "abc"}).rows[-1].result == "1 repositories, all resolved"

    def test_the_timestamp_comes_from_the_injected_clock(self) -> None:
        assert build().generated_at.startswith("2026-10-07T12:30")

    def test_outcome_defaults_to_verified_so_a_probe_must_opt_in_to_doubt(self) -> None:
        assert Outcome("ok").status == "verified"


class TestRender:
    def test_the_file_names_the_check_in_the_planning_logs_own_words(self) -> None:
        page = render(build(passing(CHECKS[0])))
        assert CHECKS[0] in page
        assert "Verification log" in page

    def test_it_says_how_many_checks_verified(self) -> None:
        assert "2 of 2 checks verified" in render(build(passing(CHECKS[0])))

    def test_it_states_that_nothing_was_estimated(self) -> None:
        page = render(build())
        assert "Nothing here is estimated" in page
        assert "not a user's data" in page

    def test_the_machine_section_is_rendered(self) -> None:
        assert "| python | 3.12.0 |" in render(build())

    def test_every_repository_is_listed_with_its_licence_link(self) -> None:
        page = render(build())
        for repo in REPOS:
            assert f"https://huggingface.co/{repo}" in page

    def test_a_sample_is_quoted_with_its_timing(self) -> None:
        sample = Sample("Hausa (ha)", "Menene?", "Abuja", 2.5)
        page = render(build(Step(CHECKS[2], lambda: Outcome("ok", sample=sample))))
        assert "- Prompt: Menene?" in page
        assert "- Output: Abuja" in page
        assert "- Seconds: 2.50" in page

    def test_a_run_with_no_samples_omits_the_section(self) -> None:
        assert "## Sample output" not in render(build())

    def test_the_file_ends_with_a_newline(self) -> None:
        assert render(build()).endswith("\n")


class TestTheScript:
    def test_the_filename_is_one_file_per_day(self) -> None:
        assert evidence_filename(FIXED) == "live_2026-10-07.md"

    def test_it_writes_the_file_it_promises(self, script: ModuleType, tmp_path: Path) -> None:
        assert script.main(["--out", str(tmp_path)]) == 0
        written = tmp_path / evidence_filename()
        assert written.is_file()
        page = written.read_text(encoding="utf-8")
        assert "# Live evidence" in page
        # The fixture's one step passed and its one revision resolved, so both rows verified.
        assert "2 of 2 checks verified" in page

    def test_a_run_that_could_not_read_the_hub_still_writes_the_file(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreachable() -> dict[str, str]:
            raise ConnectionError("offline")

        monkeypatch.setattr(script, "hub_revisions", unreachable)
        assert script.main(["--out", str(tmp_path)]) == 0
        assert (tmp_path / evidence_filename()).is_file()

    def test_it_creates_the_output_directory(self, script: ModuleType, tmp_path: Path) -> None:
        assert script.main(["--out", str(tmp_path / "nested" / "evidence")]) == 0
        assert (tmp_path / "nested" / "evidence" / evidence_filename()).is_file()

    def test_importing_it_pulls_in_no_heavy_framework(self) -> None:
        """The core install has no torch, so this must stay importable on a laptop."""
        code = (
            "import sys, atlasforge.livecheck; "
            "sys.exit(1 if {'torch', 'transformers'} & set(sys.modules) else 0)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr

    def test_help_works_without_a_gpu_which_is_how_a_reader_learns_the_flags(self) -> None:
        # No `encoding=`, deliberately: the child writes with the platform default, so the
        # parent has to read with it too. Pinning UTF-8 here decodes cp1252 on Windows, the
        # reader thread dies with UnicodeDecodeError, `stdout` arrives as None and the
        # assertion below is a TypeError rather than a comparison. It passes on Linux either
        # way, which is why only the Windows runner caught it.
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "live_smoke.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        for flag in ("--out", "--quantize", "--base-url", "--audio-dir"):
            assert flag in result.stdout


def test_evidence_counts_only_verified_rows() -> None:
    evidence = Evidence(
        generated_at="2026-10-07T00:00:00+00:00",
        facts={},
        shas={},
        rows=[Row("a", "", "verified"), Row("b", "", "failed"), Row("c", "", "not run")],
    )
    assert evidence.verified == 1


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_is_a_plain_string_a_reader_can_grep_for(check: str) -> None:
    assert check.strip() == check
    assert "\n" not in check
