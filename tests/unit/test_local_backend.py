"""LocalBackend wiring, tested against stand-in torch/transformers modules.

This proves the *plumbing* (prompt handling, generation settings, limits, error mapping,
lazy loading). It does not, and cannot, prove real model behaviour: that needs real
weights on a real machine.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from atlasforge.asr.audio import SAMPLE_RATE
from atlasforge.backends import Backend
from atlasforge.backends.local import LocalBackend
from atlasforge.errors import (
    AudioError,
    BackendError,
    ConfigError,
    ModelAccessError,
    ResourceError,
)
from atlasforge.types import GenParams

PROMPT_TOKENS = 10
LIMIT = 100


@contextmanager
def _inference_mode() -> Iterator[None]:
    yield


class Seq(list[int]):
    """A token sequence with a torch-like ``shape``."""

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def __getitem__(self, item: Any) -> Any:
        result = super().__getitem__(item)
        return Seq(result) if isinstance(item, slice) else result


class Ids:
    shape = (1, PROMPT_TOKENS)


class Encoded(dict[str, Any]):
    def to(self, _device: object) -> "Encoded":
        return self


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, chat_template: str | None = "template") -> None:
        self.chat_template = chat_template
        self.chat_calls: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Encoded:
        self.chat_calls.append(messages)
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["return_dict"] is True
        return Encoded(input_ids=Ids(), attention_mask=Ids())

    def decode(self, tokens: Seq, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return f"  decoded {len(tokens)} tokens  "


class FakeLLM:
    device = "cpu"

    def __init__(self, new_tokens: int = 5, error: Exception | None = None) -> None:
        self.config = SimpleNamespace(max_position_embeddings=LIMIT, _commit_hash="abc123def456")
        self.new_tokens = new_tokens
        self.error = error
        self.generate_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> list[Seq]:
        if self.error:
            raise self.error
        self.generate_kwargs = kwargs
        return [Seq(range(PROMPT_TOKENS + self.new_tokens))]

    def to(self, _device: object) -> "FakeLLM":
        return self


class Harness:
    """Installs fake torch + transformers and records how they were used."""

    def __init__(self, *, cuda: bool = False, mps: bool = False) -> None:
        self.tokenizer = FakeTokenizer()
        self.llm = FakeLLM()
        self.model_kwargs: dict[str, Any] = {}
        self.tokenizer_loads = 0
        self.model_loads = 0
        self.seeds: list[int] = []
        self.pipeline_calls: list[tuple[str, Any]] = []
        self.pipeline_inputs: list[Any] = []
        self.pipeline_result: Any = {"text": "  ina kwana  "}
        self.load_error: Exception | None = None

        torch = ModuleType("torch")
        torch.float16 = "fp16"  # type: ignore[attr-defined]
        torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
        torch.manual_seed = self.seeds.append  # type: ignore[attr-defined]
        torch.inference_mode = _inference_mode  # type: ignore[attr-defined]
        torch.cuda = SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None)  # type: ignore[attr-defined]
        torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))  # type: ignore[attr-defined]
        self.torch = torch

        outer = self

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(_model: str, **_kw: Any) -> FakeTokenizer:
                outer.tokenizer_loads += 1
                return outer.tokenizer

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(_model: str, **kw: Any) -> FakeLLM:
                if outer.load_error:
                    raise outer.load_error
                outer.model_loads += 1
                outer.model_kwargs = kw
                return outer.llm

        def pipeline(task: str, **kw: Any) -> Any:
            outer.pipeline_calls.append((task, kw))

            def run(inputs: Any) -> Any:
                outer.pipeline_inputs.append(inputs)
                return outer.pipeline_result

            return run

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
        transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
        transformers.BitsAndBytesConfig = SimpleNamespace  # type: ignore[attr-defined]
        transformers.pipeline = pipeline  # type: ignore[attr-defined]
        self.transformers = transformers

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", self.torch)
        monkeypatch.setitem(sys.modules, "transformers", self.transformers)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Harness:
    harness = Harness()
    harness.install(monkeypatch)
    return harness


MESSAGES = [{"role": "user", "content": "Ina kwana?"}]


class TestGenerate:
    def test_uses_the_chat_template_and_maps_the_output(self, fake: Harness) -> None:
        gen = LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert fake.tokenizer.chat_calls == [MESSAGES]
        assert gen.text == "decoded 5 tokens"
        assert gen.usage is not None
        assert (gen.usage.prompt_tokens, gen.usage.completion_tokens) == (PROMPT_TOKENS, 5)
        assert gen.finish_reason == "stop"
        assert gen.latency_ms >= 0

    def test_model_card_defaults_reach_generate(self, fake: Harness) -> None:
        LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        kw = fake.llm.generate_kwargs
        assert (kw["temperature"], kw["repetition_penalty"], kw["do_sample"]) == (0.1, 1.12, True)
        assert kw["pad_token_id"] == 2

    def test_max_new_tokens_is_capped_by_the_context_window(self, fake: Harness) -> None:
        LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert fake.llm.generate_kwargs["max_new_tokens"] == LIMIT - PROMPT_TOKENS

    def test_small_request_is_not_inflated(self, fake: Harness) -> None:
        LocalBackend().generate(MESSAGES, GenParams(max_new_tokens=20))  # type: ignore[arg-type]
        assert fake.llm.generate_kwargs["max_new_tokens"] == 20

    def test_zero_temperature_means_greedy(self, fake: Harness) -> None:
        LocalBackend().generate(MESSAGES, GenParams(temperature=0))  # type: ignore[arg-type]
        kw = fake.llm.generate_kwargs
        assert kw["do_sample"] is False
        assert "temperature" not in kw
        assert "top_p" not in kw

    def test_top_p_and_seed(self, fake: Harness) -> None:
        LocalBackend().generate(MESSAGES, GenParams(top_p=0.9, seed=7))  # type: ignore[arg-type]
        assert fake.llm.generate_kwargs["top_p"] == 0.9
        assert fake.seeds == [7]

    def test_hitting_the_token_budget_reports_length(self, fake: Harness) -> None:
        fake.llm = FakeLLM(new_tokens=20)
        gen = LocalBackend().generate(MESSAGES, GenParams(max_new_tokens=20))  # type: ignore[arg-type]
        assert gen.finish_reason == "length"

    def test_prompt_filling_the_context_is_a_resource_error(self, fake: Harness) -> None:
        fake.llm.config.max_position_embeddings = PROMPT_TOKENS
        with pytest.raises(ResourceError, match="context") as info:
            LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.hint is not None

    def test_missing_chat_template_is_reported_not_guessed(self, fake: Harness) -> None:
        fake.tokenizer = FakeTokenizer(chat_template=None)
        with pytest.raises(ConfigError, match="chat template") as info:
            LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert "planning/21" in (info.value.hint or "")

    def test_out_of_memory_during_generation(self, fake: Harness) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        fake.llm = FakeLLM(error=OutOfMemoryError("CUDA out of memory: secret prompt text"))
        with pytest.raises(ResourceError) as info:
            LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert "secret" not in info.value.format()

    def test_other_runtime_failures_never_leak_exception_text(self, fake: Harness) -> None:
        fake.llm = FakeLLM(error=ValueError("private prompt echoed here"))
        with pytest.raises(BackendError) as info:
            LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert "private" not in info.value.format()
        assert "ValueError" in str(info.value)


class TestLoading:
    def test_nothing_is_loaded_until_first_use(self, fake: Harness) -> None:
        LocalBackend()
        assert (fake.tokenizer_loads, fake.model_loads) == (0, 0)

    def test_model_loads_once_across_calls(self, fake: Harness) -> None:
        backend = LocalBackend()
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert (fake.tokenizer_loads, fake.model_loads) == (1, 1)

    def test_default_load_settings(self, fake: Harness) -> None:
        LocalBackend(revision="deadbeef").generate(MESSAGES)  # type: ignore[arg-type]
        kw = fake.model_kwargs
        assert kw["torch_dtype"] == "fp16"
        assert kw["device_map"] == "auto"
        assert kw["revision"] == "deadbeef"
        assert "quantization_config" not in kw

    def test_explicit_device_is_used_instead_of_device_map(self, fake: Harness) -> None:
        LocalBackend(device="mps").generate(MESSAGES)  # type: ignore[arg-type]
        assert "device_map" not in fake.model_kwargs

    def test_missing_torch_gives_the_extras_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ConfigError) as info:
            LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.hint == 'pip install "atlasforge[local]"'

    def test_invalid_quantize(self) -> None:
        with pytest.raises(ConfigError, match="quantize"):
            LocalBackend(quantize="2bit")  # type: ignore[arg-type]

    def test_4bit_without_cuda_points_to_llama_cpp(self, fake: Harness) -> None:
        with pytest.raises(ResourceError, match="NVIDIA") as info:
            LocalBackend(quantize="4bit").generate(MESSAGES)  # type: ignore[arg-type]
        assert "llama.cpp" in (info.value.hint or "")
        assert fake.model_loads == 0

    def test_4bit_with_cuda_builds_the_quantisation_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = Harness(cuda=True)
        harness.install(monkeypatch)
        LocalBackend(quantize="4bit").generate(MESSAGES)  # type: ignore[arg-type]
        config = harness.model_kwargs["quantization_config"]
        assert (config.load_in_4bit, config.load_in_8bit) == (True, False)
        assert "torch_dtype" not in harness.model_kwargs

    @pytest.mark.parametrize(
        ("error_name", "expected"),
        [
            ("GatedRepoError", ModelAccessError),
            ("RepositoryNotFoundError", ModelAccessError),
            ("OutOfMemoryError", ResourceError),
            ("SomethingElse", BackendError),
        ],
    )
    def test_load_errors_are_mapped_and_never_echo_the_original_text(
        self, fake: Harness, error_name: str, expected: type[Exception]
    ) -> None:
        fake.load_error = type(error_name, (Exception,), {})("token hf_SECRET123456 rejected")
        with pytest.raises(expected) as info:
            LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert "hf_SECRET" not in str(info.value)
        assert getattr(info.value, "hint", None)

    def test_access_error_hint_mentions_the_token(self, fake: Harness) -> None:
        fake.load_error = type("GatedRepoError", (Exception,), {})("x")
        with pytest.raises(ModelAccessError) as info:
            LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]
        assert "HF_TOKEN" in (info.value.hint or "")


class TestTranscribe:
    def patch_audio(self, monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
        monkeypatch.setattr(
            "atlasforge.backends.local.decode_audio",
            lambda _a: b"\x00\x00" * int(seconds * SAMPLE_RATE),
        )

    def test_transcribes_with_the_language_specific_official_model(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_audio(monkeypatch, 5)
        result = LocalBackend().transcribe(b"audio", "ha")
        assert result.text == "ina kwana"
        assert result.lang == "ha"
        task, kw = fake.pipeline_calls[0]
        assert task == "automatic-speech-recognition"
        assert kw["model"] == "NCAIR1/Hausa-ASR"
        sent = fake.pipeline_inputs[0]
        assert sent["sampling_rate"] == SAMPLE_RATE
        assert len(sent["raw"]) == 5 * SAMPLE_RATE

    @pytest.mark.parametrize(
        ("lang", "repo"),
        [
            ("ha", "NCAIR1/Hausa-ASR"),
            ("yo", "NCAIR1/Yoruba-ASR"),
            ("ig", "NCAIR1/Igbo-ASR"),
            ("en", "NCAIR1/NigerianAccentedEnglish"),
        ],
    )
    def test_each_language_maps_to_its_official_model(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch, lang: str, repo: str
    ) -> None:
        self.patch_audio(monkeypatch, 1)
        LocalBackend().transcribe(b"a", lang)  # type: ignore[arg-type]
        assert fake.pipeline_calls[0][1]["model"] == repo

    def test_pipeline_is_created_once_per_language(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_audio(monkeypatch, 1)
        backend = LocalBackend()
        backend.transcribe(b"a", "ha")
        backend.transcribe(b"a", "ha")
        backend.transcribe(b"a", "yo")
        assert len(fake.pipeline_calls) == 2

    def test_asr_model_override(self, fake: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch_audio(monkeypatch, 1)
        LocalBackend(asr_models={"ha": "me/my-finetuned-hausa"}).transcribe(b"a", "ha")
        assert fake.pipeline_calls[0][1]["model"] == "me/my-finetuned-hausa"

    def test_audio_over_the_model_limit_points_to_transcribe_long(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_audio(monkeypatch, 45)
        with pytest.raises(AudioError, match="45s") as info:
            LocalBackend().transcribe(b"a", "ha")
        assert "transcribe_long" in (info.value.hint or "")
        assert fake.pipeline_calls == []

    def test_unexpected_pipeline_result(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_audio(monkeypatch, 1)
        fake.pipeline_result = ["not", "a", "dict"]
        with pytest.raises(BackendError, match="unexpected"):
            LocalBackend().transcribe(b"a", "ha")

    def test_asr_load_failure_is_mapped(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_audio(monkeypatch, 1)

        def failing(_task: str, **_kw: Any) -> None:
            raise type("GatedRepoError", (Exception,), {})("x")

        fake.transformers.pipeline = failing  # type: ignore[attr-defined]
        with pytest.raises(ModelAccessError):
            LocalBackend().transcribe(b"a", "ha")

    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"), [(True, False, 0), (False, True, "mps"), (False, False, -1)]
    )
    def test_device_selection(
        self, monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: object
    ) -> None:
        harness = Harness(cuda=cuda, mps=mps)
        harness.install(monkeypatch)
        self.patch_audio(monkeypatch, 1)
        LocalBackend().transcribe(b"a", "ha")
        assert harness.pipeline_calls[0][1]["device"] == expected

    def test_explicit_device_passes_through(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_audio(monkeypatch, 1)
        LocalBackend(device="cpu").transcribe(b"a", "ha")
        assert fake.pipeline_calls[0][1]["device"] == "cpu"


class TestInfoAndLifecycle:
    def test_info_before_loading(self, fake: Harness) -> None:
        info = LocalBackend(revision="r1").info()
        assert (info.backend, info.model, info.revision, info.device) == (
            "local",
            "NCAIR1/N-ATLaS",
            "r1",
            "auto",
        )
        assert info.dtype == "float16"
        assert info.capabilities == {"generate", "transcribe"}

    def test_info_after_loading_reports_the_resolved_commit(self, fake: Harness) -> None:
        backend = LocalBackend()
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        info = backend.info()
        assert info.revision == "abc123def456"
        assert info.device == "cpu"

    def test_info_reports_quantisation_as_the_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        Harness(cuda=True).install(monkeypatch)
        assert LocalBackend(quantize="4bit").info().dtype == "4bit"

    def test_close_releases_models_and_is_idempotent(self, fake: Harness) -> None:
        backend = LocalBackend()
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        backend.close()
        backend.close()
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert fake.model_loads == 2  # reloaded after close

    def test_close_without_torch_installed_is_safe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        LocalBackend().close()

    def test_satisfies_the_backend_protocol(self, fake: Harness) -> None:
        assert isinstance(LocalBackend(), Backend)

    def test_repr_has_no_secrets_and_names_the_model(self) -> None:
        assert "NCAIR1/N-ATLaS" in repr(LocalBackend())


class TestAdapter:
    def install_peft(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, str]]:
        applied: list[tuple[object, str]] = []

        class PeftModel:
            @staticmethod
            def from_pretrained(model: object, adapter: str) -> object:
                applied.append((model, adapter))
                wrapped = FakeLLM()
                wrapped.config.max_position_embeddings = LIMIT
                return wrapped

        peft = ModuleType("peft")
        peft.PeftModel = PeftModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "peft", peft)
        return applied

    def test_adapter_is_applied_on_top_of_the_base_model(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        applied = self.install_peft(monkeypatch)
        LocalBackend(adapter="my/adapter").generate(MESSAGES)  # type: ignore[arg-type]
        assert applied == [(fake.llm, "my/adapter")]

    def test_no_adapter_means_peft_is_never_touched(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "peft", None)  # would raise if imported
        LocalBackend().generate(MESSAGES)  # type: ignore[arg-type]

    def test_the_adapter_is_part_of_the_model_identity(self, fake: Harness) -> None:
        assert LocalBackend().info().model == "NCAIR1/N-ATLaS"
        assert LocalBackend(adapter="runs/hausa").info().model == "NCAIR1/N-ATLaS+runs/hausa"

    def test_a_missing_peft_gives_the_extras_hint(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "peft", None)
        with pytest.raises(ConfigError) as info:
            LocalBackend(adapter="x").generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.hint == 'pip install "atlasforge[local]"'

    def test_a_bad_adapter_is_mapped_without_echoing_the_error(
        self, fake: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class PeftModel:
            @staticmethod
            def from_pretrained(model: object, adapter: str) -> object:
                raise ValueError("secret path details")

        peft = ModuleType("peft")
        peft.PeftModel = PeftModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "peft", peft)
        with pytest.raises(BackendError) as info:
            LocalBackend(adapter="x").generate(MESSAGES)  # type: ignore[arg-type]
        assert "secret" not in info.value.format()
