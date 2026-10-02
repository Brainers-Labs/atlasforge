"""Run the official N-ATLaS models locally with Hugging Face ``transformers``.

**Status:** the ASR path has been run against the real official speech models (Mac M1, 30 Sep 2026).
The LLM path is written against the model card and tested only with stand-in ``torch``/``transformers``
modules: fp16 needs a GPU, so its real behaviour (memory, speed) is still unverified. Findings are
recorded in ``planning/21_NATLAS_DISCOVERY.md``.

Needs ``pip install "atlasforge[local]"``. Model files are gated on Hugging Face: accept
the licence on each model page and provide ``HF_TOKEN``. Weights are never bundled.

Heavy imports happen lazily, on first use, so importing this module is cheap.
"""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any, Final, Literal

from atlasforge.asr.audio import SAMPLE_RATE, decode_audio, duration_s, pcm_to_float32
from atlasforge.backends.factory import DEFAULT_MODEL
from atlasforge.errors import (
    AudioError,
    BackendError,
    ConfigError,
    ModelAccessError,
    ResourceError,
)
from atlasforge.types import BackendInfo, Generation, GenParams, Transcript, Usage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from atlasforge.types import AudioInput, Lang, Message

ASR_MODELS: Final[dict[str, str]] = {
    "ha": "NCAIR1/Hausa-ASR",
    "yo": "NCAIR1/Yoruba-ASR",
    "ig": "NCAIR1/Igbo-ASR",
    "en": "NCAIR1/NigerianAccentedEnglish",
}
ASR_LIMIT_S: Final = 30.0

Quantize = Literal["none", "4bit", "8bit"]
_EXTRAS_HINT: Final = 'pip install "atlasforge[local]"'
_ACCESS_HINT: Final = (
    "Accept the licence on the model's Hugging Face page, then set HF_TOKEN "
    "(or run `huggingface-cli login`). `atlasforge doctor` checks the token."
)


class LocalBackend:
    """N-ATLaS LLM and ASR models running in-process."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        revision: str | None = None,
        quantize: Quantize = "none",
        device: str = "auto",
        dtype: str = "float16",
        asr_models: Mapping[str, str] | None = None,
    ) -> None:
        if quantize not in ("none", "4bit", "8bit"):
            raise ConfigError(f"quantize must be 'none', '4bit' or '8bit', got {quantize!r}.")
        self._model_id = model
        self._revision = revision
        self._quantize = quantize
        self._device = device
        self._dtype = dtype
        self._asr_repos = {**ASR_MODELS, **(asr_models or {})}
        self._tokenizer: Any = None
        self._llm: Any = None
        self._pipelines: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"LocalBackend(model={self._model_id!r}, quantize={self._quantize!r})"

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        """Chat with the LLM using the tokenizer's own chat template."""
        p = params or GenParams()
        torch = _import("torch")
        tokenizer, llm = self._load_llm()
        if not getattr(tokenizer, "chat_template", None):
            raise ConfigError(
                f"The tokenizer of {self._model_id} has no chat template.",
                hint="Record how N-ATLaS expects prompts in planning/21, then add support.",
            )
        encoded = tokenizer.apply_chat_template(
            [dict(m) for m in messages],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(llm.device)
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        limit = getattr(llm.config, "max_position_embeddings", None)
        max_new = p.max_new_tokens
        if isinstance(limit, int):
            max_new = min(max_new, limit - prompt_tokens)
            if max_new < 1:
                raise ResourceError(
                    f"The prompt ({prompt_tokens} tokens) fills the model's {limit}-token context.",
                    hint="Shorten the prompt.",
                )

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "repetition_penalty": p.repetition_penalty,
            "do_sample": p.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if p.temperature > 0:
            kwargs["temperature"] = p.temperature
            if p.top_p is not None:
                kwargs["top_p"] = p.top_p
        if p.seed is not None:
            torch.manual_seed(p.seed)

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output = llm.generate(**encoded, **kwargs)
        except Exception as exc:
            raise _map_runtime_error(exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        new_tokens = output[0][prompt_tokens:]
        completion_tokens = int(new_tokens.shape[-1])
        return Generation(
            text=tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            latency_ms=latency_ms,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            finish_reason="length" if completion_tokens >= max_new else "stop",
        )

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        """Transcribe up to 30 s with the official ASR model for ``lang``.

        Longer audio: use :func:`atlasforge.asr.transcribe_long`, which chunks it.
        """
        pcm = decode_audio(audio)
        if duration_s(pcm) > ASR_LIMIT_S + 0.5:
            raise AudioError(
                f"Audio is {duration_s(pcm):.0f}s; the ASR models take at most {ASR_LIMIT_S:.0f}s.",
                hint="Use atlasforge.asr.transcribe_long(), which splits long audio.",
            )
        pipeline = self._load_asr(lang)
        start = time.perf_counter()
        try:
            result = pipeline({"raw": pcm_to_float32(pcm), "sampling_rate": SAMPLE_RATE})
        except Exception as exc:
            raise _map_runtime_error(exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str):
            raise BackendError(
                "The ASR pipeline returned an unexpected result.",
                hint="Check the transformers version against the model card.",
            )
        return Transcript(text=text.strip(), lang=lang, latency_ms=latency_ms)

    def info(self) -> BackendInfo:
        revision = self._revision
        device = self._device
        if self._llm is not None:
            revision = revision or getattr(self._llm.config, "_commit_hash", None)
            device = str(self._llm.device)
        return BackendInfo(
            backend="local",
            model=self._model_id,
            revision=revision,
            device=device,
            dtype=self._dtype if self._quantize == "none" else self._quantize,
            capabilities=frozenset({"generate", "transcribe"}),
        )

    def close(self) -> None:
        """Drop loaded models and free accelerator memory. Safe to call twice."""
        self._tokenizer = None
        self._llm = None
        self._pipelines.clear()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return
        if getattr(torch.cuda, "is_available", lambda: False)():
            torch.cuda.empty_cache()

    def _load_llm(self) -> tuple[Any, Any]:
        if self._llm is not None:
            return self._tokenizer, self._llm
        torch = _import("torch")
        transformers = _import("transformers")
        kwargs: dict[str, Any] = {"revision": self._revision}
        if self._quantize != "none":
            if not torch.cuda.is_available():
                raise ResourceError(
                    f"{self._quantize} quantisation through bitsandbytes needs an NVIDIA GPU.",
                    hint="On a Mac or CPU-only machine, serve a quantised model with llama.cpp or "
                    "Ollama and use the openai backend.",
                )
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=self._quantize == "4bit",
                load_in_8bit=self._quantize == "8bit",
                bnb_4bit_compute_dtype=getattr(torch, self._dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, self._dtype)
        if self._device == "auto":
            kwargs["device_map"] = "auto"
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_id, revision=self._revision
            )
            llm = transformers.AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)
            if self._device != "auto" and self._quantize == "none":
                llm = llm.to(self._device)
        except Exception as exc:
            raise _map_load_error(exc, self._model_id) from exc
        self._tokenizer, self._llm = tokenizer, llm
        return tokenizer, llm

    def _load_asr(self, lang: Lang) -> Any:
        if lang in self._pipelines:
            return self._pipelines[lang]
        transformers = _import("transformers")
        repo = self._asr_repos[lang]
        try:
            pipeline = transformers.pipeline(
                "automatic-speech-recognition", model=repo, device=_pipeline_device(self._device)
            )
        except Exception as exc:
            raise _map_load_error(exc, repo) from exc
        self._pipelines[lang] = pipeline
        return pipeline


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfigError(
            f"The local backend needs '{name}', which is not installed.", hint=_EXTRAS_HINT
        ) from exc


def _pipeline_device(device: str) -> Any:
    """Translate our device setting to the transformers pipeline argument."""
    if device != "auto":
        return device
    torch = _import("torch")
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else -1


def _map_load_error(exc: Exception, repo: str) -> Exception:
    name = type(exc).__name__
    text = str(exc)
    if "OutOfMemory" in name or isinstance(exc, MemoryError):
        return ResourceError(
            f"Not enough memory to load {repo}.",
            hint="Use quantize='4bit' on an NVIDIA GPU, or serve the model elsewhere "
            "(llama.cpp / vLLM) and use the openai backend.",
        )
    if "Gated" in name or "RepositoryNotFound" in name or "401" in text or "403" in text:
        return ModelAccessError(f"Cannot access {repo}.", hint=_ACCESS_HINT)
    if "ConnectionError" in name or "OfflineMode" in name:
        return BackendError(
            f"Could not download {repo}.", hint="Check the network, or use a local cache."
        )
    return BackendError(f"Could not load {repo} ({name}).", hint="Run `atlasforge doctor`.")


def _map_runtime_error(exc: Exception) -> Exception:
    if "OutOfMemory" in type(exc).__name__ or isinstance(exc, MemoryError):
        return ResourceError(
            "Ran out of memory during generation.",
            hint="Lower max_new_tokens, shorten the prompt, or use quantize='4bit'.",
        )
    return BackendError(
        f"Generation failed ({type(exc).__name__}).", hint="Run `atlasforge doctor`."
    )
