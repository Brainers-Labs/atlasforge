"""Backend for any OpenAI-compatible server: vLLM, llama.cpp, Ollama, HF Endpoints, gateways.

Speaking this protocol is a transport choice, not a claim that N-ATLaS is an OpenAI
model. The model behind the endpoint must be an official NCAIR1 model; ``info()``
records which one was requested so every report names it.

Privacy: error messages never contain response bodies (servers sometimes echo the
prompt). A short body excerpt is kept on ``BackendHTTPError.body_excerpt`` for debugging.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

import httpx

from atlasforge.errors import (
    AudioError,
    BackendConnectionError,
    BackendError,
    BackendHTTPError,
    BackendTimeout,
    ConfigError,
    UnsupportedFeatureError,
)
from atlasforge.types import BackendInfo, Generation, GenParams, Transcript, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from atlasforge.types import AudioInput, Lang, Message

_RETRYABLE: Final = frozenset({429, 500, 502, 503, 504})
_LOOPBACK: Final = frozenset({"localhost", "127.0.0.1", "::1"})
_EXCERPT_CHARS: Final = 300
_MAX_BACKOFF_S: Final = 8.0
_MAX_RETRY_AFTER_S: Final = 30.0

_HTTP_HINTS: Final = {
    400: (
        "The server rejected the request. If it mentions repetition_penalty, "
        "pass send_repetition_penalty=False."
    ),
    401: "Check the API key (api_key=..., or the ATLASFORGE_API_KEY environment variable).",
    403: "The key was accepted but is not allowed to do this. Check its permissions.",
    404: "Check that base_url ends with /v1 and that the model name matches what the server serves.",
    429: "Rate limited. Lower the concurrency or retry later.",
}


class OpenAIBackend:
    """Run the N-ATLaS LLM (and optionally ASR) through an OpenAI-compatible HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = "ATLASFORGE_API_KEY",
        asr_model: str | None = None,
        asr: bool = True,
        timeout: float = 120.0,
        max_retries: int = 2,
        send_repetition_penalty: bool = True,
        allow_insecure_http: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ConfigError("max_retries must be >= 0.")
        self._base_url = _validate_base_url(base_url, allow_insecure_http=allow_insecure_http)
        self.model = model
        self._asr_model = asr_model or model
        self._asr = asr
        self._timeout = timeout
        self._max_retries = max_retries
        self._send_repetition_penalty = send_repetition_penalty
        self._sleep = sleep

        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"OpenAIBackend(base_url={self._base_url!r}, model={self.model!r})"

    def __enter__(self) -> OpenAIBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        """Chat completion. ``params=None`` uses the model-card defaults."""
        p = params or GenParams()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": p.temperature,
            "max_tokens": p.max_new_tokens,
        }
        if p.top_p is not None:
            payload["top_p"] = p.top_p
        if p.seed is not None:
            payload["seed"] = p.seed
        if self._send_repetition_penalty:
            payload["repetition_penalty"] = p.repetition_penalty

        response, latency_ms = self._request("chat/completions", json=payload)
        data = _json_object(response)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                "Unexpected response shape from /chat/completions.",
                hint="Is base_url really an OpenAI-compatible server?",
            ) from exc
        usage = data.get("usage")
        return Generation(
            text=text,
            latency_ms=latency_ms,
            usage=(
                Usage(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
                if isinstance(usage, dict)
                else None
            ),
            finish_reason=finish_reason,
            request_id=_request_id(response, data),
        )

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        """POST the audio to ``/audio/transcriptions`` with the language hint."""
        if not self._asr:
            raise UnsupportedFeatureError(
                "This backend was created with asr=False.",
                hint="Create it with asr=True and an endpoint that serves /audio/transcriptions.",
            )
        if isinstance(audio, bytes):
            content, filename = audio, "audio"
        else:
            path = Path(audio)
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise AudioError(
                    f"Cannot read audio file {path}: {exc.strerror or exc}",
                    hint="Check the path.",
                ) from exc
            filename = path.name

        response, latency_ms = self._request(
            "audio/transcriptions",
            data={"model": self._asr_model, "language": lang, "response_format": "json"},
            files={"file": (filename, content)},
        )
        text = _json_object(response).get("text")
        if not isinstance(text, str):
            raise BackendError(
                "Unexpected response shape from /audio/transcriptions.",
                hint="The endpoint should return JSON with a 'text' field.",
            )
        return Transcript(text=text, lang=lang, latency_ms=latency_ms)

    def info(self) -> BackendInfo:
        return BackendInfo(
            backend="openai",
            model=self.model,
            capabilities=frozenset({"generate", "transcribe"} if self._asr else {"generate"}),
        )

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> tuple[httpx.Response, float]:
        """POST with retries on 429/5xx and connection errors. Returns (response, latency_ms)."""
        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                response = self._client.post(path, json=json, data=data, files=files)
            except httpx.TimeoutException as exc:
                raise BackendTimeout(
                    f"No response within {self._timeout:g}s.",
                    hint="Raise timeout=..., or check that the server is not overloaded.",
                ) from exc
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    self._backoff(attempt, None)
                    attempt += 1
                    continue
                raise BackendConnectionError(
                    f"Could not reach {self._base_url} ({type(exc).__name__}).",
                    hint="Is the server running, and is base_url correct?",
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code in _RETRYABLE and attempt < self._max_retries:
                self._backoff(attempt, response.headers.get("retry-after"))
                attempt += 1
                continue
            if response.is_error:
                raise _http_error(response)
            return response, latency_ms

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(0.5 * 2**attempt, _MAX_BACKOFF_S)
        if retry_after is not None:
            with contextlib.suppress(ValueError):  # HTTP-date form: keep the exponential delay
                delay = min(float(retry_after), _MAX_RETRY_AFTER_S)
        self._sleep(delay)


def _validate_base_url(base_url: str, *, allow_insecure_http: bool) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(
            f"base_url must be an http(s) URL, got {base_url!r}.",
            hint="Example: http://127.0.0.1:8000/v1",
        )
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK and not allow_insecure_http:
        raise ConfigError(
            f"Refusing plain http to a non-local host ({parts.hostname}).",
            hint="Use https, or pass allow_insecure_http=True if you trust the network.",
        )
    return base_url.rstrip("/") + "/"


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise BackendError(
            "The server returned a non-JSON body.",
            hint="Is base_url really an OpenAI-compatible server?",
        ) from exc
    if not isinstance(data, dict):
        raise BackendError(
            "The server returned an unexpected JSON value.",
            hint="Is base_url really an OpenAI-compatible server?",
        )
    return data


def _request_id(response: httpx.Response, data: dict[str, Any]) -> str | None:
    header: str | None = response.headers.get("x-request-id") or response.headers.get("request-id")
    if header:
        return header
    body_id: object = data.get("id")
    return body_id if isinstance(body_id, str) else None


def _http_error(response: httpx.Response) -> BackendHTTPError:
    status = response.status_code
    hint = _HTTP_HINTS.get(status) or (
        "Server-side error. Check the server logs." if status >= 500 else None
    )
    return BackendHTTPError(
        f"Endpoint returned HTTP {status}.",
        status_code=status,
        request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
        body_excerpt=response.text[:_EXCERPT_CHARS],
        hint=hint,
    )
