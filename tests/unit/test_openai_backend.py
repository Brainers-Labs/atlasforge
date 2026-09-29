import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from atlasforge.backends import Backend
from atlasforge.backends.openai import OpenAIBackend
from atlasforge.errors import (
    AudioError,
    BackendConnectionError,
    BackendError,
    BackendHTTPError,
    BackendTimeout,
    ConfigError,
    UnsupportedFeatureError,
)
from atlasforge.types import GenParams

OK_BODY = {
    "id": "chatcmpl-1",
    "choices": [{"message": {"role": "assistant", "content": "Sannu!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}
MESSAGES = [{"role": "user", "content": "Ina kwana?"}]
Handler = Callable[[httpx.Request], httpx.Response]


def make(
    handler: Handler, **kwargs: object
) -> tuple[OpenAIBackend, list[httpx.Request], list[float]]:
    """Backend wired to a mock transport; returns it plus the seen requests and sleeps."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    options: dict[str, object] = {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "NCAIR1/N-ATLaS",
        "api_key": None,
        "api_key_env": None,
        "transport": httpx.MockTransport(recording),
        "sleep": slept.append,
    }
    options.update(kwargs)
    return OpenAIBackend(**options), seen, slept  # type: ignore[arg-type]


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=OK_BODY, headers={"x-request-id": "req-9"})


class TestGenerate:
    def test_maps_response(self) -> None:
        backend, _, _ = make(ok)
        gen = backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert gen.text == "Sannu!"
        assert gen.finish_reason == "stop"
        assert gen.usage is not None
        assert (gen.usage.prompt_tokens, gen.usage.completion_tokens) == (7, 3)
        assert gen.request_id == "req-9"
        assert gen.latency_ms >= 0

    def test_request_id_falls_back_to_body_id(self) -> None:
        backend, _, _ = make(lambda _r: httpx.Response(200, json=OK_BODY))
        assert backend.generate(MESSAGES).request_id == "chatcmpl-1"  # type: ignore[arg-type]

    def test_sends_model_card_defaults(self) -> None:
        backend, seen, _ = make(ok)
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        body = json.loads(seen[0].content)
        assert seen[0].url.path == "/v1/chat/completions"
        assert body["model"] == "NCAIR1/N-ATLaS"
        assert body["messages"] == MESSAGES
        assert (body["temperature"], body["repetition_penalty"], body["max_tokens"]) == (
            0.1,
            1.12,
            1000,
        )
        assert "top_p" not in body
        assert "seed" not in body

    def test_optional_params_sent_when_set(self) -> None:
        backend, seen, _ = make(ok)
        backend.generate(MESSAGES, GenParams(temperature=0.7, top_p=0.9, seed=5, max_new_tokens=50))  # type: ignore[arg-type]
        body = json.loads(seen[0].content)
        assert (body["temperature"], body["top_p"], body["seed"], body["max_tokens"]) == (
            0.7,
            0.9,
            5,
            50,
        )

    def test_repetition_penalty_can_be_disabled(self) -> None:
        backend, seen, _ = make(ok, send_repetition_penalty=False)
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "repetition_penalty" not in json.loads(seen[0].content)

    def test_null_content_becomes_empty_string(self) -> None:
        body = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        backend, _, _ = make(lambda _r: httpx.Response(200, json=body))
        gen = backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert gen.text == ""
        assert gen.usage is None

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, json={"nope": 1}),
            httpx.Response(200, json={"choices": []}),
            httpx.Response(200, json=[1, 2]),
            httpx.Response(200, text="<html>not json</html>"),
        ],
    )
    def test_malformed_success_bodies_raise_backend_error(self, response: httpx.Response) -> None:
        backend, _, _ = make(lambda _r: response)
        with pytest.raises(BackendError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.hint is not None


class TestAuth:
    def test_key_sent_as_bearer(self) -> None:
        backend, seen, _ = make(ok, api_key="sk-test-123456")
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert seen[0].headers["authorization"] == "Bearer sk-test-123456"

    def test_no_key_no_header(self) -> None:
        backend, seen, _ = make(ok)
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "authorization" not in seen[0].headers

    def test_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "env-key-abcdef")
        backend, seen, _ = make(ok, api_key_env="MY_KEY")
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert seen[0].headers["authorization"] == "Bearer env-key-abcdef"

    def test_explicit_key_beats_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "from-env")
        backend, seen, _ = make(ok, api_key="explicit", api_key_env="MY_KEY")
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert seen[0].headers["authorization"] == "Bearer explicit"

    def test_key_never_appears_in_repr_or_errors(self) -> None:
        backend, _, _ = make(
            lambda _r: httpx.Response(401, text="bad key sk-test-123456"), api_key="sk-test-123456"
        )
        assert "sk-test-123456" not in repr(backend)
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "sk-test-123456" not in str(info.value)
        assert "sk-test-123456" not in (info.value.hint or "")


class TestHttpErrors:
    def test_401_has_auth_hint_and_status(self) -> None:
        backend, _, _ = make(lambda _r: httpx.Response(401, text="nope"))
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.status_code == 401
        assert "API key" in (info.value.hint or "")

    def test_404_hint_mentions_base_url_and_model(self) -> None:
        backend, _, _ = make(lambda _r: httpx.Response(404))
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "/v1" in (info.value.hint or "")

    def test_400_hint_mentions_repetition_penalty(self) -> None:
        backend, _, _ = make(lambda _r: httpx.Response(400, text="unknown field"))
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "send_repetition_penalty=False" in (info.value.hint or "")

    def test_response_body_never_leaks_into_message_but_is_kept_for_debugging(self) -> None:
        echoed = "the user asked: SECRET PROMPT TEXT"
        backend, _, _ = make(lambda _r: httpx.Response(400, text=echoed))
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "SECRET" not in str(info.value)
        assert "SECRET" not in info.value.format()
        assert info.value.body_excerpt == echoed

    def test_body_excerpt_is_truncated(self) -> None:
        backend, _, _ = make(lambda _r: httpx.Response(400, text="x" * 5000))
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.body_excerpt is not None
        assert len(info.value.body_excerpt) == 300

    def test_request_id_captured_from_error_response(self) -> None:
        backend, _, _ = make(lambda _r: httpx.Response(400, headers={"x-request-id": "rid-5"}))
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.request_id == "rid-5"


class TestRetries:
    def test_429_then_success_respects_retry_after(self) -> None:
        responses = iter([httpx.Response(429, headers={"retry-after": "2"}), ok(None)])  # type: ignore[arg-type]
        backend, seen, slept = make(lambda _r: next(responses))
        assert backend.generate(MESSAGES).text == "Sannu!"  # type: ignore[arg-type]
        assert len(seen) == 2
        assert slept == [2.0]

    def test_5xx_exhausts_retries_then_raises(self) -> None:
        backend, seen, slept = make(lambda _r: httpx.Response(503), max_retries=2)
        with pytest.raises(BackendHTTPError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert info.value.status_code == 503
        assert len(seen) == 3
        assert slept == [0.5, 1.0]  # exponential backoff

    def test_max_retries_zero_means_one_attempt(self) -> None:
        backend, seen, slept = make(lambda _r: httpx.Response(500), max_retries=0)
        with pytest.raises(BackendHTTPError):
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert (len(seen), slept) == (1, [])

    def test_client_errors_are_not_retried(self) -> None:
        backend, seen, _ = make(lambda _r: httpx.Response(400))
        with pytest.raises(BackendHTTPError):
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert len(seen) == 1

    def test_retry_after_is_capped_and_garbage_ignored(self) -> None:
        responses = iter(
            [
                httpx.Response(429, headers={"retry-after": "9999"}),
                httpx.Response(429, headers={"retry-after": "soon"}),
                ok(None),  # type: ignore[arg-type]
            ]
        )
        backend, _, slept = make(lambda _r: next(responses), max_retries=2)
        backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert slept == [30.0, 1.0]

    def test_backoff_is_capped(self) -> None:
        backend, _, slept = make(lambda _r: httpx.Response(503), max_retries=8)
        with pytest.raises(BackendHTTPError):
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert max(slept) == 8.0

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ConfigError):
            OpenAIBackend(base_url="http://127.0.0.1/v1", model="m", max_retries=-1)


class TestTransportFailures:
    def test_timeout_is_not_retried_and_maps_to_backend_timeout(self) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        backend, seen, slept = make(slow, timeout=5)
        with pytest.raises(BackendTimeout) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "5s" in str(info.value)
        assert info.value.hint is not None
        assert (len(seen), slept) == (1, [])

    def test_connection_error_retries_then_raises(self) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        backend, seen, slept = make(down, max_retries=2)
        with pytest.raises(BackendConnectionError) as info:
            backend.generate(MESSAGES)  # type: ignore[arg-type]
        assert "127.0.0.1" in str(info.value)
        assert len(seen) == 3
        assert slept == [0.5, 1.0]

    def test_recovers_after_transient_connection_error(self) -> None:
        state = {"calls": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if state["calls"] == 1:
                raise httpx.ConnectError("blip", request=request)
            return ok(request)

        backend, _, _ = make(flaky)
        assert backend.generate(MESSAGES).text == "Sannu!"  # type: ignore[arg-type]


class TestBaseUrl:
    def test_loopback_http_allowed(self) -> None:
        for host in (
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
        ):
            OpenAIBackend(base_url=host, model="m").close()

    def test_https_anywhere_allowed(self) -> None:
        OpenAIBackend(base_url="https://gpu.example.com/v1", model="m").close()

    def test_plain_http_to_remote_host_refused(self) -> None:
        with pytest.raises(ConfigError, match="plain http") as info:
            OpenAIBackend(base_url="http://gpu.example.com/v1", model="m")
        assert "allow_insecure_http" in (info.value.hint or "")

    def test_insecure_http_can_be_opted_into(self) -> None:
        OpenAIBackend(
            base_url="http://192.168.1.20:8000/v1", model="m", allow_insecure_http=True
        ).close()

    @pytest.mark.parametrize("bad", ["ftp://x/v1", "gpu.example.com/v1", "", "http://"])
    def test_bad_urls_rejected(self, bad: str) -> None:
        with pytest.raises(ConfigError, match="http"):
            OpenAIBackend(base_url=bad, model="m")

    def test_trailing_slash_does_not_change_the_request_path(self) -> None:
        for url in ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1/"):
            backend, seen, _ = make(ok, base_url=url)
            backend.generate(MESSAGES)  # type: ignore[arg-type]
            assert seen[0].url.path == "/v1/chat/completions"


class TestTranscribe:
    def transcribing(self, **kwargs: object) -> tuple[OpenAIBackend, list[httpx.Request]]:
        backend, seen, _ = make(
            lambda _r: httpx.Response(200, json={"text": "ina kwana"}), **kwargs
        )
        return backend, seen

    def test_uploads_multipart_with_language_and_model(self, tmp_path: Path) -> None:
        audio = tmp_path / "note.ogg"
        audio.write_bytes(b"OggS-fake-audio")
        backend, seen = self.transcribing(asr_model="NCAIR1/Hausa-ASR")
        result = backend.transcribe(audio, "ha")
        assert result.text == "ina kwana"
        assert result.lang == "ha"
        request = seen[0]
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["content-type"].startswith("multipart/form-data")
        body = request.content
        assert b'name="language"' in body
        assert b"ha" in body
        assert b"NCAIR1/Hausa-ASR" in body
        assert b'filename="note.ogg"' in body
        assert b"OggS-fake-audio" in body

    def test_asr_model_defaults_to_llm_model_name(self, tmp_path: Path) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF")
        backend, seen = self.transcribing()
        backend.transcribe(audio, "yo")
        assert b"NCAIR1/N-ATLaS" in seen[0].content

    def test_accepts_raw_bytes(self) -> None:
        backend, seen = self.transcribing()
        assert backend.transcribe(b"RIFFdata", "ig").text == "ina kwana"
        assert b"RIFFdata" in seen[0].content

    def test_missing_file_is_an_audio_error_with_hint(self, tmp_path: Path) -> None:
        backend, seen = self.transcribing()
        with pytest.raises(AudioError) as info:
            backend.transcribe(tmp_path / "missing.wav", "ha")
        assert info.value.hint is not None
        assert seen == []

    def test_asr_disabled(self) -> None:
        backend, seen = self.transcribing(asr=False)
        with pytest.raises(UnsupportedFeatureError):
            backend.transcribe(b"x", "ha")
        assert seen == []

    def test_response_without_text_field(self) -> None:
        backend, _, _ = make(lambda _r: httpx.Response(200, json={"words": []}))
        with pytest.raises(BackendError, match="audio/transcriptions"):
            backend.transcribe(b"x", "ha")


class TestInfoAndLifecycle:
    def test_info_names_the_model(self) -> None:
        info = make(ok)[0].info()
        assert (info.backend, info.model) == ("openai", "NCAIR1/N-ATLaS")
        assert info.capabilities == {"generate", "transcribe"}

    def test_info_reflects_asr_flag(self) -> None:
        assert make(ok, asr=False)[0].info().capabilities == {"generate"}

    def test_satisfies_the_backend_protocol(self) -> None:
        assert isinstance(make(ok)[0], Backend)

    def test_close_is_idempotent_and_context_manager_closes(self) -> None:
        backend, _, _ = make(ok)
        backend.close()
        backend.close()
        with make(ok)[0] as inner:
            assert inner.info().backend == "openai"
