"""Build a backend from CLI-style options."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from atlasforge.errors import ConfigError

if TYPE_CHECKING:
    from atlasforge.backends.base import Backend

DEFAULT_MODEL: Final = "NCAIR1/N-ATLaS"
BACKEND_NAMES: Final = ("openai", "local")


def build_backend(
    name: str,
    *,
    base_url: str | None = None,
    model: str = DEFAULT_MODEL,
    api_key_env: str | None = "ATLASFORGE_API_KEY",
    asr_model: str | None = None,
    timeout: float = 120.0,
    retries: int = 2,
    quantize: str = "none",
    device: str = "auto",
    adapter: str | None = None,
    send_repetition_penalty: bool = True,
    allow_insecure_http: bool = False,
) -> Backend:
    """Create the named backend. Heavy dependencies are imported only for ``local``."""
    if adapter and name != "local":  # checked first so it can never be silently ignored
        raise ConfigError(
            "--adapter only works with the local backend.",
            hint="With a server, load the adapter there and pass its served name as --model.",
        )
    if name == "openai":
        if not base_url:
            raise ConfigError(
                "The openai backend needs a server URL.",
                hint="Pass --base-url (or set ATLASFORGE_BASE_URL), e.g. http://127.0.0.1:8000/v1",
            )
        from atlasforge.backends.openai import OpenAIBackend  # noqa: PLC0415 - keep import light

        return OpenAIBackend(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            asr_model=asr_model,
            timeout=timeout,
            max_retries=retries,
            send_repetition_penalty=send_repetition_penalty,
            allow_insecure_http=allow_insecure_http,
        )
    if name == "local":
        from atlasforge.backends.local import LocalBackend  # noqa: PLC0415 - keep import light

        return LocalBackend(
            adapter=adapter,
            model=model,
            quantize=quantize,  # type: ignore[arg-type]  # validated in LocalBackend
            device=device,
        )
    raise ConfigError(f"Unknown backend {name!r}.", hint=f"Use one of: {', '.join(BACKEND_NAMES)}.")
