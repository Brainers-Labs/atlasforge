"""The backend contract.

Local ``transformers`` weights, an OpenAI-compatible server (vLLM, HF Endpoints,
community gateways) and, later, the official NAIC API all implement this.
A backend that lacks a capability raises ``UnsupportedFeatureError`` and does
not advertise it in ``info().capabilities``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atlasforge.types import (
        AudioInput,
        BackendInfo,
        Generation,
        GenParams,
        Lang,
        Message,
        Transcript,
    )


@runtime_checkable
class Backend(Protocol):
    """Anything that can run N-ATLaS models."""

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        """Run the LLM on a chat. ``params=None`` means the model-card defaults."""
        ...

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        """Transcribe audio with the official ASR model for ``lang``."""
        ...

    def info(self) -> BackendInfo:
        """Identity of what is running (model, revision, device, dtype)."""
        ...

    def close(self) -> None:
        """Release models, sockets and GPU memory. Safe to call twice."""
        ...
