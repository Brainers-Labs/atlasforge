"""Exception hierarchy.

Every error carries an optional ``hint``: the concrete next step for the user.
The CLI prints ``message`` and ``hint``; library callers can use either.
"""

from __future__ import annotations


class AtlasForgeError(Exception):
    """Base class for all errors raised deliberately by AtlasForge."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message

    def format(self) -> str:
        """Message plus hint, as shown to CLI users."""
        if self.hint:
            return f"{self.message}\n  -> {self.hint}"
        return self.message


class ConfigError(AtlasForgeError):
    """Bad or missing configuration, or a missing optional extra."""


class ModelAccessError(AtlasForgeError):
    """A gated model repository is not accessible (licence not accepted, bad token)."""


class ResourceError(AtlasForgeError):
    """Not enough VRAM, RAM or disk, or a required system tool is missing."""


class UnsupportedFeatureError(AtlasForgeError):
    """The chosen backend or model does not provide the requested capability."""


class AudioError(AtlasForgeError):
    """Audio input is unreadable or in an unsupported format."""


class DatasetError(AtlasForgeError):
    """A dataset file is malformed. ``line`` is 1-based when known."""

    def __init__(self, message: str, *, line: int | None = None, hint: str | None = None) -> None:
        super().__init__(f"line {line}: {message}" if line is not None else message, hint=hint)
        self.line = line


class RunAborted(AtlasForgeError):  # noqa: N818 - reads as a state, like KeyboardInterrupt
    """An evaluation run stopped early because too many examples failed in a row."""


class BackendError(AtlasForgeError):
    """A backend failed to produce a result."""


class BackendTimeout(BackendError):  # noqa: N818 - named to avoid shadowing builtin TimeoutError
    """A backend call exceeded its timeout."""


class BackendConnectionError(BackendError):
    """The backend could not be reached."""


class BackendHTTPError(BackendError):
    """An HTTP backend answered with an error status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_id: str | None = None,
        body_excerpt: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.status_code = status_code
        self.request_id = request_id
        self.body_excerpt = body_excerpt
