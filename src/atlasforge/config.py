"""The optional ``atlasforge.toml`` project file.

Some settings are the same for every run in a project: which backend, which endpoint, which
model. Writing them on every command line is noise, so a project may keep them in one file:

.. code-block:: toml

    [atlasforge]
    backend = "openai"
    base_url = "http://127.0.0.1:8000/v1"
    model = "NCAIR1/N-ATLaS"

The file is read from the nearest directory at or above the one you run in, so a project root
holds it once and subdirectories inherit it.

**Precedence, highest first:** the flag you type → the environment variable → this file → the
built-in default. A file never overrides something you asked for explicitly.

Two deliberate limits. Only the keys in :data:`KEYS` are accepted, and an unknown key is an
error rather than a silent no-op: a misspelled setting that quietly does nothing is worse than
one that stops the run. And nothing is read from the user's home directory or from anywhere
system-wide, so a run is reproducible from the repository it was made in.

``allow_insecure_http`` is deliberately *not* a key here: plain http to a non-local server
should be a decision typed on the command line and visible in the shell history, not a property
of a checked-in file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from atlasforge.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

#: The file a project's settings live in.
FILE_NAME: Final = "atlasforge.toml"

#: The only table read. Other tables in the same file belong to other tools.
TABLE: Final = "atlasforge"

#: Every accepted key, and the type its value must have.
KEYS: Final[dict[str, type]] = {
    "backend": str,
    "base_url": str,
    "model": str,
    "quantize": str,
    "device": str,
    "timeout": float,
    "retries": int,
}

_EXAMPLES: Final[dict[str, str]] = {"str": '"text"', "float": "120.0", "int": "2"}
_NAMES: Final[dict[str, str]] = {"str": "string", "float": "number", "int": "whole number"}


@dataclass(frozen=True, slots=True)
class Config:
    """What the project file said: the file it came from, and the values it set."""

    path: Path | None = None
    values: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.values)


def type_name(key: str) -> str:
    """The type a key must have, in words — the same words its error message uses."""
    return _NAMES[KEYS[key].__name__]


def find_file(start: str | Path | None = None) -> Path | None:
    """The nearest ``atlasforge.toml`` at or above ``start`` (default: the current directory)."""
    directory = Path.cwd() if start is None else Path(start)
    if directory.is_file():  # tolerate being handed a file, e.g. from a test
        directory = directory.parent
    for candidate in (directory.resolve(), *directory.resolve().parents):
        path = candidate / FILE_NAME
        if path.is_file():
            return path
    return None


def load_config(start: str | Path | None = None) -> Config:
    """The project file found from ``start``, or an empty :class:`Config` when there is none."""
    path = find_file(start)
    return load_file(path) if path is not None else Config()


def load_file(path: str | Path) -> Config:
    """Read exactly this file. Missing or malformed files are errors, not empty configs."""
    target = Path(path)
    try:
        with target.open("rb") as handle:
            parsed = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"No configuration file at {target}.",
            hint=f"Create it with a [{TABLE}] table, or drop the path to look for the nearest "
            f"{FILE_NAME}.",
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"{target} is not valid TOML: {exc}",
            hint="Check the quoting and the structure; see the configuration reference.",
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Could not read {target}: {exc}",
            hint="Check the file permissions, or remove it to fall back to the defaults.",
        ) from exc

    table = parsed.get(TABLE, {})
    if not isinstance(table, dict):
        raise ConfigError(
            f"[{TABLE}] in {target} is not a table of settings.",
            hint=f"Write it as a table: [{TABLE}] followed by key = value lines.",
        )
    return Config(path=target, values=_checked(table, target))


def _checked(table: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Validate the keys and their types, so a typo stops the run instead of doing nothing."""
    unknown = sorted(set(table) - set(KEYS))
    if unknown:
        raise ConfigError(
            f"{path} has unknown setting(s): {', '.join(unknown)}.",
            hint=f"Valid keys under [{TABLE}]: {', '.join(sorted(KEYS))}.",
        )
    checked: dict[str, Any] = {}
    for key, value in table.items():
        expected = KEYS[key]
        # `timeout = 60` is an int in TOML and obviously means 60.0 seconds; `True` is an int in
        # Python and is never what any of these settings mean.
        number = expected is float and isinstance(value, int)
        if (not isinstance(value, expected) and not number) or isinstance(value, bool):
            raise ConfigError(
                f"{path}: {key} must be a {_NAMES[expected.__name__]}, not {type(value).__name__}.",
                hint=f"Write it as {key} = {_EXAMPLES[expected.__name__]}.",
            )
        checked[key] = float(value) if expected is float else value
    return checked
