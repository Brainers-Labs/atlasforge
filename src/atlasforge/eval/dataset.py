"""JSONL dataset loading with strict, line-numbered validation.

One JSON object per line. Recognised keys:

``id``         optional; unique. Derived from the content when omitted.
``input``      the user prompt (shorthand for a single user message), or
``messages``   a list of ``{"role", "content"}`` turns. Not both.
``audio``      path to an audio file (relative to the dataset file). ASR only.
``reference``  expected output. Required for ``classification`` and ``asr``.
``lang``       one of ha / yo / ig / en.
``meta``       free-form object; used for slice analysis (domain, length bucket, ...).

Unknown top-level keys are rejected so typos like ``refrence`` fail loudly.
Put extra columns under ``meta``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, get_args

from atlasforge.errors import AtlasForgeError, ConfigError, DatasetError
from atlasforge.types import parse_lang

if TYPE_CHECKING:
    from collections.abc import Mapping

    from atlasforge.types import Lang, Message

Task: TypeAlias = Literal["generation", "classification", "asr"]
TASKS: Final[tuple[Task, ...]] = get_args(Task)

_KEYS: Final = frozenset({"id", "input", "messages", "audio", "reference", "lang", "meta"})
_ROLES: Final = frozenset({"system", "user", "assistant"})
_REFERENCE_REQUIRED: Final = frozenset({"classification", "asr"})


@dataclass(frozen=True, slots=True, kw_only=True)
class Example:
    """One validated evaluation example."""

    id: str
    messages: tuple[Message, ...] = ()
    audio: Path | None = None
    reference: str | None = None
    lang: Lang | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Dataset:
    """A validated dataset plus the fingerprint used to detect silent changes."""

    task: Task
    path: Path
    sha256: str
    examples: tuple[Example, ...]

    def __len__(self) -> int:
        return len(self.examples)


def load_dataset(path: str | Path, task: Task) -> Dataset:
    """Read and validate ``path`` for ``task``. Raises :class:`DatasetError` with line numbers."""
    if task not in TASKS:
        raise ConfigError(f"Unknown task {task!r}.", hint=f"Use one of: {', '.join(TASKS)}.")
    file = Path(path)
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise DatasetError(
            f"cannot read {file}: {exc.strerror or exc}", hint="Check the path."
        ) from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"{file} is not valid UTF-8 (bad byte at offset {exc.start}).",
            hint="Re-save the file as UTF-8.",
        ) from exc

    examples: list[Example] = []
    seen: dict[str, int] = {}
    # Split on \n only: str.splitlines() would also split on U+2028/U+2029 inside JSON strings.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON ({exc.msg})", line=number) from exc
        example = _parse(obj, number, task, file.parent)
        if example.id in seen:
            raise DatasetError(
                f"duplicate id {example.id!r} (first seen on line {seen[example.id]})",
                line=number,
                hint="Give each example a unique 'id'.",
            )
        seen[example.id] = number
        examples.append(example)

    if not examples:
        raise DatasetError(f"{file} contains no examples.")
    return Dataset(
        task=task,
        path=file,
        sha256=hashlib.sha256(raw).hexdigest(),
        examples=tuple(examples),
    )


def _parse(obj: object, line: int, task: Task, base: Path) -> Example:
    if not isinstance(obj, dict):
        raise DatasetError("each line must be a JSON object", line=line)
    unknown = sorted(set(obj) - _KEYS)
    if unknown:
        raise DatasetError(
            f"unknown key(s): {', '.join(unknown)}",
            line=line,
            hint=f"Allowed keys: {', '.join(sorted(_KEYS))}. Put extra columns under 'meta'.",
        )

    messages = _messages(obj, line, task)
    audio = _audio(obj, line, task, base)
    reference = obj.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise DatasetError("'reference' must be a string", line=line)
    if reference is None and task in _REFERENCE_REQUIRED:
        raise DatasetError(f"'reference' is required for task {task!r}", line=line)

    lang: Lang | None = None
    if obj.get("lang") is not None:
        raw_lang = obj["lang"]
        if not isinstance(raw_lang, str):
            raise DatasetError("'lang' must be a string", line=line)
        try:
            lang = parse_lang(raw_lang)
        except AtlasForgeError as exc:
            raise DatasetError(exc.message, line=line, hint=exc.hint) from exc

    meta = obj.get("meta", {})
    if not isinstance(meta, dict):
        raise DatasetError("'meta' must be an object", line=line)

    return Example(
        id=_identify(obj, line, messages, audio, reference),
        messages=messages,
        audio=audio,
        reference=reference,
        lang=lang,
        meta=meta,
    )


def _messages(obj: dict[str, Any], line: int, task: Task) -> tuple[Message, ...]:
    has_input, has_messages = "input" in obj, "messages" in obj
    if task == "asr":
        if has_input or has_messages:
            raise DatasetError("task 'asr' takes 'audio', not 'input'/'messages'", line=line)
        return ()
    if has_input == has_messages:
        raise DatasetError("provide exactly one of 'input' or 'messages'", line=line)
    if has_input:
        content = obj["input"]
        if not isinstance(content, str) or not content.strip():
            raise DatasetError("'input' must be a non-empty string", line=line)
        return ({"role": "user", "content": content},)

    turns = obj["messages"]
    if not isinstance(turns, list) or not turns:
        raise DatasetError("'messages' must be a non-empty list", line=line)
    parsed: list[Message] = []
    for turn in turns:
        if not (
            isinstance(turn, dict)
            and turn.get("role") in _ROLES
            and isinstance(turn.get("content"), str)
        ):
            raise DatasetError(
                "each message needs a 'role' (system/user/assistant) and string 'content'",
                line=line,
            )
        parsed.append({"role": turn["role"], "content": turn["content"]})
    return tuple(parsed)


def _audio(obj: dict[str, Any], line: int, task: Task, base: Path) -> Path | None:
    if task != "asr":
        if "audio" in obj:
            raise DatasetError(f"'audio' is only valid for task 'asr', not {task!r}", line=line)
        return None
    value = obj.get("audio")
    if not isinstance(value, str) or not value:
        raise DatasetError("task 'asr' requires a string 'audio' path", line=line)
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise DatasetError(f"audio file not found: {path}", line=line)
    return path


def _identify(
    obj: dict[str, Any],
    line: int,
    messages: tuple[Message, ...],
    audio: Path | None,
    reference: str | None,
) -> str:
    raw_id = obj.get("id")
    if raw_id is None:
        payload = json.dumps(
            [messages, audio.name if audio else None, reference], ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or str(raw_id) == "":
        raise DatasetError("'id' must be a non-empty string or integer", line=line)
    return str(raw_id)
