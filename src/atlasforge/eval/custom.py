"""Custom metrics: your own scoring function, scored like a built-in one.

A metric is any callable shaped

    def mentions_dosage(prediction: str, reference: str, example: Example) -> float

Use it from the library by putting the callable in ``metrics=[...]``, or from the command
line by naming it ``module:function`` (``--metric my_metrics:mentions_dosage``), where
``my_metrics`` is importable from the working directory.

Both strings are **normalised for the view being scored** — tone-aware or tone-insensitive,
lower-cased, punctuation stripped — exactly like every built-in metric, so your metric is
reported under both views and can be compared with the paired statistics. ``example`` is the
whole row, including the untouched ``example.reference`` and ``example.lang``.

Set ``higher_is_better = False`` on the function (a module-level or function attribute) when
a lower value is the better one, and the reports will say so instead of implying the
opposite. Custom metrics have no pooled corpus figure: only the mean over examples is
reported.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable  # used at runtime, in the alias below
from typing import TYPE_CHECKING, cast

from atlasforge.errors import ConfigError

if TYPE_CHECKING:
    from atlasforge.eval.dataset import Example

#: What a custom metric must look like.
MetricFn = Callable[[str, str, "Example"], float]

_HINT = (
    "Write `def my_metric(prediction, reference, example) -> float` in an importable "
    "module and pass it as --metric module:function."
)


def resolve(spec: str) -> MetricFn:
    """Import ``module:function`` and return the callable, or explain what is wrong.

    Importing runs the module. That is the point of a plug-in, but it does mean a
    ``--metric`` spec is as trusted as the shell you typed it in.
    """
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise ConfigError(
            f"Custom metric {spec!r} is not in 'module:function' form.",
            hint=_HINT,
        )
    try:
        module = importlib.import_module(module_name.strip())
    except ImportError as exc:
        raise ConfigError(
            f"Could not import {module_name.strip()!r} for metric {spec!r}: {exc}",
            hint="The module must be importable from where you run AtlasForge. " + _HINT,
        ) from exc
    try:
        found = getattr(module, attribute.strip())
    except AttributeError as exc:
        raise ConfigError(
            f"{module_name.strip()!r} has no {attribute.strip()!r}.",
            hint=_HINT,
        ) from exc
    if not callable(found):
        raise ConfigError(
            f"{spec!r} is not callable.",
            hint=_HINT,
        )
    return cast("MetricFn", found)


def name_of(fn: MetricFn) -> str:
    """The name a custom metric is reported under, taken from the function's ``__name__``."""
    name = getattr(fn, "__name__", "")
    if not isinstance(name, str) or not name.isidentifier():
        raise ConfigError(
            "A custom metric needs a usable __name__ (a Python identifier).",
            hint=_HINT,
        )
    return name


def declares_higher_is_better(fn: MetricFn) -> bool:
    """``fn.higher_is_better``, defaulting to ``True`` when the metric does not say."""
    return bool(getattr(fn, "higher_is_better", True))
