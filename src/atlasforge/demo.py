"""Synthetic demo data: try AtlasForge without any model, token, GPU or API.

``atlasforge demo`` writes a small toy dataset and two pre-computed runs, a "base" and a
"tuned" model, so the whole evaluate -> compare workflow can be followed offline.

**Everything here is synthetic.** The answers come from fixed rules, not from N-ATLaS or any
other model, and the manifests say so. The data is shaped to show every feature:

* an overall improvement (the tuned model is better at classifying fruit and vegetables),
* a hidden regression (it is worse at arithmetic, so the ``has_number`` slice regresses),
* tone-only differences (the base model drops Yoruba tone marks),
* failed calls (counted as wrong, never hidden),
* a slice too small to judge (12 Yoruba items, below the 30-example minimum).

The Yoruba items are a plain *copy-exactly* task, so they test tone handling without making
any claim about translation. The phrases are common greetings but have not been reviewed by a
native speaker.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from atlasforge.errors import BackendTimeout, ConfigError
from atlasforge.eval.dataset import load_dataset
from atlasforge.eval.normalize import strip_tones
from atlasforge.eval.runner import MANIFEST_NAME, RunConfig, run
from atlasforge.types import BackendInfo, Generation, GenParams, Lang, Message, Transcript

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from atlasforge.types import AudioInput

DATASET_NAME: Final = "toy_qa.jsonl"
BASE_NAME: Final = "base"
TUNED_NAME: Final = "tuned"
_FIXED_TIME: Final = "2026-01-01T00:00:00+00:00"

_FRUITS: Final = (
    "mango", "orange", "banana", "pawpaw", "guava", "pineapple", "watermelon", "apple",
    "grape", "lemon",
)  # fmt: skip
_VEGETABLES: Final = (
    "cabbage", "carrot", "onion", "lettuce", "spinach", "beetroot", "garlic", "celery",
    "broccoli", "cauliflower",
)  # fmt: skip
_YORUBA: Final = (
    "Ẹ káàárọ̀",
    "Ẹ ṣé",
    "Ẹ káàbọ̀",
    "Ẹ jọ̀wọ́",
    "Ẹ má bínú",
    "Ẹ kú iṣẹ́",
    "Ó dàbọ̀",
    "Mo wà dáadáa",
    "Báwo ni?",
    "Ẹ kú ọ̀dún",
    "Ẹ kú àbọ̀",
    "Ẹ kú alẹ́",
)  # fmt: skip
_BASE_TIMEOUTS: Final = frozenset({5, 21})  # agri indices where the base run "timed out"


@dataclass(frozen=True, slots=True, kw_only=True)
class Item:
    """One synthetic question with the correct answer and both models' answers."""

    id: str
    prompt: str
    reference: str
    domain: str
    lang: Lang
    base: str
    tuned: str
    base_times_out: bool = False


def build_items() -> list[Item]:
    """The 92 synthetic items, fully determined by fixed arithmetic rules."""
    items: list[Item] = []

    # Fruit or vegetable (40). Base is right 60% of the time, tuned about 92%.
    pool = [(x, "fruit") for x in _FRUITS] + [(x, "vegetable") for x in _VEGETABLES]
    for k in range(40):
        name, answer = pool[k % 20]
        prompt = (
            f"Is {name} a fruit or a vegetable?"
            if k < 20
            else f"Classify {name}: fruit or vegetable?"
        )
        wrong = "vegetable" if answer == "fruit" else "fruit"
        items.append(
            Item(
                id=f"agri-{k:03d}",
                prompt=prompt,
                reference=answer,
                domain="agri",
                lang="en",
                base=answer if (k * 7 + 3) % 10 < 6 else wrong,
                tuned=answer if (k * 3 + 1) % 13 != 0 else wrong,
                base_times_out=k in _BASE_TIMEOUTS,
            )
        )

    # Arithmetic (40). Base is right 85% of the time, tuned only 65%: a regression.
    for m in range(40):
        a, b = 10 + m * 3, 7 + (m * 5) % 23
        answer = str(a + b)
        items.append(
            Item(
                id=f"num-{m:03d}",
                prompt=f"What is {a} + {b}?",
                reference=answer,
                domain="numeracy",
                lang="en",
                base=answer if (m * 11 + 2) % 20 >= 3 else str(a + b + 1),
                tuned=answer if (m * 7 + 5) % 20 >= 7 else str(a + b - 1),
            )
        )

    # Copy a Yoruba phrase exactly (12). Base drops tone marks, tuned mostly keeps them.
    for y, phrase in enumerate(_YORUBA):
        reference = unicodedata.normalize("NFC", phrase)
        items.append(
            Item(
                id=f"yo-{y:02d}",
                prompt=f"Repeat exactly: {reference}",
                reference=reference,
                domain="greetings",
                lang="yo",
                base=strip_tones(reference),
                tuned=strip_tones(reference) if y in (3, 8) else reference,
            )
        )
    return items


class SyntheticBackend:
    """Answers from fixed rules. Not a model, and not registered as a backend."""

    def __init__(self, items: Sequence[Item], *, which: str) -> None:
        self._which = which
        self._answers = {i.prompt: (i.base if which == BASE_NAME else i.tuned) for i in items}
        self._timeouts = {i.prompt for i in items if which == BASE_NAME and i.base_times_out}
        self._calls = 0

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:  # noqa: ARG002
        prompt = messages[-1]["content"]
        if prompt in self._timeouts:
            raise BackendTimeout("synthetic timeout (demo data)")
        self._calls += 1
        latency = (850.0 if self._which == BASE_NAME else 1100.0) + (self._calls * 37) % 220
        return Generation(text=self._answers[prompt], latency_ms=latency)

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        raise NotImplementedError("the demo backend only generates text")

    def info(self) -> BackendInfo:
        return BackendInfo(
            backend="synthetic",
            model=f"synthetic-demo/{self._which}",
            revision="sample-data",
            capabilities=frozenset({"generate"}),
        )

    def close(self) -> None:
        return None


def build_demo(out_dir: Path) -> Path:
    """Write the demo dataset and both runs into ``out_dir`` (must not hold other files)."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ConfigError(
            f"{out_dir} already exists and is not empty.",
            hint="Choose a new folder, e.g. `atlasforge demo my-demo`.",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_items()

    dataset_path = out_dir / DATASET_NAME
    rows = [
        {
            "id": i.id,
            "input": i.prompt,
            "reference": i.reference,
            "lang": i.lang,
            "meta": {"domain": i.domain},
        }
        for i in items
    ]
    dataset_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
        newline="\n",
    )
    dataset = load_dataset(dataset_path, "generation")

    for which in (BASE_NAME, TUNED_NAME):
        run_dir = out_dir / "runs" / which
        backend = SyntheticBackend(items, which=which)
        run(backend, dataset, run_dir, config=RunConfig(max_consecutive_failures=0))
        _freeze_manifest(run_dir)
    return out_dir


def _freeze_manifest(run_dir: Path) -> None:
    """Pin the volatile manifest fields so the demo files are identical on every machine."""
    path = run_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["started_at"] = _FIXED_TIME
    manifest["atlasforge_version"] = "sample-data"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
