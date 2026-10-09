"""Write ``evidence/live_<date>.md`` from a machine that has the real N-ATLaS models.

    python scripts/live_smoke.py --out evidence
    python scripts/live_smoke.py --quantize 4bit --audio-dir clips --base-url http://127.0.0.1:8000/v1

This is the script ``planning/16_SUBMISSION_CHECKLIST.md`` asks for and ``planning/23`` lists as
part of G3: the eight-row verification log in ``planning/21`` cannot be filled in from a machine
without the weights, and this is the run that fills it.

Needs ``HF_TOKEN``, the Awarri licence accepted on all five NCAIR1 repositories, and the local
extra (``pip install "brainers-atlasforge[local]"``, which brings torch, transformers and the Hub
client). ``--audio-dir`` wants one clip per language named ``ha``, ``yo``, ``ig``, ``en``.

Nothing is inferred. A check that cannot run is written down as ``not run`` and stays outstanding —
a blank cell is more honest than a plausible number, and this file is submitted as evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from atlasforge.backends.factory import DEFAULT_MODEL
from atlasforge.livecheck import (
    collect,
    environment_facts,
    environment_secrets,
    evidence_filename,
    gpu_facts,
    hub_revisions,
    live_steps,
    render,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options. Mirrors the flags ``atlasforge run`` already uses where they overlap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evidence"), help="directory to write to")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the LLM repository to load")
    parser.add_argument(
        "--quantize",
        default="none",
        choices=("none", "4bit", "8bit"),
        help="4bit needs an NVIDIA GPU; 'none' is fp16",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    parser.add_argument(
        "--base-url",
        default=None,
        help="an OpenAI-compatible server to check too, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--audio-dir", type=Path, default=None, help="one clip per language")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    secrets = environment_secrets()
    try:
        revisions = hub_revisions()
    except Exception as exc:  # no Hub client or no network: the rest of the run still counts
        print(f"could not read commit SHAs ({type(exc).__name__}: {exc})")
        revisions = {}
    evidence = collect(
        live_steps(
            model=args.model,
            quantize=args.quantize,
            device=args.device,
            base_url=args.base_url,
            audio_dir=args.audio_dir,
        ),
        shas=revisions,
        facts={**environment_facts(), **gpu_facts()},
        secrets=secrets,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / evidence_filename()
    target.write_text(render(evidence), encoding="utf-8")
    print(f"{target}: {evidence.verified} of {len(evidence.rows)} checks verified")
    for row in evidence.rows:
        if row.status != "verified":
            print(f"  {row.status}: {row.check} — {row.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
