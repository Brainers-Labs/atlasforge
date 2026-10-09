# Failure-mode flags

A score tells you *how often* a model was right. It does not tell you *how* it was wrong. AtlasForge
also counts a small set of **failure-mode flags**: plain rules over the answer and the reference,
computed for every answer that came back.

!!! quote "What these are not"
    A flag is not a quality score and not a **hallucination rate**. There is no judge model
    anywhere in AtlasForge (decision D023): judging would need a second model, and judging N-ATLaS
    with a non-N-ATLaS model risks the submission. Every flag here is a rule you can read, re-run
    and argue with. An answer can be flagged and still be correct.

{{ flags_table() }}

Flags appear in `report.md`, `report.json` and `report.html` for a single run, and side by
side (base vs candidate, with a delta) in `comparison.md` and `comparison.html`. The per-example
flags are in `report.json` under `per_example_flags`, so you can pull the flagged rows out and
read them.

## What counts as a flag, exactly

**Empty output.** The call succeeded and the answer was blank. A *failed* call is not flagged: it
is counted on its own as a failure, and scored as wrong. Flags describe answers we received.

**Repeated output.** One word makes up half or more of an answer of eight words or more, or a
four-word phrase appears three times. Speech models loop; so do small language models.

**Truncated output.** The reference ends in a full stop, and the answer is five words or longer and
does not. That is what a cut-off generation looks like. A reference with no final punctuation
proves nothing, so it never flags.

**Format non-compliance.** The example asked for a shape and the answer is not that shape. Two ways
to ask:

```json
{"id": "a", "input": "Is mango a fruit?", "reference": "fruit", "meta": {"format": "number"}}
```

`meta.format` may be `json` (the answer must parse) or `number` (the answer must *be* a number, not
a sentence containing one). If the reference itself is JSON, that is taken as the requested format
even without `meta`. Any other value of `meta.format` is ignored rather than guessed at.

**Missing required terms.** An entity or acronym in the reference that the answer does not contain.
"A capitalised word" means a word capitalised anywhere except the start of a sentence — `The
capital of Nigeria is Abuja.` requires *Nigeria* and *Abuja*, not *The*. An acronym at the start of
a sentence (`NECO was founded ...`) still counts.

If the heuristic is not what you want, say what you require:

```json
{"id": "b", "input": "Give the dosage", "reference": "5 mg twice daily", "meta": {"require": ["mg", "twice daily"]}}
```

`meta.require` replaces the entity heuristic for that example. Terms are matched the tone-insensitive
way, the same as every metric: dropping tone marks is not a failure, but swapping `ẹ` for `e` is.

**Number mismatch.** The set of numbers in the answer differs from the set in the reference.
Thousands separators are ignored, so `1,000` and `1000` agree. `3.5` and `3` do not. A missing
number is the most common failure on Nigerian numeracy tasks, and this is the rule that finds it.

**Script mismatch.** A substantial share of the answer (20% or more of its letters, in an answer of
ten letters or more) is in a non-Latin script. This is **script statistics, not language
identification** (D024): Hausa, Yoruba, Igbo and English are all written in Latin script, so a
fluent answer in the wrong one of those four is *invisible* to this check, and AtlasForge does not
pretend otherwise. What it catches is an answer in the wrong alphabet altogether.

## Reading the numbers

Counts are of *answers*, over the answers that came back. `report.md` gives the total that had at
least one flag; `comparison.md` gives the base and candidate count for each flag and the
difference. A flag that the candidate raises more often than the base, next to a metric that went
the other way, is a lead worth chasing — that is the whole point of flagging rather than scoring.

The demo data shows it working: the tuned model regresses on arithmetic, and the comparison picks
up 8 more `number_mismatch` flags than the base model.
