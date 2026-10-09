# Benchmark with AfroBench-LITE

AfroBench-LITE is the multilingual suite a published study used to evaluate N-ATLaS: seven task
families covering the languages the official models target. This page does not reproduce its
figures — see the note at the end.

It is a *research* benchmark with its own runner, EleutherAI's
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness). AtlasForge does not
reimplement the tasks. `atlasforge bench afrobench` builds the harness command for an official
`NCAIR1` model, records what came back, and writes it in the same shape as every other AtlasForge
report — so these figures sit beside the ones from `eval` and `compare`.

It needs the `bench` extra, which brings the harness and the `torch`/`transformers` stack it loads
models with:

```bash
pip install "brainers-atlasforge[bench]"
```

## The seven families

| Family | Covers |
|---|---|
| `afrixnli` | Natural-language inference across African languages |
| `belebele` | Reading comprehension |
| `afrimmlu` | Multiple-choice knowledge |
| `flores` | Translation |
| `sib` | A culturally grounded completion task |
| `injongo` | Intent classification |
| `afrimgsm` | Grade-school mathematics |

Families are matched as substrings of the harness's **own** task names, which is why AtlasForge
never ships a task list of its own: the harness names them per language (`afrixnli_yo`,
`belebele_hau`) and that set changes between harness versions.

## See what this harness has

```bash
atlasforge bench afrobench --list
```

That prints the harness task names this installation matches, followed by a line for every family
it has nothing for. **A family with no match is absent from the report, never a zero** — "we did not
run it" and "it scored nothing" are not the same claim.

## Check the command before running it

```bash
atlasforge bench afrobench --dry-run
```

Nothing is executed; the exact `lm_eval` command is printed, so you can see the task list, the
model arguments and where the harness output will go.

## Run it

Loading the official weights needs a GPU, a Hugging Face token and the Awarri licence accepted on
each `NCAIR1` model page — see [access and licences](../get-started/access-and-licences.md).

```bash
atlasforge bench afrobench --out benches/afrobench --device cuda:0 --batch-size 8
```

Pin a revision and pass the harness's own task names when you want to be explicit, or when a
harness version spells a family differently:

```bash
atlasforge bench afrobench --tasks afrixnli_yo --tasks belebele_hau --revision <commit-sha>
```

Every other harness setting is passable too: `--few-shot`, `--chat-template`, `--model`,
`--batch-size` and `--device`.

## What it writes

| File | What it is |
|---|---|
| `bench.json` | Machine-readable: every numeric figure per task, plus model, revision, harness version, few-shot setting and the `raw_path` below |
| `bench.md` | The same as a table, with the headline metric per task |
| `harness/results.json` | The harness's own output, kept verbatim |

The harness's file is the one place this wrapper does not decide for you: `--output_path` is a
request, and some harness versions treat it as a directory and invent a name underneath. The report
names the file that was actually found, in `raw_path` and in the prose of `bench.md`, so it never
points at a path that isn't there.

**Few-shot is recorded, including when it was not set.** `--few-shot` is passed to the harness as
`--num_fewshot`; leave it off and each task runs with whatever the harness itself defaults to for
that task, which is not necessarily zero-shot. The report says which of the two happened rather
than printing a `0` it cannot vouch for — and that matters, because a score with examples in the
prompt is not comparable to one without, so a bare figure would be easy to quote out of context.

The headline figure per task is the first metric the harness reported from this list —
`acc_norm`, `acc`, `exact_match`, `f1`, `chrf`, `bleu` — and every metric it reported is listed
underneath, unscaled, so a figure can be checked against the harness's raw output without
arithmetic. Accuracy-like figures print as percentages and chrF and BLEU stay on the harness's own
0-100 scale, because one table holds a multiple-choice task and a translation task at once.

!!! warning "This wrapper has not been run against the real harness or weights"
    Its wiring is tested against stand-ins: a fake harness, a fake results file. Which task
    identifiers your installed harness exposes is discovered at run time by `--list`, and the
    figures in a report are whatever the harness on your machine measured. The published
    AfroBench-LITE numbers are someone else's measurements on a specific harness revision — they
    are recorded with their source in `planning/21_NATLAS_DISCOVERY.md` and are deliberately not
    copied into a report, where they would read as ours.
