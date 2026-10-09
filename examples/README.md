# Example output

**Everything under `reports/` is synthetic.** It was not produced by N-ATLaS or by any model. It is
here so you can read the shape of a real report — and see the quality of the HTML page — without a
GPU, a token or a model download. The run manifests inside say the same thing.

## What is in `reports/`

| File | What it is |
|---|---|
| `report.md` | The Markdown report for one run: metrics under both tone views, failure-mode flags, slice table |
| `report.json` | The same as structured data — this is what `compare` and `card` read |
| `report.html` | The self-contained HTML page: inline SVG charts, no JavaScript, no external reference |
| `comparison.md` | Two runs compared example by example: the headline difference, McNemar, the regressions |
| `comparison.json` | The same as structured data |
| `comparison.html` | The self-contained HTML page for the comparison |

The numbers are the demo's deliberate ones: the "tuned" model is better overall and **worse** on
arithmetic, so `has_number` regresses. A report that only shows improvement is not a useful example.

## Regenerating them

```bash
python -m atlasforge demo /tmp/atlasforge-demo
cd /tmp/atlasforge-demo
python -m atlasforge report runs/base --dataset toy_qa.jsonl
python -m atlasforge compare toy_qa.jsonl --base runs/base --candidate runs/tuned --slice domain
```

The output is deterministic and free of paths and timestamps, so regenerating it byte-for-byte is a
test (`tests/unit/test_examples.py`, "the committed reports are the ones this code produces"). If a
change alters a report, that test fails and these files are regenerated in the same commit — they
cannot drift away from the code that made them.
