# Python API Design

Design targets. Signatures may change during implementation, but the shape should not.

## Backends

```python
from atlasforge import load_backend

llm = load_backend("local", model="NCAIR1/N-ATLaS", quantize="4bit")      # needs atlasforge[local]
llm = load_backend("openai", base_url="http://gpu-box:8000/v1",
                   model="NCAIR1/N-ATLaS", api_key_env="ATLASFORGE_API_KEY")

gen = llm.generate([{"role": "user", "content": "Fassara zuwa Turanci: Ina kwana?"}])
print(gen.text, gen.latency_ms, gen.usage)
```

Defaults (from the model card, VERIFIED): `temperature=0.1`, `repetition_penalty=1.12`, `max_new_tokens` capped by the context window. `repetition_penalty` is not a standard OpenAI parameter. The `openai` backend sends it as vLLM `extra_body` and records if the server ignored it.

## ASR

```python
from atlasforge.asr import transcribe

t = transcribe("voice-note.ogg", lang="yo", backend="local")
print(t.text)
for c in t.chunks:             # offsets are computed by our chunker, not model timestamps
    print(c.start_s, c.end_s, c.text)
```

Languages: `ha`, `yo`, `ig`, `en` (Nigerian-accented English). There is no automatic language detection: each ASR model is monolingual. Detection could be added later only as a clearly labelled heuristic.

## Evaluation

```python
from atlasforge.eval import evaluate

report = evaluate(
    backend=llm,
    dataset="examples/translation_ha_en.jsonl",
    task="generation",
    metrics=["chrf", "exact_match"],
    normalize="auto",          # per-lang rules; reports tone-aware and tone-insensitive
    out_dir="runs/ha_en_base",
)
print(report.summary())
```

Custom metric:

```python
def mentions_dosage(pred, ref, ex) -> float:
    return float("mg" in pred)

evaluate(..., metrics=["chrf", mentions_dosage])
```

## Compare

```python
from atlasforge.compare import compare

cmp = compare(base="runs/ha_en_base", candidate="runs/ha_en_lora", metric="chrf", n_boot=1000, seed=0)
cmp.to_markdown("runs/comparison.md")
```

## Errors

```text
AtlasForgeError
├── ConfigError             bad/missing config, missing extra ("pip install atlasforge[local]")
├── ModelAccessError        gated repo not accepted / HF token missing or invalid
├── ResourceError           insufficient VRAM/RAM/disk, no ffmpeg
├── BackendError
│   ├── BackendTimeout      (not named TimeoutError, to avoid shadowing the builtin)
│   ├── BackendConnectionError
│   └── BackendHTTPError    status + body excerpt, request id if provided
├── AudioError              unreadable/unsupported audio
└── DatasetError            invalid JSONL line (with line number)
```

Each error carries a `hint` with the next step to take. `atlasforge doctor` reuses the same checks.

## Configuration

Precedence (highest first): explicit argument → CLI flag → environment variable → `atlasforge.toml` in the project → defaults.

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Hugging Face token (standard `huggingface_hub` variable) used for gated downloads |
| `ATLASFORGE_BASE_URL` | default OpenAI-compatible endpoint |
| `ATLASFORGE_API_KEY` | key for that endpoint |
| `ATLASFORGE_CACHE` | results/cache directory |

## Compatibility stance

Speaking the OpenAI-compatible protocol is a **transport choice**. It is how vLLM and most gateways serve open models, not a claim that N-ATLaS is OpenAI. The model behind the endpoint must be an NCAIR1 model. The `info()` call records which one.
