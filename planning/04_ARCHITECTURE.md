# Technical Architecture

```text
                Developer
                   |
        +----------+-----------+
        |                      |
      CLI (typer)        Python API (atlasforge.*)
        |                      |
        +----------+-----------+
                   v
   +---------------------------------------------+
   |               AtlasForge core               |
   |                                             |
   |  eval/       tasks, runner, metrics,        |
   |              normalize (NG langs), report   |
   |  compare/    paired bootstrap, report       |
   |  finetune/   llm_qlora, asr_whisper   (P1)  |
   |  cards/      license-aware model cards (P1) |
   |  asr/        audio io, chunking, routing    |
   |                                             |
   |  backends/   Backend protocol               |
   |     +-- local.py    transformers (+4-bit)   |
   |     +-- openai.py   OpenAI-compatible HTTP  |
   |     +-- official.py NAIC API (post-shortlist)|
   +---------------------------------------------+
            |                         |
            v                         v
   NCAIR1/N-ATLaS (LLM)      NCAIR1/{Hausa,Yoruba,Igbo}-ASR,
   via HF weights or          NCAIR1/NigerianAccentedEnglish
   vLLM endpoint              via HF weights or endpoint
```

## Key design decisions

1. **Backend protocol, not an HTTP client.** N-ATLaS has no public API yet, so the core depends on a small protocol:
   ```python
   class Backend(Protocol):
       def generate(self, messages: list[Message], params: GenParams) -> Generation: ...
       def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript: ...
       def info(self) -> BackendInfo: ...   # model id, revision, device, dtype
   ```
2. **Eval never knows which backend it is using.** The same dataset and metrics run against local weights, a vLLM server or a future official API.
3. **Model identity is recorded, not assumed.** Every report stores the HF repo ID plus commit SHA (local) or the served model name (endpoint).
4. **Heavy dependencies are optional.** torch, transformers, bitsandbytes, peft, trl and librosa sit behind extras and are imported lazily. Missing extras raise a clear `pip install atlasforge[local]` message.
5. **Normalization lives in one module** (`eval/normalize.py`) with per-language rules and tests, because it decides whether scores mean anything.
6. **CLI stays thin over the Python API.** All logic is importable and testable.
7. **Never log credentials, prompts or audio by default.** `--log-samples` opts in.

## Repository layout

```text
atlasforge/
├── README.md  LICENSE  CONTRIBUTING.md  CODE_OF_CONDUCT.md  SECURITY.md  CHANGELOG.md
├── pyproject.toml
├── src/atlasforge/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── types.py            # Message, GenParams, Generation, Transcript, Example
│   ├── errors.py
│   ├── backends/{base,local,openai}.py
│   ├── asr/{audio,chunking,models}.py
│   ├── eval/{dataset,runner,metrics,normalize,report}.py
│   ├── compare/{bootstrap,report}.py
│   ├── finetune/{llm_qlora,asr_whisper}.py      # P1
│   └── cards/model_card.py                      # P1
├── tests/{unit,contract,live}/
├── examples/                # datasets + scripts for J1–J3
├── notebooks/               # Colab J1, J2
├── benchmarks/              # starter JSONL packs + provenance/licence per pack
├── deploy/vllm/             # documented vLLM command + compose for the beta endpoint
└── docs/                    # user-facing docs (mkdocs)
```

Note: this `docs/` folder (planning) should move to `planning/` once the product repo exists, so it doesn't collide with user docs.

## Serving strategy for v0.1

- We do **not** build a gateway. For endpoint use we document `vllm serve NCAIR1/N-ATLaS` (to verify on Day 1–2) and support any OpenAI-compatible server.
- For the beta period we run **one temporary shared vLLM endpoint** on a rented GPU with per-tester API keys and a spending cap, so testers skip the 16 GB download. It is shut down after validation.
- ASR over an endpoint: target the `/v1/audio/transcriptions` shape so AtlasForge also works against community ASR gateways. For our beta, testers run ASR locally: 244M-parameter models are CPU-feasible.
