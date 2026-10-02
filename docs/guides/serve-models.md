# Serve the models

The N-ATLaS LLM is an 8-billion-parameter model whose weights are 16.08 GB. Most laptops cannot hold that in memory, and there is no public hosted API. The practical answer is to **serve the model somewhere** and let AtlasForge talk to it over the OpenAI-compatible protocol (`--backend openai`).

This page covers how to serve it, what has been verified, and the traps to avoid.

!!! abstract "Verification status of each route"
    | Route | Status |
    |---|---|
    | **Ollama**, int4 import of the official weights, on an Apple M1 16 GB | **VERIFIED** end to end through `atlasforge run` (October 2026). Quantised, so quality is not representative |
    | vLLM on an NVIDIA GPU | **UNKNOWN**: not yet tested (no GPU available so far) |
    | llama.cpp (`llama-server`) | **INFERRED** to work for a Llama-architecture model; not tested |
    | A community gateway such as N-ATLAS Kit | not tested by us |

## What AtlasForge needs from a server

- The OpenAI chat endpoint: `POST {base_url}/chat/completions`.
- A base URL ending in `/v1`, for example `http://127.0.0.1:11434/v1`.
- A model name that matches what the server serves (`--model`).
- For speech over HTTP: `POST {base_url}/audio/transcriptions` (multipart, Whisper-style).
- Optionally an API key. Set the environment variable `ATLASFORGE_API_KEY`; AtlasForge sends it as `Authorization: Bearer ...`. (The CLI has no `--api-key` flag, so the key never appears in your shell history or process list.)

AtlasForge sends `temperature`, `max_tokens`, `repetition_penalty` and, when set, `top_p` and `seed`. It refuses plain `http://` to anything but `localhost`, `127.0.0.1` and `::1` unless you pass `--allow-insecure-http`.

Sanity-check a server before pointing AtlasForge at it:

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"natlas-local-chat","messages":[{"role":"user","content":"Hello"}]}'
```

## Ollama: quantised local serve

This is the route that has been **run for real**. It lets a 16 GB Mac serve N-ATLaS as a 6.6 GB int4 model. Two warnings first:

!!! danger "Licence: keep it local"
    The N-ATLaS licence restricts redistributing and deriving from the weights. A quantised copy for your own testing on your own machine is one thing; **publishing or sharing quantised N-ATLaS weights is not something AtlasForge does or supports.** Read [Licence and compliance](../natlas/licence.md).

!!! warning "Quantisation changes the model"
    An int4 model is a degraded copy. Scores from it describe the int4 model, not the official fp16 one. Always record how a model was served when you report numbers.

### 1. Download the official weights

Accept the licence on [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS) first (see [Models and access](../getting-started/models-and-access.md)), then:

```bash
pip install huggingface_hub
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; print(snapshot_download('NCAIR1/N-ATLaS'))"
```

It prints the snapshot directory. About 16 GB is downloaded.

!!! tip "If the download fails with `CAS Client Error`"
    Hugging Face's newer chunked-transfer path (Xet) failed repeatedly in testing with `CAS Client Error: ... error sending request`. Setting `HF_HUB_DISABLE_XET=1` forces plain HTTP downloads and fixed it. Re-running resumes from the cache.

### 2. Import and quantise

Create a file named `Modelfile` that points at the snapshot directory:

```text
FROM /path/to/models--NCAIR1--N-ATLaS/snapshots/<revision>
PARAMETER temperature 0.1
PARAMETER repeat_penalty 1.12
PARAMETER num_predict 1000
```

```bash
ollama create natlas-local-q4 -f Modelfile --quantize int4
```

Ollama converts the safetensors weights and quantises them. The accepted `--quantize` types in Ollama 0.34.4 are `int4`, `int8`, `nvfp4`, `mxfp4` and `mxfp8`; older releases used GGUF names such as `q4_K_M`, so check `ollama create --help`. `ollama show natlas-local-q4` should report `parameters 8.0B`, `context length 131072`, `quantization int4`.

### 3. Restore the chat template (important)

Ollama's import **does not carry over the model's chat template**. Right after import, `ollama show natlas-local-q4 --template` prints only `{{ .Prompt }}`, which means chat messages are sent without role headers. The model still answers, but not in the format it was trained on.

The official template is the Llama-3.1-Instruct format. You can render it from the tokenizer to see exactly what it produces:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("NCAIR1/N-ATLaS")
print(repr(tok.apply_chat_template([{"role": "user", "content": "USERMSG"}],
                                   tokenize=False, add_generation_prompt=True)))
```

```text
'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nUSERMSG<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
```

Layer a matching template and the stop tokens on top of the imported model (this is cheap and does not re-quantise). Save as `Modelfile.chat`:

```text
FROM natlas-local-q4
TEMPLATE """{{- if ne (index .Messages 0).Role "system" }}<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|>{{ end }}
{{- range .Messages }}
{{- if eq .Role "system" }}<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

{{ .Content }}<|eot_id|>
{{- else }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>

"""
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|eom_id|>"
PARAMETER temperature 0.1
PARAMETER repeat_penalty 1.12
PARAMETER num_predict 1000
```

```bash
ollama create natlas-local-chat -f Modelfile.chat
```

!!! note "What this template does and does not cover"
    It reproduces the default system block and the user/assistant turn format. It does **not** implement the tool-calling parts of the full template. The beginning-of-text token is left to Ollama to add; whether it is added exactly once was not separately verified.

### 4. Use it from AtlasForge

```bash
atlasforge run "What is the capital of Nigeria? Answer in one sentence." \
  --base-url http://127.0.0.1:11434/v1 --model natlas-local-chat
```

```text
The capital of Nigeria is Abuja.
```

In the verification run, English, Yoruba and Igbo replies were fluent, an Igbo reply contained a factual error, and a Hausa reply did not answer the question. That is one sample on a quantised model, and must not be read as the official model's quality. Typical latency on an M1 16 GB was roughly 1 to 12 seconds for short replies, the first including model load.

## vLLM

!!! warning "Not yet verified"
    No NVIDIA GPU has been available to test this. Treat the commands below as the standard vLLM workflow, not a tested N-ATLaS recipe.

vLLM serves an OpenAI-compatible API and batches requests well, which is why `--concurrency` helps with it.

```bash
pip install vllm
vllm serve NCAIR1/N-ATLaS --api-key "$SERVER_KEY"
```

```bash
export ATLASFORGE_BASE_URL=http://<host>:8000/v1
export ATLASFORGE_API_KEY=$SERVER_KEY
atlasforge eval data.jsonl --out runs/base --model NCAIR1/N-ATLaS --concurrency 8
```

Open questions to confirm and record in the [verification log](../natlas/model-facts.md#our-verification-log): whether `vllm serve` works with the repo unmodified, memory use at fp16, and speed. The model's `config.json` allows a 131072-token context, which can be far more than a given GPU can hold; vLLM's `--max-model-len` lowers the limit to fit memory.

If you expose a server beyond your own machine: use **HTTPS**, an API key per user, rate limits and a spending cap, and do not log prompt content. See [Security and privacy](../operations/security-privacy.md).

## llama.cpp

!!! warning "Not yet verified"
    llama.cpp is expected to work because the model is a standard Llama architecture, but nothing has been run.

The usual flow is to convert the Hugging Face weights to GGUF with llama.cpp's conversion script, optionally quantise, then run `llama-server`, which exposes an OpenAI-compatible API. As with Ollama, check that the chat template is applied correctly and that the model name you pass matches.

## Pitfalls checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| `404` from the endpoint | wrong base URL or model name | the URL must end in `/v1`; `--model` must equal the served name |
| `400` mentioning `repetition_penalty` | the server rejects that field | use the Python API with `send_repetition_penalty=False` |
| `401`/`403` | missing or wrong key | set `ATLASFORGE_API_KEY` |
| `429` | rate limited | lower `--concurrency`; AtlasForge retries `429` with backoff |
| refused plain `http://` | non-local host over http | use `https://` or `--allow-insecure-http` |
| replies look unformatted or odd | chat template missing | restore the template as above |
| very slow or out of memory | model too big for the machine | quantise, shorten `--max-new-tokens`, or serve on a GPU |

More in [Troubleshooting](../operations/troubleshooting.md).
