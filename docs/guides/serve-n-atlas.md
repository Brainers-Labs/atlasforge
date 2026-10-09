# Serve N-ATLaS

AtlasForge's `openai` backend talks to any server that speaks the OpenAI-compatible protocol. This page shows three ways to run N-ATLaS behind one.

!!! warning "Only Option 3 has been run with the real N-ATLaS weights"
    **Option 3 (Ollama, int4)** was run end to end on an Apple M1 with 16 GB on 2 October 2026: the official `NCAIR1/N-ATLaS` safetensors were imported, quantised, served, and answered through `atlasforge run`. **Options 1 and 2 (vLLM, llama.cpp) have not been run with the real weights.** Their commands follow each project's own documentation as we understand it, and each tool's flags change between releases, so check its current docs. Whether a server applies N-ATLaS's chat template correctly is the thing to verify on any of them; Ollama, for one, does not do it by itself (see Option 3). If you try one, please report what happened ([Project status](../help/status.md)).

Before any of them: [get access to the model](../get-started/access-and-licences.md) and set `HF_TOKEN`.

## Option 1: vLLM (NVIDIA GPU)

Best for a Linux machine with a GPU of 24 GB or more.

```bash
pip install vllm
vllm serve NCAIR1/N-ATLaS          # serves http://127.0.0.1:8000/v1
```

Then point AtlasForge at it:

```bash
atlasforge run "Ina kwana?" --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
```

vLLM accepts the `repetition_penalty` field AtlasForge sends by default.

## Option 2: llama.cpp (Mac, CPU, small GPUs)

For machines that cannot hold the 8B model in full precision, run a **4-bit quantised** copy. This is the expected route on an Apple Silicon Mac with 16 GB.

1. Convert the Hugging Face weights to GGUF and quantise them, using the scripts in the llama.cpp repository (`convert_hf_to_gguf.py` and `llama-quantize`).
2. Serve the result:

    ```bash
    llama-server -m natlas-q4_k_m.gguf --port 8080 --repeat-penalty 1.12
    ```

3. Connect, **turning off** the field that llama.cpp does not use under that name:

    ```bash
    atlasforge run "Ina kwana?" \
      --base-url http://127.0.0.1:8080/v1 \
      --model natlas \
      --no-send-repetition-penalty
    ```

The `--repeat-penalty 1.12` on the server reproduces the model card's recommended setting.

## Option 3: Ollama (simplest on a laptop)

This is the route that has been run for real. Ollama imports the Hugging Face weights directly, so there is no GGUF conversion step, and quantises them as it imports. On a 16 GB Mac the result is a 6.6 GB model that fits.

!!! danger "Keep it on your machine"
    The N-ATLaS licence does not let you redistribute the weights, and a quantised copy is still the weights. Use it for your own testing and never upload or share it. See [Access and licences](../get-started/access-and-licences.md).

!!! warning "A quantised model is a degraded copy"
    Scores from an int4 model describe the int4 model, not the official fp16 one. Record how a model was served whenever you report numbers.

**1. Download the official weights** (about 16 GB) after accepting the licence on the model page:

```bash
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; print(snapshot_download('NCAIR1/N-ATLaS'))"
```

It prints the snapshot directory. `HF_HUB_DISABLE_XET=1` is there because Hugging Face's newer chunked-transfer path failed twice in testing with `CAS Client Error`; plain HTTP finished. Re-running resumes from the cache.

**2. Import and quantise.** Put the snapshot path in this `Modelfile`:

```text
--8<-- "examples/ollama/Modelfile"
```

```bash
ollama create natlas-local-q4 -f Modelfile --quantize int4
```

The accepted `--quantize` names differ between Ollama releases (`int4`, `int8`, `nvfp4`, `mxfp4` and `mxfp8` in 0.34.4; older ones used GGUF names such as `q4_K_M`), so check `ollama create --help`. `ollama show natlas-local-q4` should report `8.0B` parameters, a context length of `131072` and `int4`.

**3. Restore the chat template.** Ollama's safetensors import keeps the weights but **not the chat template**: `ollama show natlas-local-q4 --template` prints only the bare prompt field. The model still answers, but without the role headers it was trained on. Layer the official format on top of the import (cheap, and it does not quantise again):

```text
--8<-- "examples/ollama/Modelfile.chat"
```

```bash
ollama create natlas-local-chat -f Modelfile.chat
```

You can see exactly what the official template produces with `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)` from `transformers`. The Modelfile above reproduces it, including the default "Cutting Knowledge Date" system block, but not the tool-calling parts; whether the beginning-of-text token ends up added exactly once was not separately verified.

**4. Connect:**

```bash
atlasforge run "What is the capital of Nigeria? Answer in one sentence." \
  --base-url http://127.0.0.1:11434/v1 \
  --model natlas-local-chat
```

```text
The capital of Nigeria is Abuja.
```

The request succeeded without `--no-send-repetition-penalty`, so Ollama accepted the extra `repetition_penalty` field. We did not verify whether it applies it, which is why the Modelfile also sets `repeat_penalty 1.12` itself. Add the flag if you would rather the request carried only fields the server documents.

**What the verification run showed.** English, Yoruba and Igbo replies were fluent, an Igbo reply contained a factual error, and a Hausa reply did not answer the question. That is one sample each from a quantised model, so it says nothing reliable about the official model's quality. Short replies took about 1 to 12 seconds on the M1, the first including the model load.

**Alternative: from a GGUF.** If you already converted the weights with llama.cpp, import that instead with a `Modelfile` containing `FROM ./natlas-q4_k_m.gguf`, then `ollama create natlas -f Modelfile` and serve it the same way. Restoring the chat template applies here too.

## Remote servers

Plain `http://` is only allowed for `localhost`, `127.0.0.1` and `::1`. For a server on another machine use **https**, or opt in knowingly:

```bash
atlasforge run "..." --base-url http://192.168.1.20:8000/v1 --allow-insecure-http
```

Never expose a model server to the internet without authentication. If your server needs a key, put it in the `ATLASFORGE_API_KEY` environment variable; AtlasForge sends it as a bearer token and never prints it.

## Licence reminders

- Keep any converted or quantised copy **private to you**. Do not upload it: the licence does not let AtlasForge, or you, redistribute the weights without meeting its terms.
- A shared server used by many people counts toward the **1,000 active end-user cap**.
- Commercial use needs a separate agreement. See [Access and licences](../get-started/access-and-licences.md).

## Check that it works

```bash
atlasforge run "Hello" --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS --json
```

You should see the answer, the model name, the latency and token counts. If not, see [Troubleshooting](../help/troubleshooting.md).
