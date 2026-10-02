# Serve N-ATLaS

AtlasForge's `openai` backend talks to any server that speaks the OpenAI-compatible protocol. This page shows three ways to run N-ATLaS behind one.

!!! danger "None of these have been tested with N-ATLaS yet"
    The commands below follow each project's own documentation as we understand it. They have **not** been run against the real `NCAIR1/N-ATLaS` weights, and each tool's flags change between releases, so check its current docs. Whether a given server handles N-ATLaS's chat template correctly is exactly what still needs verifying ([Project status](../help/status.md)). If you try one, please report what happened.

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

1. Import your GGUF file with a `Modelfile` containing `FROM ./natlas-q4_k_m.gguf`, then `ollama create natlas -f Modelfile`.
2. Ollama serves an OpenAI-compatible API at `http://127.0.0.1:11434/v1`:

    ```bash
    atlasforge run "Ina kwana?" \
      --base-url http://127.0.0.1:11434/v1 \
      --model natlas \
      --no-send-repetition-penalty
    ```

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
