# Models and access

AtlasForge works with the five official N-ATLaS models published by NCAIR on Hugging Face. It never bundles, mirrors or redistributes them. You download them yourself, after accepting the licence with your own account.

## The five models

| Model | Hugging Face repo | What it is | Size |
|---|---|---|---|
| **N-ATLaS** (LLM) | [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS) | 8B-parameter chat model (Llama architecture), English, Hausa, Igbo, Yoruba | 16.08 GB (bf16) |
| Hausa ASR | [`NCAIR1/Hausa-ASR`](https://huggingface.co/NCAIR1/Hausa-ASR) | Whisper-based speech recognition | small |
| Yoruba ASR | [`NCAIR1/Yoruba-ASR`](https://huggingface.co/NCAIR1/Yoruba-ASR) | Whisper-based speech recognition | small |
| Igbo ASR | [`NCAIR1/Igbo-ASR`](https://huggingface.co/NCAIR1/Igbo-ASR) | Whisper-based speech recognition | small |
| Nigerian-accented English ASR | [`NCAIR1/NigerianAccentedEnglish`](https://huggingface.co/NCAIR1/NigerianAccentedEnglish) | Whisper-based speech recognition | small |

The language codes AtlasForge uses map to ASR models like this:

| Code | Language | Model |
|---|---|---|
| `ha` | Hausa | `NCAIR1/Hausa-ASR` |
| `yo` | Yoruba | `NCAIR1/Yoruba-ASR` |
| `ig` | Igbo | `NCAIR1/Igbo-ASR` |
| `en` | Nigerian-accented English | `NCAIR1/NigerianAccentedEnglish` |

You can also write `hausa`, `yoruba`, `igbo` or `english` anywhere a language is expected.

Detailed, individually verified facts about each model are in [Verified model facts](../natlas/model-facts.md).

## Getting access

All five repositories are **gated**. Listing a repository works without access, but downloading its files fails with a `403 GatedRepoError` until the licence is accepted on that specific repository.

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co).
2. **Accept the licence on each of the five model pages.** Open each link above while signed in and accept the terms. Approval is automatic for these repositories, but it is **per repository**: accepting one does not unlock the others.
3. **Create an access token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A read-only token is enough.
4. **Give the token to your machine**, using one of:

    ```bash
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx     # for this shell session
    ```

    or store a login once:

    ```bash
    hf auth login            # newer huggingface_hub
    huggingface-cli login    # older versions
    ```

5. **Check it:**

    ```bash
    atlasforge doctor
    ```

    The `hf-token` row should read `found (hf_****abcd)` or `cached login found`.

!!! danger "Treat the token like a password"
    Do not commit it, paste it into a chat or screenshot it. If you ever expose one, revoke it at the settings page and create a new one. AtlasForge only ever displays tokens masked, and never writes them to logs or reports.

!!! info "`doctor` checks that a token exists, not that your licences are accepted"
    A valid token without accepted licences still fails to download. The error then reads `Cannot access NCAIR1/...`, and the hint tells you to accept the licence on the model page. The quickest manual check is to open each model page while signed in.

## Do you need to download anything?

It depends on how you reach the models.

| You use | You download | Notes |
|---|---|---|
| `--backend openai` to someone else's endpoint | nothing | the server owner handles access and licensing |
| `--backend openai` to a server you run | the LLM weights, into that server | vLLM, llama.cpp or Ollama fetch or import them |
| `--backend local` for speech | the relevant ASR repo, automatically on first use | cached under `~/.cache/huggingface` |
| `--backend local` for the LLM | the full 16 GB | needs a large GPU; see [Run locally with transformers](../guides/local-backend.md) |

## There is no hosted API

There is **no public hosted N-ATLaS API**. The NAIC competition provides API credentials only to shortlisted teams, and Hugging Face does not offer an Inference Provider deployment for the ASR models. That is why AtlasForge talks to *any* OpenAI-compatible server and can load weights directly: you choose where the model runs.

## Licence

All five models are released under Awarri's "Open-Source Research and Innovation License", which has real obligations (an active-user cap, attribution, and a required name suffix for derivatives). Read [Licence and compliance](../natlas/licence.md) before building on the models.
