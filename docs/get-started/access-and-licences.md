# Access and licences

N-ATLaS is distributed as **gated open weights** on Hugging Face. There is no public hosted API. This page covers how to get access and what you are agreeing to.

## The models

| Model | Repository | What it is |
|---|---|---|
| LLM | [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS) | Llama-3 8B fine-tune: English, Hausa, Igbo, Yoruba |
| Hausa speech | [`NCAIR1/Hausa-ASR`](https://huggingface.co/NCAIR1/Hausa-ASR) | Whisper-Small fine-tune |
| Yoruba speech | [`NCAIR1/Yoruba-ASR`](https://huggingface.co/NCAIR1/Yoruba-ASR) | Whisper-Small fine-tune |
| Igbo speech | [`NCAIR1/Igbo-ASR`](https://huggingface.co/NCAIR1/Igbo-ASR) | Whisper-Small fine-tune |
| Nigerian English speech | [`NCAIR1/NigerianAccentedEnglish`](https://huggingface.co/NCAIR1/NigerianAccentedEnglish) | Whisper-Small fine-tune |

## Get access in four steps

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co).
2. **Accept the licence on each model page** you need. Each page asks you to agree to its terms before showing the files. Approval can take a little while, so do this first.
3. **Create an access token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A *read* token is enough.
4. **Give the token to your shell,** typing it yourself so it never appears in a file or a screenshot:

=== "macOS / Linux"

    ```bash
    export HF_TOKEN=hf_your_token_here
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:HF_TOKEN = "hf_your_token_here"
    ```

Then confirm it registered:

```bash
atlasforge doctor
```

The `hf-token` row should read `OK` and show the token **masked** (`hf_****abcd`). AtlasForge never prints a token in full.

!!! warning "Keep the token out of version control"
    Never commit a token, paste it in an issue, or leave it visible in a screenshot or demo video. If you do, revoke it and create a new one.

## The licence, in summary

The models use a custom *Open-Source Research and Innovation License*. As summarised from the model cards:

| Term | What it means for you |
|---|---|
| **1,000 active end-user cap** | Use is limited to organisations with no more than 1,000 active end-users |
| **Commercial use** | Needs a separate licensing agreement |
| **Attribution** | You must credit Awarri Technologies and the Federal Ministry of Communications |
| **"Powered by Awarri"** | Derivative works must carry that suffix |

!!! danger "This is a summary, not the licence"
    These terms were read from the model cards. **The licence text governs.** Read it on the model page before building on the models, and check it again before you release anything.

## What AtlasForge will never do

- **Redistribute the weights.** You download them yourself, under your own accepted licence.
- **Publish or help you publish quantised or merged copies of the base model.** Fine-tuning produces a small *adapter*, not a copy of N-ATLaS.
- **Send your data anywhere** other than the endpoint you configure. There is no telemetry.
- **Use any other company's model** for scoring or judging. There is no LLM-as-judge.

## NAIC participants

For the National AI Innovation Challenge 2026, API credentials are issued only to **shortlisted** teams. Until then the way to use the models is the Hugging Face weights, through your own hardware or an endpoint you run. See [Choose your setup](choose-your-setup.md).
