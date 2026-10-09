"""Fine-tune plumbing and the local LLM backend, end to end, with a TINY public stand-in model.

NOT an N-ATLaS result: HuggingFaceTB/SmolLM2-135M-Instruct is used purely so the real code paths (tokenizer chat
template, generation, LoRA training, adapter loading, model card, evaluation of an adapter) can run on a Mac.
"""
import json
import shutil

from common import *

c = Checker("9 Fine-tune and local LLM plumbing (tiny stand-in model)")
W = WORK / f"ft_{LABEL}"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"

rows = [{"id": f"k{i:02d}", "input": f"Give the code for item {i}.", "reference": f"CODE-{i:03d}", "lang": "en", "meta": {"set": "a" if i < 20 else "b"}} for i in range(40)]
write_jsonl(W / "train.jsonl", rows)

r = sh("atlasforge --version", log="t09_version")
py = sh([PY, "-c", "import sys; print(sys.version.split()[0])"]).out.strip()
c("9.01", f"The package installs and starts on Python {py} (outside the documented 3.10 to 3.13: forward-compatibility data point)", r.rc == 0, r.out.strip() + f" on Python {py}", "t09_version")

(W / "ft.json").write_text(json.dumps({"base_model": TINY, "train_file": "train.jsonl", "output_dir": "adapter", "quantize": "none",
                                        "lora_r": 8, "lora_alpha": 16, "learning_rate": 0.0005, "num_epochs": 4, "batch_size": 4, "grad_accum": 1,
                                        "max_seq_len": 128, "min_examples": 20, "logging_steps": 5}))
r = sh("atlasforge finetune ft.json --dry-run", cwd=W, log="t09_dry", timeout=300)
c("9.02", "finetune --dry-run plans a LoRA run on a non-default base model", r.rc == 0 and "lora" in r.out.lower(), " ".join(r.out.split())[:220], "t09_dry")

r = sh("atlasforge finetune ft.json", cwd=W, log="t09_train", timeout=400)
crashed = r.rc in (-11, -6, 139, 134, 124)
c.na("9.03", "A real LoRA training run (needs CUDA; the docs say it will not work on a Mac)",
     f"Attempted on this Mac with quantize=none: the process died (exit {r.rc}) while loading the model with device_map='auto' in float16 on Apple's GPU. "
     "The SAME crash reproduces with plain transformers and no AtlasForge code (see logs), so it is the environment, not a verdict on the recipe. Training remains unverified.")
record("9 Fine-tune and local LLM plumbing (tiny stand-in model)", "9.03b", "Observation: on a non-CUDA machine, `finetune` with quantize=none gives no friendly message (the 4-bit path does); it crashes inside torch",
       PASS if not crashed else FAIL, f"exit {r.rc}; stderr tail: {(r.err or '').strip()[-120:]}", "t09_train")

# a real PEFT adapter made with plain peft (NOT by atlasforge) so the adapter-loading path can be exercised
mk = """
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("%s"); m = AutoModelForCausalLM.from_pretrained("%s", torch_dtype=torch.float32)
pm = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "down_proj", "up_proj"], task_type="CAUSAL_LM"))
for n, p in pm.named_parameters():
    if "lora_B" in n: torch.nn.init.normal_(p, std=0.08)
pm.save_pretrained("adapter"); tok.save_pretrained("adapter"); print("ADAPTER_OK")
""" % (TINY, TINY)
(W / "mk_adapter.py").write_text(mk)
ra = sh([PY, "mk_adapter.py"], cwd=W, log="t09_mkadapter", timeout=600)
ad = W / "adapter"
c("9.04", "Precondition: a real PEFT LoRA adapter (built with plain peft, not by AtlasForge) exists to evaluate", ra.rc == 0 and (ad / "adapter_model.safetensors").exists(), f"files={sorted(p.name for p in ad.iterdir()) if ad.exists() else ra.err[-150:]}", "t09_mkadapter")
if ra.rc != 0:
    raise SystemExit(0)

GEN = f"--backend local --device cpu --model {TINY} --temperature 0 --max-new-tokens 12"
r = sh(f"atlasforge eval train.jsonl --out base {GEN} -m exact_match -m chrf", cwd=W, log="t09_eval_base", timeout=1800)
r2 = sh(f"atlasforge eval train.jsonl --out tuned {GEN} --adapter adapter -m exact_match -m chrf", cwd=W, log="t09_eval_tuned", timeout=1800)
ok = r.rc == 0 and r2.rc == 0
mb = json.loads((W / "base/run.json").read_text()) if ok else {}
mt = json.loads((W / "tuned/run.json").read_text()) if ok else {}
c("9.06", "eval --backend local runs the real tokenizer chat template and generation, and --adapter loads a LoRA adapter",
  ok, f"base rc={r.rc}, tuned rc={r2.rc}; {(r2.err or '').strip()[-160:]}", "t09_eval_tuned")
c("9.07", "An adapter is part of the model's identity in the manifest (a tuned run can never be mistaken for the base run)",
  ok and mt["model"] != mb["model"] and "adapter" in json.dumps(mt).lower(), f"base model={mb.get('model')!r}  tuned model={mt.get('model')!r}")
if ok:
    pb = {x["id"]: x["prediction"] for x in read_jsonl(W / "base/results.jsonl")}
    pt = {x["id"]: x["prediction"] for x in read_jsonl(W / "tuned/results.jsonl")}
    differ = sum(pb[k] != pt[k] for k in pb)
    c("9.08", "The adapter is actually APPLIED: with a non-zero LoRA the tuned model's answers differ from the base model's", differ > 0 and len(pb) == 40, f"{differ} of 40 answers differ; e.g. base={pb['k00']!r} tuned={pt['k00']!r}")
    r3 = sh("atlasforge compare train.jsonl --base base --candidate tuned --out cmp --slice set -m exact_match -m chrf", cwd=W, log="t09_compare", timeout=300)
    cj = json.loads((W / "cmp/comparison.json").read_text()) if r3.rc == 0 else {}
    c("9.09", "compare works between a base run and an adapter run on real local-backend output (with slices)", r3.rc == 0 and len(cj.get("metrics", [])) >= 2 and any(s["field"] == "set" for s in cj.get("slices", [])),
      f"verdicts={[(m['key'], m['verdict']) for m in cj.get('metrics', [])][:2]}", "t09_compare")

# local LLM mechanics
r1 = sh(f"atlasforge run 'Name one fruit.' {GEN} --json", cwd=W, log="t09_run1", timeout=600)
r2 = sh(f"atlasforge run 'Name one fruit.' {GEN} --json", cwd=W, log="t09_run2", timeout=600)
try:
    j1, j2 = json.loads(r1.out), json.loads(r2.out)
    ok = j1["text"] == j2["text"] and j1["usage"]["completion_tokens"] <= 12 and j1["usage"]["prompt_tokens"] > 0 and j1["finish_reason"] in ("stop", "length")
    ev = f"{j1['text']!r} tokens={j1['usage']} finish={j1['finish_reason']}"
except Exception as exc:  # noqa: BLE001
    ok, ev = False, f"{exc}: {r1.err[-200:]}"
c("9.10", "local backend: greedy decoding is deterministic, max-new-tokens is honoured, token usage and finish_reason are reported", ok, ev, "t09_run1")
r = sh(f"atlasforge run hi --backend local --device cpu --model sshleifer/tiny-gpt2 --temperature 0", cwd=W, log="t09_nochat", timeout=600)
c("9.11", "A model whose tokenizer has no chat template is refused with a clear error (never a guessed prompt format)", r.rc == 2 and "chat template" in r.err and "Traceback" not in r.both, r.err.strip()[:200], "t09_nochat")
r = sh(f"atlasforge eval train.jsonl --out x --base-url http://127.0.0.1:9/v1 --adapter adapter", cwd=W, log="t09_adapter_http")
c("9.12", "--adapter with the openai backend is refused with an explanation", r.rc == 2 and "local" in r.err and "Traceback" not in r.both, r.err.strip()[:200], "t09_adapter_http")
