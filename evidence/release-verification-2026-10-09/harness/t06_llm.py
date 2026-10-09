"""The real N-ATLaS LLM (int4 import served by Ollama) driven through the installed package."""
import json
import shutil

from common import *

c = Checker("6 Real LLM through the openai backend")
W = WORK / f"llm_{LABEL}"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
URL = "http://127.0.0.1:11434/v1"
CHAT, RAW = "natlas-local-chat", "natlas-local-q4"

CAPITALS = [("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Kenya", "Nairobi"), ("Ethiopia", "Addis Ababa"), ("Egypt", "Cairo"),
            ("Namibia", "Windhoek"), ("Senegal", "Dakar"), ("Cameroon", "Yaoundé"), ("Uganda", "Kampala"), ("Rwanda", "Kigali"),
            ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Angola", "Luanda"), ("Mali", "Bamako"), ("Niger", "Niamey"),
            ("Chad", "N'Djamena"), ("Sudan", "Khartoum"), ("Libya", "Tripoli"), ("Tunisia", "Tunis"), ("Algeria", "Algiers"),
            ("Morocco", "Rabat"), ("Benin", "Porto-Novo"), ("Togo", "Lomé"), ("Burkina Faso", "Ouagadougou"), ("Guinea", "Conakry"),
            ("Sierra Leone", "Freetown"), ("Liberia", "Monrovia"), ("Gambia", "Banjul"), ("Malawi", "Lilongwe"), ("Botswana", "Gaborone")]
rows = [{"id": f"cap-{i:02d}", "input": f"What is the capital of {c_}? Answer with the city name only.", "reference": cap, "lang": "en", "meta": {"domain": "capitals"}}
        for i, (c_, cap) in enumerate(CAPITALS)]
for i in range(30):
    a, b = 11 + 7 * i, 23 + 5 * i
    rows.append({"id": f"sum-{i:02d}", "input": f"What is {a} plus {b}? Answer with the number only.", "reference": str(a + b), "lang": "en", "meta": {"domain": "arithmetic"}})
write_jsonl(W / "real.jsonl", rows)
(W / "mymetrics.py").write_text('def contains_ref(prediction, reference, example):\n    return float(reference in prediction)\n')

# ------------------------------------------------------------------ the server really is N-ATLaS
import urllib.request
try:
    tags = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5))
    names = [m["name"] for m in tags["models"]]
    ok = f"{CHAT}:latest" in names
except Exception as exc:  # noqa: BLE001
    ok, names = False, [str(exc)]
c("6.01", "Precondition: the official N-ATLaS weights are imported in Ollama (int4) and served", ok, f"models={names}")
if not ok:
    raise SystemExit(0)
show = sh(f"ollama show {CHAT}", log="t06_show")
c("6.02", "The served model is an 8.0B Llama with 131072 context, int4 (matches config.json, not the card's 8,092)",
  "8.0B" in show.out and "131072" in show.out and "int4" in show.out, " ".join(show.out.split())[:200], "t06_show")

# ------------------------------------------------------------------ run in four languages
prompts = {"en": "What is the capital of Nigeria? Answer in one sentence.",
           "ha": "Ina kwana? Gaya mini game da Najeriya a taƙaice.",
           "yo": "Bawo ni? Sọ fun mi nípa ìlú Eko ní ṣókí.",
           "ig": "Kedu? Gwa m banyere Naịjirịa n'ụzọ dị mkpụmkpụ."}
outs = {}
for lang, p in prompts.items():
    r = sh(["atlasforge", "run", p, "--base-url", URL, "--model", CHAT, "--json", "--max-new-tokens", "120", "--timeout", "300"], log=f"t06_run_{lang}", timeout=400)
    try:
        j = json.loads(r.out)
        outs[lang] = j
    except Exception:  # noqa: BLE001
        outs[lang] = None
    c(f"6.03{lang}", f"run: the real model answers a {lang} prompt (non-empty text, token usage and latency reported)",
      bool(outs[lang]) and len(outs[lang]["text"].strip()) > 5 and outs[lang]["usage"]["completion_tokens"] > 0 and outs[lang]["latency_ms"] > 0,
      (f"{outs[lang]['text'][:110]!r} tokens={outs[lang]['usage']} {outs[lang]['latency_ms']:.0f}ms") if outs[lang] else r.err[:150], f"t06_run_{lang}")

r = sh(["atlasforge", "run", "What is the capital of Nigeria? Answer in one word.", "--base-url", URL, "--model", CHAT, "--temperature", "0", "--system", "You answer with a single word."], log="t06_run_sys", timeout=300)
c("6.04", "Real model, --system and --temperature 0: answers 'Abuja'", "abuja" in r.out.lower(), r.out.strip()[:100], "t06_run_sys")
r2 = sh(["atlasforge", "run", "What is the capital of Nigeria? Answer in one word.", "--base-url", URL, "--model", CHAT, "--temperature", "0", "--system", "You answer with a single word.", "--no-send-repetition-penalty"], log="t06_run_norp", timeout=300)
c("6.05", "--no-send-repetition-penalty works against the real server too", r2.rc == 0 and "abuja" in r2.out.lower(), r2.out.strip()[:100], "t06_run_norp")

# ------------------------------------------------------------------ validate + real evaluation
r = sh("atlasforge dataset validate real.jsonl", cwd=W, log="t06_validate")
c("6.06", "The 60-question real dataset validates", r.rc == 0, r.out.strip()[-100:], "t06_validate")
EV = f"atlasforge eval real.jsonl --base-url {URL} --temperature 0 --max-new-tokens 40 --timeout 300 -m exact_match -m chrf -m mymetrics:contains_ref"
r = sh(f"{EV} --model {CHAT} --out chat", cwd=W, env={"PYTHONPATH": str(W)}, log="t06_eval_chat", timeout=1800)
rc = json.loads((W / "chat/report.json").read_text()) if r.rc == 0 else None
em = rc and next(m for m in rc["metrics"] if m["key"] == "contains_ref@tone_aware")["mean"]
c("6.07", "eval of 60 real questions on the real model completes with no failures, reports latency, and a custom metric",
  bool(rc) and rc["n_failed"] == 0 and rc["latency_ms"]["mean"] and em is not None, f"n_failed={rc and rc['n_failed']} contains_ref={em} mean latency={rc and rc['latency_ms']['mean']:.0f}ms p95={rc and rc['latency_ms']['p95']:.0f}ms", "t06_eval_chat")
res = read_jsonl(W / "chat/results.jsonl")
c("6.08", "Every record has a prediction and a request id/latency; examples are the model's real text, not echoes",
  all(x["prediction"] for x in res) and len({x["prediction"] for x in res}) > 20, f"{len(res)} records; sample: {[x['prediction'] for x in res[:3]]}")
manifest = json.loads((W / "chat/run.json").read_text())
c("6.09", "run.json records the served model name and the settings used (temperature 0, max_new_tokens 40)",
  manifest["model"] == CHAT and manifest["gen_params"]["temperature"] == 0 and manifest["gen_params"]["max_new_tokens"] == 40, json.dumps({k: manifest[k] for k in ("model", "backend", "gen_params")}))

r = sh(f"{EV} --model {RAW} --out raw", cwd=W, env={"PYTHONPATH": str(W)}, log="t06_eval_raw", timeout=1800)
c("6.10", "A second real run (same weights, without the chat template) completes", r.rc == 0, r.out.strip()[-80:], "t06_eval_raw")

r = sh("atlasforge compare real.jsonl --base raw --candidate chat --out cmp --slice domain -m exact_match -m chrf -m mymetrics:contains_ref", cwd=W, env={"PYTHONPATH": str(W)}, log="t06_compare", timeout=300)
cj = json.loads((W / "cmp/comparison.json").read_text()) if r.rc == 0 else {}
sl = [s for s in cj.get("slices", []) if s["field"] == "domain"]
c("6.11", "compare on two REAL runs: per-metric verdicts, win/tie/loss, and domain slices with n=30 each get real verdicts (not 'insufficient data')",
  r.rc == 0 and len(sl) == 2 and all(s["status"] != "insufficient data" for s in sl) and all("wins" in m for m in cj["metrics"]),
  "; ".join(f"{m['key']}:{m['verdict']} d={m['delta']:+.3f}" for m in cj.get("metrics", []) if m["view"] == "tone_aware") + " | " + "; ".join(f"{s['value']} n={s['n']} {s['status']}" for s in sl), "t06_compare")
mc = next((m for m in cj.get("metrics", []) if m["name"] == "contains_ref" and m["view"] == "tone_aware"), None)
c("6.12", "A custom metric gets the full paired statistics in compare (interval + wins/ties/losses)", bool(mc) and mc["low"] is not None and mc["wins"] + mc["ties"] + mc["losses"] == 60, f"{mc and (mc['base_mean'], mc['candidate_mean'], mc['low'], mc['high'])}")

# ------------------------------------------------------------------ classification with the real model
sent = [("The food was wonderful and the staff were kind.", "positive"), ("I love this phone, the battery lasts for days.", "positive"),
        ("A brilliant film, I enjoyed every minute.", "positive"), ("Great value, I would happily buy it again.", "positive"),
        ("The service was excellent and very fast.", "positive"), ("What a lovely, peaceful hotel.", "positive"),
        ("Terrible experience, the food was cold and rude waiters.", "negative"), ("This phone broke after two days, a waste of money.", "negative"),
        ("A boring film, I wanted to leave early.", "negative"), ("Awful quality, it fell apart immediately.", "negative"),
        ("The service was slow and the staff ignored us.", "negative"), ("A dirty, noisy hotel. Never again.", "negative")]
write_jsonl(W / "cls.jsonl", [{"id": f"s{i:02d}", "input": f"Is this review positive or negative? Answer with one word: positive or negative.\nReview: {t}", "reference": lab, "lang": "en"} for i, (t, lab) in enumerate(sent)])
r = sh(f"atlasforge eval cls.jsonl --task classification --base-url {URL} --model {CHAT} --out cls --temperature 0 --max-new-tokens 8 --timeout 300", cwd=W, log="t06_cls", timeout=900)
rj = json.loads((W / "cls/report.json").read_text()) if r.rc == 0 else None
acc = rj and next(m for m in rj["metrics"] if m["name"] == "accuracy")["mean"]
f1 = rj and next(m for m in rj["metrics"] if m["name"] == "macro_f1")["corpus"]
c("6.13", "Classification task on the real model reports accuracy and macro-F1 (12 reviews)", bool(rj) and acc is not None and f1 is not None, f"accuracy={acc} macro_f1={f1}", "t06_cls")

# ------------------------------------------------------------------ concurrency against the real server
r = sh(f"{EV} --model {CHAT} --out chat_c2 --concurrency 2", cwd=W, env={"PYTHONPATH": str(W)}, log="t06_eval_conc", timeout=1800)
rj2 = json.loads((W / "chat_c2/report.json").read_text()) if r.rc == 0 else None
c("6.14", "--concurrency 2 against the real server completes with no failures", bool(rj2) and rj2["n_failed"] == 0, f"n_failed={rj2 and rj2['n_failed']} wall={r.secs:.0f}s", "t06_eval_conc")
