"""Dataset validation and scoring correctness against known answers."""
import json
import shutil
import unicodedata

from common import *

c = Checker("3 Dataset validation, normalisation and metrics")
W = WORK / "data"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)


def validate(name, rows=None, raw=None, args="", task="generation"):
    p = W / name
    if raw is not None:
        p.write_bytes(raw)
    else:
        write_jsonl(p, rows)
    r = sh(f"atlasforge dataset validate {name} --task {task} {args} --json", cwd=W, log=f"t03_{name}")
    try:
        j = json.loads(r.out)
        codes = [i["code"] for i in j["issues"]]
    except Exception:  # noqa: BLE001
        j, codes = None, []
    return r, j, codes


good = [{"id": f"g{i}", "input": f"Question number {i}?", "reference": f"answer {i}", "lang": "en"} for i in range(30)]
r, j, codes = validate("ok.jsonl", good)
c("3.01", "A valid dataset passes (exit 0, no issues, stats present)", r.rc == 0 and j and j["issues"] == [] and "stats" in j, f"rc={r.rc} n={j and j['n_examples']}", "t03_ok.jsonl")

bad = [
    {"id": "a", "input": "q1", "reference": "r1"},
    {"id": "a", "input": "q2", "reference": "r2"},          # duplicate id (line 2)
    {"id": "b", "input": "q3", "refrence": "x"},            # unknown key (line 3)
    {"id": "c", "input": "q", "messages": [{"role": "user", "content": "q"}], "reference": "r"},  # both (line 4)
    {"id": "d", "reference": "r"},                           # neither (line 5)
    {"id": "e", "input": "q", "reference": "r", "lang": "klingon"},  # bad lang (line 6)
]
p = W / "bad.jsonl"
write_jsonl(p, bad)
with open(p, "a", encoding="utf-8") as fh:
    fh.write("not json at all\n[1, 2]\n")                    # lines 7, 8
r, j, codes = validate("bad.jsonl", None, raw=p.read_bytes())
lines = sorted(i["line"] for i in j["issues"] if i["code"] == "line-error")
c("3.02", "validate reports EVERY malformed line at once, each with its line number (exit 1)",
  r.rc == 1 and lines == [2, 3, 4, 5, 6, 7, 8], f"error lines={lines}", "t03_bad.jsonl")
msgs = " ".join(i["message"] for i in j["issues"])
c("3.03", "Typos in keys are caught loudly ('refrence' is named), duplicates name the first line",
  "refrence" in msgs and "first seen on line 1" in msgs, msgs[:200])
c("3.04", "Line numbers are not printed twice in messages (regression fixed in HEAD)",
  not any(i["message"].startswith("line ") for i in j["issues"]), f"first message: {j['issues'][0]['message']}")

r, j, codes = validate("nonutf8.jsonl", raw=b'{"id":"a","input":"caf\xe9","reference":"x"}\n')
c("3.05", "A non-UTF-8 file is refused with a clear message, not a traceback",
  r.rc == 2 and "UTF-8" in (r.err + r.out) and "Traceback" not in r.both, (r.err or r.out).strip()[:160], "t03_nonutf8.jsonl")

r, j, codes = validate("empty.jsonl", raw=b"\n\n")
c("3.06", "An empty dataset is an error", r.rc == 1 and "no-examples" in codes, f"codes={codes}", "t03_empty.jsonl")

train = [{"id": f"t{i}", "input": f"Train question {i}", "reference": f"a{i}"} for i in range(25)]
test = [{"id": "x1", "input": "TRAIN QUESTION 3!", "reference": "a3"}, {"id": "x2", "input": "Completely new", "reference": "z"}]
write_jsonl(W / "train.jsonl", train)
r, j, codes = validate("test.jsonl", test, args="--against train.jsonl")
c("3.07", "Train/test leakage is detected even when it differs only in case and punctuation (error, exit 1)",
  r.rc == 1 and "train-test-leakage" in codes, f"codes={codes}", "t03_test.jsonl")
r2 = sh("atlasforge dataset validate test.jsonl --task generation --json", cwd=W)
c("3.08", "Without --against the same file raises no leakage error", "train-test-leakage" not in r2.out, "ok")

dup = [{"id": "d1", "input": "Same question", "reference": "same"}, {"id": "d2", "input": "same QUESTION", "reference": "Same"},
       {"id": "d3", "input": "Conflict", "reference": "one"}, {"id": "d4", "input": "Conflict", "reference": "two"}]
r, j, codes = validate("dups.jsonl", dup)
c("3.09", "Duplicate content and conflicting references are warnings, not errors (exit 0)",
  r.rc == 0 and "duplicate-content" in codes and "conflicting-references" in codes, f"codes={codes}", "t03_dups.jsonl")

damage = [{"id": "u1", "input": "café bad � text", "reference": "x"},
          {"id": "u2", "input": "école decomposed", "reference": "y"},
          {"id": "u3", "input": "ctrl\x07char", "reference": "z"}]
r, j, codes = validate("damage.jsonl", damage)
c("3.10", "Broken Unicode is flagged: replacement character, non-NFC text, control characters",
  {"replacement-character", "non-nfc", "control-characters"} <= set(codes), f"codes={codes}", "t03_damage.jsonl")

yo_stripped = [{"id": f"y{i}", "input": f"Bawo ni o se wa {i}", "reference": f"Mo wa dara {i}", "lang": "yo"} for i in range(25)]
r, j, codes = validate("yo_stripped.jsonl", yo_stripped)
c("3.11", "Stripped Yoruba diacritics (no tone marks or underdots in 25 'yo' texts) raise a warning",
  "diacritics-missing" in codes and r.rc == 0, f"codes={codes}", "t03_yo_stripped.jsonl")
yo_ok = [{"id": f"y{i}", "input": f"Bawo ni o ṣe wà {i}", "reference": f"Mo wà dáadáa {i}", "lang": "yo"} for i in range(25)]
r, j, codes = validate("yo_ok.jsonl", yo_ok)
c("3.12", "Properly marked Yoruba text does NOT raise the diacritics warning (no false alarm)", "diacritics-missing" not in codes, f"codes={codes}", "t03_yo_ok.jsonl")

cls = [{"id": f"c{i}", "input": f"review {i}", "reference": "positive" if i < 19 else "negative"} for i in range(20)]
r, j, codes = validate("cls_imb.jsonl", cls, task="classification")
c("3.13", "Class imbalance is flagged for classification datasets", "class-imbalance" in codes, f"codes={codes}", "t03_cls_imb.jsonl")
r, j, codes = validate("cls_one.jsonl", [{"id": f"c{i}", "input": f"r{i}", "reference": "positive"} for i in range(5)], task="classification")
c("3.14", "A single-class classification dataset is flagged", "single-class" in codes, f"codes={codes}", "t03_cls_one.jsonl")
r, j, codes = validate("cls_empty.jsonl", [{"id": "c1", "input": "r", "reference": " "}, {"id": "c2", "input": "r2", "reference": "x"}], task="classification")
c("3.15", "An empty reference in a classification dataset is an error (cannot be scored)", r.rc == 1 and "empty-reference" in codes, f"rc={r.rc} codes={codes}", "t03_cls_empty.jsonl")

r, j, codes = validate("asr_missing.jsonl", [{"id": "s1", "audio": "nope.wav", "reference": "x", "lang": "ha"}], task="asr")
c("3.16", "An ASR dataset pointing at a missing audio file is an error naming the file", r.rc == 1 and "audio file not found" in json.dumps(j), f"codes={codes}", "t03_asr_missing.jsonl")

# --------------------------------------------------------------------- scoring against known answers
prompts = {
    "Q-precomposed": "Ẹ káàárọ̀", "Q-tones": "Ẹ káàárọ̀", "Q-hooked": "ɓarna", "Q-igbo": "ụlọ", "Q-punct": "Hello, World!",
    "Q-wer": "a x c", "Q-empty": "anything",
}
refs = {"Q-precomposed": "Ẹ káàárọ̀", "Q-tones": "Ẹ káàárọ̀", "Q-hooked": "ɓarna", "Q-igbo": "ụlọ", "Q-punct": "hello world",
        "Q-wer": "a b c", "Q-empty": "needs an answer"}
answers = {
    "Q-precomposed": unicodedata.normalize("NFD", "Ẹ káàárọ̀"),   # same text, combining form
    "Q-tones": "Ẹ kaaarọ",                                        # underdots kept, tone marks dropped
    "Q-hooked": "barna",                                          # hooked letter replaced by plain b
    "Q-igbo": "ulo",                                              # Igbo underdots dropped
    "Q-punct": "HELLO world!!!",
    "Q-wer": "a x c",
    "Q-empty": "",
}
stub = StubServer(answer=lambda p: answers.get(p, "?")).start()
ds = [{"id": k, "input": k, "reference": refs[k], "lang": "yo" if k in ("Q-precomposed", "Q-tones") else None} for k in prompts]
for d in ds:
    if d["lang"] is None:
        del d["lang"]
write_jsonl(W / "known.jsonl", ds)
r = sh(f"atlasforge eval known.jsonl --out known_run --base-url {stub.url} --model m -m exact_match -m chrf -m wer -m cer --retries 0", cwd=W, log="t03_known_eval")
rep = json.loads((W / "known_run/report.json").read_text())
pe = rep["per_example"]
em = lambda k, v: pe[k][f"exact_match@{v}"]
c("3.17", "Precomposed and combining Unicode of the same Yoruba text score identically (both views = 1)",
  em("Q-precomposed", "tone_aware") == 1 and em("Q-precomposed", "tone_insensitive") == 1, f"aware={em('Q-precomposed','tone_aware')} insens={em('Q-precomposed','tone_insensitive')}", "t03_known_eval")
c("3.18", "Tone marks matter in the tone-aware view (0) and are ignored in the tone-insensitive view (1); the underdot is kept",
  em("Q-tones", "tone_aware") == 0 and em("Q-tones", "tone_insensitive") == 1, f"aware={em('Q-tones','tone_aware')} insens={em('Q-tones','tone_insensitive')}")
c("3.19", "Hausa hooked letters are never stripped (barna != ɓarna in BOTH views)",
  em("Q-hooked", "tone_aware") == 0 and em("Q-hooked", "tone_insensitive") == 0, "both 0")
c("3.20", "Igbo underdot letters are never stripped (ulo != ụlọ in BOTH views)",
  em("Q-igbo", "tone_aware") == 0 and em("Q-igbo", "tone_insensitive") == 0, "both 0")
c("3.21", "Case and punctuation are normalised away ('HELLO world!!!' == 'hello world')", em("Q-punct", "tone_aware") == 1, "match")
w = pe["Q-wer"]["wer@tone_aware"]; cc = pe["Q-wer"]["cer@tone_aware"]
c("3.22", "WER is exactly 1/3 for one wrong word of three; CER = 1 wrong char of 5 (jiwer-consistent)",
  abs(w - 1 / 3) < 1e-9 and abs(cc - 1 / 5) < 1e-9, f"wer={w:.4f} cer={cc:.4f}")
c("3.23", "An empty answer scores 0 on chrF and 1.0 on WER (all words deleted)",
  pe["Q-empty"]["chrf@tone_aware"] == 0 and pe["Q-empty"]["wer@tone_aware"] == 1.0, f"chrf={pe['Q-empty']['chrf@tone_aware']} wer={pe['Q-empty']['wer@tone_aware']}")
c("3.24", "An answer identical after normalisation scores chrF 100", pe["Q-punct"]["chrf@tone_aware"] == 100.0, f"{pe['Q-punct']['chrf@tone_aware']}")

# classification metrics with known answers
cl_items = [("c1", "positive", "positive"), ("c2", "positive", "The sentiment is positive."), ("c3", "negative", "negative"),
            ("c4", "negative", "positive"), ("c5", "positive", "not positive"), ("c6", "negative", "dunno")]
cans = {i: a for i, _, a in cl_items}
stub.answer = lambda p: cans.get(p, "?")
write_jsonl(W / "cls.jsonl", [{"id": i, "input": i, "reference": ref} for i, ref, _ in cl_items])
r = sh(f"atlasforge eval cls.jsonl --task classification --out cls_run --base-url {stub.url} --model m -m accuracy -m macro_f1 --retries 0", cwd=W, log="t03_cls_eval")
rc = json.loads((W / "cls_run/report.json").read_text())
acc = next(m for m in rc["metrics"] if m["key"] == "accuracy@tone_aware")["mean"]
f1 = next(m for m in rc["metrics"] if m["key"] == "macro_f1@tone_aware")["corpus"]
# loose: c1 ok, c2 ok, c3 ok, c4 wrong, c5 'not positive' -> positive (ok, known negation limitation), c6 none -> wrong => 4/6
c("3.25", "Classification accuracy matches a hand calculation (loose label match; 4 of 6)", abs(acc - 4 / 6) < 1e-9, f"accuracy={acc:.4f}", "t03_cls_eval")
# macro-F1 hand calc: preds: c1 pos, c2 pos, c3 neg, c4 pos, c5 pos, c6 None ; refs: pos,pos,neg,neg,pos,neg
# pos: tp=3 (c1,c2,c5) fp=1 (c4) fn=0 -> F1=6/7 ; neg: tp=1 (c3) fp=0 fn=2 (c4,c6) -> F1=2/4=0.5 ; mean = (6/7+0.5)/2
c("3.26", "Macro-F1 matches a hand calculation ((6/7 + 1/2) / 2); an unlabelled answer counts as a miss",
  abs(f1 - (6 / 7 + 0.5) / 2) < 1e-9, f"macro_f1={f1:.4f} expected={(6/7+0.5)/2:.4f}")
r = sh(f"atlasforge eval cls.jsonl --task classification --out cls_strict --base-url {stub.url} --model m -m accuracy -m accuracy_strict --retries 0", cwd=W, log="t03_cls_strict")
if r.rc == 0:
    rs = json.loads((W / "cls_strict/report.json").read_text())
    s = next(m for m in rs["metrics"] if m["key"] == "accuracy_strict@tone_aware")["mean"]
    c("3.27", "The strict metric rejects negation ('not positive') that the loose metric accepts (strict < loose)", s < acc, f"loose={acc:.3f} strict={s:.3f}", "t03_cls_strict")
else:
    c("3.27", "accuracy_strict exists (documented in HEAD docs)", False, (r.err or r.out).strip()[:160], "t03_cls_strict")
stub.stop()
