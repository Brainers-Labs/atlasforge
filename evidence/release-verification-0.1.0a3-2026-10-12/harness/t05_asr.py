"""Speech: the four official ASR models on real audio, through the installed package."""
import json
import re
import shutil
import unicodedata

import jiwer

from common import *

c = Checker("5 Speech models on real audio")
W = WORK / f"asr_{LABEL}"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
AUD = DATA / "audio"
shutil.copytree(AUD, W / "audio")
for lang in ("ha", "yo", "ig", "en"):
    shutil.copy(DATA / f"asr_{lang}.jsonl", W / f"asr_{lang}.jsonl")


def norm(t, strip=False):
    t = unicodedata.normalize("NFD", t.lower())
    if strip:
        t = re.sub("[\u0300\u0301\u0302\u0304\u030c]", "", t)
    t = unicodedata.normalize("NFC", t)
    t = t.translate({ord(ch): "'" for ch in "\u2018\u2019\u201b\u02bc\u02bb\u02b9`\u00b4"})
    t = "".join(" " if unicodedata.category(ch)[0] in "PS" and ch != "'" else ch for ch in t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)
    return " ".join(t.split())


def wer(hyp, ref, strip=False):
    h, r = norm(hyp, strip), norm(ref, strip)
    return jiwer.wer(r, h) if h else 1.0


refs = {l: [json.loads(x) for x in (W / f"asr_{l}.jsonl").read_text(encoding="utf-8").splitlines()] for l in ("ha", "yo", "ig", "en")}

# ----------------------------------------------------------------- can the ASR extra be used at all?
r = sh("atlasforge transcribe audio/ha_00.wav --lang ha --backend local", cwd=W, log="t05_first", timeout=900)
first_text = r.out.strip()
works = r.rc == 0 and len(first_text) > 10
c("5.01", "[asr] extra: `transcribe --backend local` loads the official Hausa model and returns text on a fresh install",
  works, (first_text[:150] if works else (r.err.strip()[-260:])), "t05_first")
if not works:
    c.na("5.02-5.20", "All further speech tests", "blocked by 5.01 on this build; see the log")
    raise SystemExit(0)

ref0 = refs["ha"][0]["reference"]
w0 = wer(first_text, ref0)
c("5.02", "Hausa transcription of a real clip is close to the reference (WER < 0.35, tone-aware view)", w0 < 0.35, f"WER={w0:.3f}\nref: {ref0}\nhyp: {first_text}")

# ----------------------------------------------------------------- formats
fmt = {}
for ext in ("ogg", "mp3", "m4a", "flac"):
    rr = sh(f"atlasforge transcribe audio/ha_00.{ext} --lang ha --backend local", cwd=W, log=f"t05_fmt_{ext}", timeout=600)
    fmt[ext] = (rr.rc, rr.out.strip())
agree = {e: round(wer(t, first_text), 3) for e, (rc, t) in fmt.items() if rc == 0}
c("5.03", "Any format ffmpeg reads works, including WhatsApp-style opus/ogg, mp3, m4a and flac (result within WER 0.35 of the wav result)",
  len(agree) == 4 and all(v < 0.35 for v in agree.values()), f"WER vs wav: {agree}")

# ----------------------------------------------------------------- real model identity per language
rr = sh("atlasforge transcribe audio/ha_00.wav --lang yo --backend local", cwd=W, log="t05_wronglang", timeout=600)
wy = wer(rr.out.strip(), ref0) if rr.rc == 0 else 9
c("5.04", "--lang selects a different model: Hausa audio through the Yoruba model is clearly worse than through the Hausa model",
  rr.rc == 0 and wy > w0 + 0.2, f"WER via ha model={w0:.3f}; via yo model={wy:.3f}")

# ----------------------------------------------------------------- multiple files, json, out, error isolation
r = sh("atlasforge transcribe audio/ha_01.wav audio/does_not_exist.wav audio/ha_02.wav --lang ha --backend local --json --out out.jsonl", cwd=W, log="t05_multi", timeout=900)
recs = read_jsonl(W / "out.jsonl") if (W / "out.jsonl").exists() else []
c("5.05", "Several files in one command: the missing one is reported on stderr (exit 1) but the others are still transcribed and written to --out",
  r.rc == 1 and [x["file"] for x in recs] == ["audio/ha_01.wav", "audio/ha_02.wav"] and "does_not_exist" in r.err,
  f"rc={r.rc} written={[x['file'] for x in recs]}; stderr={r.err.strip()[-120:]}", "t05_multi")
c("5.06", "--json records carry file, lang, text, latency_ms and chunks", bool(recs) and {"file", "lang", "text", "latency_ms", "chunks"} <= set(recs[0]), f"keys={sorted(recs[0]) if recs else None}")

(W / "fake.wav").write_text("this is not audio")
r = sh("atlasforge transcribe fake.wav --lang ha --backend local", cwd=W, log="t05_fake", timeout=300)
c("5.07", "A file that is not audio gives a clean error (exit 1, no traceback)", r.rc == 1 and "Traceback" not in r.both and "decode" in r.err.lower(), r.err.strip()[:160], "t05_fake")
r = sh("atlasforge transcribe audio/ha_00.wav --lang xx --backend local", cwd=W, log="t05_badlang")
c("5.08", "An unsupported language is a clear error naming the valid ones", r.rc == 2 and "ha" in r.err and "Traceback" not in r.both, r.err.strip()[:160], "t05_badlang")

# ----------------------------------------------------------------- long audio
r = sh("atlasforge transcribe audio/ha_long.wav --lang ha --backend local --json", cwd=W, log="t05_long", timeout=1200)
long_ref = (W / "audio/ha_long.txt").read_text(encoding="utf-8")
j = json.loads(r.out.strip().splitlines()[-1]) if r.rc == 0 else {}
chunks = j.get("chunks", [])
text = j.get("text", "")
trigrams = [tuple(norm(text).split()[i:i + 4]) for i in range(max(0, len(norm(text).split()) - 3))]
dups = len(trigrams) - len(set(trigrams))
wl = wer(text, long_ref) if text else 9
c("5.09", "A 79 s recording (limit is 30 s) is split into overlapping windows automatically and merged into one transcript",
  r.rc == 0 and len(chunks) >= 3 and wl < 0.4, f"windows={len(chunks)} boundaries={[(x['start_s'], x['end_s']) for x in chunks]} WER vs concatenated reference={wl:.3f}", "t05_long")
c("5.10", "Overlap merging does not leave duplicated passages (no repeated 4-word sequence at the seams)", dups == 0, f"repeated 4-grams={dups}")

r = sh("atlasforge transcribe audio/ha_long.wav --lang ha --backend local --json --silence-aware", cwd=W, log="t05_long_silence", timeout=1200)
if r.rc == 0:
    j2 = json.loads(r.out.strip().splitlines()[-1])
    ends = [x["end_s"] for x in j2["chunks"][:-1]]
    # My file has 0.3 s silent gaps after clips of known length; boundaries should land inside a gap
    import soundfile as sf, numpy as np
    d, sr = sf.read(W / "audio/ha_long.wav")
    quiet = [bool(np.abs(d[int((e - 0.4) * sr): int((e + 0.4) * sr)]).min() < 1e-4) for e in ends]
    c("5.11", "--silence-aware moves window cuts to a pause: each interior boundary falls inside/near a silent gap",
      len(ends) >= 1 and all(quiet) and wer(j2["text"], long_ref) < 0.4, f"boundaries={ends} near_silence={quiet} WER={wer(j2['text'], long_ref):.3f}", "t05_long_silence")
else:
    c("5.11", "--silence-aware works", False, r.err.strip()[:200], "t05_long_silence")

# Igbo clips over 30 s inside a normal dataset are handled by eval (below)

# ----------------------------------------------------------------- formal evaluation per language through the real eval pipeline
summary = {}
for lang in ("ha", "yo", "ig", "en"):
    r = sh(f"atlasforge eval asr_{lang}.jsonl --task asr --out run_{lang} --backend local", cwd=W, log=f"t05_eval_{lang}", timeout=2400)
    ok = r.rc == 0 and (W / f"run_{lang}/report.json").exists()
    if not ok:
        c(f"5.12{lang}", f"eval --task asr on 10 real {lang} clips completes", False, r.err.strip()[-200:], f"t05_eval_{lang}")
        continue
    rj = json.loads((W / f"run_{lang}/report.json").read_text())
    results = {x["id"]: x for x in read_jsonl(W / f"run_{lang}/results.jsonl")}
    refl = [x["reference"] for x in refs[lang]]
    hyp = [results[x["id"]]["prediction"] or "" for x in refs[lang]]
    mine = jiwer.wer([norm(x) for x in refl], [norm(x) for x in hyp])
    mine_i = jiwer.wer([norm(x, True) for x in refl], [norm(x, True) for x in hyp])
    tool = next(m for m in rj["metrics"] if m["key"] == "wer@tone_aware")["corpus"]
    tool_i = next(m for m in rj["metrics"] if m["key"] == "wer@tone_insensitive")["corpus"]
    cer_t = next(m for m in rj["metrics"] if m["key"] == "cer@tone_aware")["corpus"]
    lat = rj["latency_ms"]["mean"]
    summary[lang] = (tool, tool_i, cer_t, lat, rj["n_failed"])
    c(f"5.12{lang}", f"eval --task asr on 10 real FLEURS '{lang}' clips: pooled WER equals an independent jiwer computation (both tone views), nothing failed",
      abs(mine - tool) < 1e-9 and abs(mine_i - tool_i) < 1e-9 and rj["n_failed"] == 0,
      f"WER tone-aware={tool:.3f} tone-insens={tool_i:.3f} CER={cer_t:.3f} mean latency={lat:.0f} ms n_failed={rj['n_failed']}", f"t05_eval_{lang}")

json.dump(summary, open(ROOT / f"asr_summary_{LABEL}.json", "w"), indent=1)

if "ha" in summary:
    md = (W / "run_ha/report.md").read_text(encoding="utf-8")
    c("5.13", "The ASR report includes error analysis (substitutions, deletions, insertions)",
      "asr error analysis" in md.lower() and all(w in md.lower() for w in ("substituted", "dropped", "inserted")), [l for l in md.splitlines() if "utterance(s) aligned" in l][:1].__str__()[:200])
    manifest = json.loads((W / "run_ha/run.json").read_text())
    c("5.20", "The run manifest/report name the speech model that actually transcribed (NCAIR1/Hausa-ASR), not the LLM default",
      "Hausa-ASR" in json.dumps(manifest) , f"manifest model={manifest.get('model')!r}; report header: {[l for l in md.splitlines() if l.startswith('- **Model')][:1]}")
    hh = (W / "run_ha/report.html").read_text(encoding="utf-8")
    c("5.14", "ASR report.html is self-contained", "<script" not in hh.lower() and not re.search(r"(src|href)=[\"']https?:", hh), f"{len(hh)}B")

# long Igbo clips inside eval were chunked, not crashed
if "ig" in summary:
    long_ids = [x["id"] for x in refs["ig"] if x["meta"]["seconds"] > 30]
    res = {x["id"]: x for x in read_jsonl(W / "run_ig/results.jsonl")}
    c("5.15", "Clips longer than 30 s inside an eval dataset (3 Igbo clips of 31 to 34 s) are chunked automatically and transcribed",
      bool(long_ids) and all(res[i]["error"] is None and len(res[i]["prediction"] or "") > 20 for i in long_ids), f"long clips={long_ids}")

# resume on a real model
r = sh("atlasforge eval asr_ha.jsonl --task asr --out run_ha --backend local", cwd=W, log="t05_eval_ha_resume", timeout=900)
c("5.16", "Re-running a finished ASR eval resumes instantly and does not reload/re-transcribe", r.rc == 0 and "Resumed: 10" in r.out and r.secs < 60, f"{r.secs:.1f}s; {[l for l in r.out.splitlines() if 'Resumed' in l]}", "t05_eval_ha_resume")

# device selection: cpu vs auto (mps) agree closely and compare works on real ASR runs
r = sh("atlasforge eval asr_ha.jsonl --task asr --out run_ha_cpu --backend local --device cpu", cwd=W, log="t05_eval_ha_cpu", timeout=2400)
if r.rc == 0:
    cpu = json.loads((W / "run_ha_cpu/report.json").read_text())
    cw = next(m for m in cpu["metrics"] if m["key"] == "wer@tone_aware")["corpus"]
    c("5.17", "--device cpu works and gives (nearly) the same WER as the default device", abs(cw - summary["ha"][0]) < 0.03, f"cpu WER={cw:.3f} default={summary['ha'][0]:.3f}", "t05_eval_ha_cpu")
    r = sh("atlasforge compare asr_ha.jsonl --task asr --base run_ha --candidate run_ha_cpu --out cmp_asr", cwd=W, log="t05_cmp", timeout=300)
    cj = json.loads((W / "cmp_asr/comparison.json").read_text()) if r.rc == 0 else {}
    c("5.18", "compare works on real ASR runs; lower-is-better WER is oriented correctly and the two devices show no regression",
      r.rc == 0 and all(m["higher_is_better"] is False for m in cj["metrics"] if m["name"] in ("wer", "cer")) and not [m for m in cj["metrics"] if m["verdict"] == "regressed"],
      f"verdicts={[(m['key'], m['verdict']) for m in cj.get('metrics', [])][:4]}", "t05_cmp")
else:
    c("5.17", "--device cpu works", False, r.err.strip()[:200], "t05_eval_ha_cpu")

# access/doctor with real token
r = sh("atlasforge doctor --json", cwd=W, log="t05_doctor")
rows = {x["name"]: x for x in json.loads(r.out)} if r.rc == 0 else {}
c("5.19", "doctor sees the cached Hugging Face login and the installed asr extra, and checks gated access to each NCAIR1 model",
  rows.get("hf-token", {}).get("status") == "ok" and rows.get("extra:asr", {}).get("status") == "ok", f"hf-token={rows.get('hf-token', {}).get('detail')}; model-access={rows.get('model-access', {}).get('status')} {rows.get('model-access', {}).get('detail', '')[:100]}", "t05_doctor")
