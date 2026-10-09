"""Extract real FLEURS test clips from the Hugging Face cache (data prep only, not part of the product test)."""
import io
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent / "data" / "audio"
OUT.mkdir(parents=True, exist_ok=True)
N = 10
LANGS = {"ha": "ha_ng", "yo": "yo_ng", "ig": "ig_ng", "en": "en_us"}
meta = {}
for lang, code in LANGS.items():
    path = hf_hub_download("google/fleurs", f"parquet-data/{code}/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path)
    rows = table.slice(0, N).to_pylist()
    ds = []
    for i, row in enumerate(rows):
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        wav = OUT / f"{lang}_{i:02d}.wav"
        sf.write(wav, data, sr)
        ds.append({"id": f"{lang}-{i:02d}", "audio": f"audio/{wav.name}", "reference": row["raw_transcription"], "lang": lang,
                   "meta": {"seconds": round(len(data) / sr, 1), "source": f"google/fleurs {code} test #{i}"}})
    (OUT.parent / f"asr_{lang}.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in ds), encoding="utf-8")
    meta[lang] = [d["meta"]["seconds"] for d in ds]
    print(lang, "clips", len(ds), "seconds", meta[lang])

# same Hausa clip in other containers (WhatsApp-style opus/ogg, mp3, m4a)
src = OUT / "ha_00.wav"
for ext, args in (("ogg", ["-c:a", "libopus", "-b:a", "24k"]), ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]), ("m4a", ["-c:a", "aac", "-b:a", "64k"]), ("flac", [])):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *args, str(OUT / f"ha_00.{ext}")], check=True)
    print("converted", ext, (OUT / f"ha_00.{ext}").stat().st_size, "bytes")

# long recording: 5 Hausa clips back to back, reference = their references joined
import numpy as np
parts, refs = [], []
for i in range(5):
    d, sr = sf.read(OUT / f"ha_{i:02d}.wav")
    parts.append(d)
    parts.append(np.zeros(int(0.3 * sr)))
long = np.concatenate(parts)
sf.write(OUT / "ha_long.wav", long, sr)
refs = [json.loads(l)["reference"] for l in (OUT.parent / "asr_ha.jsonl").read_text(encoding="utf-8").splitlines()[:5]]
(OUT / "ha_long.txt").write_text(" ".join(refs), encoding="utf-8")
print("long audio seconds", round(len(long) / sr, 1))
