import json
import re

from common import *

c = Checker("1 Install and CLI surface")

r = sh("atlasforge --version", log="t01_version")
c("1.01", "atlasforge --version prints the installed version", r.rc == 0 and re.match(r"atlasforge \d", r.out), r.out, "t01_version")

r = sh("python -m atlasforge --version", log="t01_module")
c("1.02", "`python -m atlasforge` is equivalent to the command", r.rc == 0 and "atlasforge" in r.out, r.out, "t01_module")

r = sh([PY, "-c", "import sys, atlasforge, atlasforge.cli; heavy=[m for m in ('torch','transformers','librosa','peft','trl') if m in sys.modules]; print('HEAVY', heavy)"], log="t01_light")
c("1.03", "Importing atlasforge and its CLI loads no torch/transformers/librosa/peft/trl", r.rc == 0 and "HEAVY []" in r.out, r.out + r.err, "t01_light")

r = sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|librosa|peft|trl|accelerate) ' || echo none", log="t01_pip")
c("1.04", "Core install pulls in no ML framework packages", r.out.strip() == "none", r.out, "t01_pip")

r = sh("atlasforge --help", log="t01_help")
cmds = ["doctor", "run", "transcribe", "eval", "report", "compare", "demo", "finetune", "card", "dataset", "bench"]
missing = [x for x in cmds if not re.search(rf"^\s*│?\s*{x}\s", r.out, re.M)]
c("1.05", "Top-level help lists every documented command", r.rc == 0 and not missing, f"missing={missing}", "t01_help")

bad = []
for cmd in ["doctor", "run", "transcribe", "eval", "report", "compare", "demo", "finetune", "card", "dataset", "bench", "dataset validate"]:
    rr = sh(f"atlasforge {cmd} --help", log="t01_cmdhelp")
    if rr.rc != 0 or "Usage" not in rr.out:
        bad.append(cmd)
c("1.06", "Every command and subcommand has working --help (exit 0)", not bad, f"bad={bad}", "t01_cmdhelp")

r = sh("atlasforge frobnicate", log="t01_badcmd")
c("1.07", "An unknown command fails cleanly (non-zero, no traceback)", r.rc != 0 and "Traceback" not in r.both, f"rc={r.rc}", "t01_badcmd")

# ------------------------------------------------------------------ doctor
r = sh("atlasforge doctor", log="t01_doctor")
c("1.08", "doctor runs on a bare install and exits 0 when nothing hard-fails", r.rc == 0, f"rc={r.rc}; head: {r.out[:200]}", "t01_doctor")

r = sh("atlasforge doctor --json", log="t01_doctor_json")
try:
    rows = json.loads(r.out)
    names = {x["name"] for x in rows}
    ok = r.rc == 0 and {"python", "ffmpeg", "hf-token"} <= names and all({"name", "status", "detail", "hint"} <= set(x) for x in rows)
    ev = f"checks={sorted(names)}"
except Exception as exc:  # noqa: BLE001
    ok, ev = False, f"not JSON: {exc}"
c("1.09", "doctor --json is valid JSON with name/status/detail/hint per check", ok, ev, "t01_doctor_json")

fake = "hf_" + "ABCDEFGHIJ" + "KLMNOPQRST" + "UVWXYZ0123" + "456789abcd"  # built at runtime: a fake, never a real token
r = sh("atlasforge doctor", env={"HF_TOKEN": fake}, log="t01_doctor_token")
leaked = fake in r.both or fake[:20] in r.both
c("1.10", "doctor masks a Hugging Face token (shows hf_****abcd, never the secret)", (not leaked) and "****" in r.out and fake[-4:] in r.out, [l for l in r.out.splitlines() if "hf-token" in l][0] if "hf-token" in r.out else r.out[:200], "t01_doctor_token")
r = sh("atlasforge doctor --json", env={"HF_TOKEN": fake}, log="t01_doctor_token_json")
c("1.11", "doctor --json also never contains the token", fake not in r.both and fake[:20] not in r.both, "token absent from JSON", "t01_doctor_token_json")

r = sh(f"{BIN}/atlasforge doctor", env={"PATH": "/usr/bin:/bin"}, log="t01_doctor_noffmpeg")
# PATH replaced entirely: ffmpeg (homebrew) should be missing and reported with the install command
c("1.12", "doctor reports a missing ffmpeg with the install command for the OS", "ffmpeg" in r.out and ("brew install ffmpeg" in r.out or "not found" in r.out), [l for l in r.out.splitlines() if "ffmpeg" in l][:2].__str__(), "t01_doctor_noffmpeg")
