#!/bin/bash
# Re-runs the suite against what PyPI serves now, in fresh environments, with NO workarounds.
cd "$(dirname "$0")"
set -u
rm -rf venv-rel venv-rel-asr venv-rel314
python3 -m venv venv-rel && venv-rel/bin/pip install -q --upgrade pip && venv-rel/bin/pip install brainers-atlasforge > logs/rel__install_core.log 2>&1
python3 -m venv venv-rel-asr && venv-rel-asr/bin/pip install -q --upgrade pip && venv-rel-asr/bin/pip install "brainers-atlasforge[asr]" > logs/rel-asr__install.log 2>&1
/opt/homebrew/bin/python3.14 -m venv venv-rel314 && venv-rel314/bin/pip install -q --upgrade pip && venv-rel314/bin/pip install "brainers-atlasforge[finetune]" > logs/rel314__install.log 2>&1
export AF_VENV=$PWD/venv-rel AF_LABEL=rel
for t in t01_core t02_offline t03_data t04_eval t07_toolkit t10_extras t08_public t06_llm; do venv-rel/bin/python $t.py > logs/rel_${t}_stdout.log 2>&1; done
export AF_VENV=$PWD/venv-rel-asr AF_LABEL=rel-asr
venv-rel-asr/bin/python t05_asr.py > logs/rel_t05_stdout.log 2>&1
export AF_VENV=$PWD/venv-rel314 AF_LABEL=rel314
venv-rel314/bin/python t09_finetune.py > logs/rel_t09_stdout.log 2>&1
echo RELDONE > logs/rel_done.txt
