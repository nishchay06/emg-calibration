#!/usr/bin/env bash
# Takes an uploaded runtime directory, fresh output path, and Pod creation epoch.
set -euo pipefail
runtime=$1
output=$2
created=$3
setup_cutoff=$((created + 720))
measure_cutoff=$((created + 1320))
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

remaining_setup() {
    seconds=$((setup_cutoff - $(date +%s)))
    if ((seconds <= 0)); then return 124; fi
    printf '%s\n' "$seconds"
}

test ! -e "$output"
test -d /workspace/emg2qwerty-m5b-20261004/.git
test -s /workspace/emg2qwerty/models/generic.ckpt
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Reuse the retained wheel cache where possible; preserve audited versions.
timeout --kill-after=10s "$(remaining_setup)s" python -m pip install \
    torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121
timeout --kill-after=10s "$(remaining_setup)s" python -m pip install \
    -r "$runtime/requirements/m5b-training.txt"
timeout --kill-after=10s "$(remaining_setup)s" python "$runtime/scripts/adapt.py" \
    --check-config --user user5 --upstream-dir /workspace/emg2qwerty-m5b-20261004 \
    --data-dir /workspace/data --checkpoint /workspace/emg2qwerty/models/generic.ckpt \
    --output-dir "${output}-unused-plan" --accelerator gpu \
    --user0-result "$runtime/results/m5b-user0-full-upstream.json"
remaining_setup >/dev/null
deadline=$(python -c 'import sys,time; print(min(float(sys.argv[1]), time.time()+650))' "$measure_cutoff")
python "$runtime/scripts/diagnose_m5b_performance.py" --run \
    --upstream-dir /workspace/emg2qwerty-m5b-20261004 --data-dir /workspace/data \
    --checkpoint /workspace/emg2qwerty/models/generic.ckpt --output-dir "$output" \
    --deadline "$deadline"
