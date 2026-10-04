#!/usr/bin/env bash
# One approved user5 reproduction; all deadlines measured from Pod creation.
set -euo pipefail
runtime=$1
output=$2
created=$3
setup_cutoff=$((created + 900))
training_cutoff=$((created + 6300))
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
trap 'code=$?; printf "%s\n" "$code" > "${output}.session.exit"' EXIT

remaining() {
    seconds=$(($1 - $(date +%s)))
    if ((seconds <= 0)); then return 124; fi
    printf '%s\n' "$seconds"
}

test ! -e "$output"
test -d /workspace/emg2qwerty-m5b-20261004/.git
test -s /workspace/emg2qwerty/models/generic.ckpt
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
timeout --kill-after=30s "$(remaining "$setup_cutoff")s" python -m pip install \
    torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121
timeout --kill-after=30s "$(remaining "$setup_cutoff")s" python -m pip install \
    -r "$runtime/requirements/m5b-training.txt"
timeout --kill-after=30s "$(remaining "$setup_cutoff")s" python "$runtime/scripts/adapt.py" \
    --check-config --user user5 --upstream-dir /workspace/emg2qwerty-m5b-20261004 \
    --data-dir /workspace/data --checkpoint /workspace/emg2qwerty/models/generic.ckpt \
    --output-dir "$output" --accelerator gpu \
    --user0-result "$runtime/results/m5b-user0-full-upstream.json"
timeout --kill-after=30s "$(remaining "$setup_cutoff")s" python \
    "$runtime/scripts/smoke_adaptation_cuda.py" "$runtime" /workspace/emg2qwerty-m5b-20261004
remaining "$setup_cutoff" >/dev/null
date -u +TRAINING_START=%FT%TZ
timeout --signal=TERM --kill-after=30s "$(remaining "$training_cutoff")s" \
    python "$runtime/scripts/adapt.py" --run \
    --user user5 --budget-minutes full --method full --select upstream --seed 1501 \
    --upstream-dir /workspace/emg2qwerty-m5b-20261004 \
    --checkpoint /workspace/emg2qwerty/models/generic.ckpt \
    --data-dir /workspace/data --output-dir "$output" --accelerator gpu \
    --user0-result "$runtime/results/m5b-user0-full-upstream.json"
