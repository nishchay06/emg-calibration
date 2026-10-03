#!/usr/bin/env bash
set -uo pipefail

UPSTREAM_COMMIT="3200d91eeb952cbed1f278e47d0cc56928334fd1"
UPSTREAM_URL="https://github.com/facebookresearch/emg2qwerty.git"
CHECKPOINT_URL="https://media.githubusercontent.com/media/facebookresearch/emg2qwerty/${UPSTREAM_COMMIT}/models/generic.ckpt"
CHECKPOINT_SHA256="338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81"
ARCHIVE_URL="https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"
WORKSPACE="/workspace"

workflow() (
    set -euo pipefail

    echo "M3_START $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python --version

    export GIT_LFS_SKIP_SMUDGE=1
    git clone --filter=blob:none "${UPSTREAM_URL}" "${WORKSPACE}/emg2qwerty"
    git -C "${WORKSPACE}/emg2qwerty" checkout "${UPSTREAM_COMMIT}"
    cd "${WORKSPACE}/emg2qwerty"

    sed -i '/^import kenlm$/c\
try:\
    import kenlm\
except ModuleNotFoundError:\
    kenlm = None' emg2qwerty/decoder.py
    sed -i '/^[[:space:]]*if self\.lm_path is not None:$/a\
            if kenlm is None:\
                raise ModuleNotFoundError(\
                    "KenLM is required when a language model path is configured"\
                )' emg2qwerty/decoder.py
    git diff --check
    python -m py_compile emg2qwerty/decoder.py

    python -m pip uninstall -y torchvision >/dev/null 2>&1 || true
    python -m pip install --no-cache-dir \
        torch==2.3.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121
    python -m pip install --no-cache-dir \
        h5py==3.11.0 \
        hydra-core==1.3.2 \
        hydra-submitit-launcher==1.2.0 \
        numpy==1.24.4 \
        omegaconf==2.3.0 \
        pytorch-lightning==1.8.6 \
        python-Levenshtein==0.12.2 \
        scipy==1.10.1 \
        torchmetrics==0.11.4 \
        unidecode==1.3.8
    python -m pip install --no-deps -e .

    checkpoint="${WORKSPACE}/emg2qwerty/models/generic.ckpt"
    curl --fail --location --output "${checkpoint}.partial" "${CHECKPOINT_URL}"
    printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${checkpoint}.partial" | sha256sum --check -
    mv "${checkpoint}.partial" "${checkpoint}"

    manifest="${WORKSPACE}/user0-sessions.txt"
    awk '/^[[:space:]]+session: / {print "emg2qwerty-data-2021-08/" $2 ".hdf5"}' \
        config/user/user0.yaml > "${manifest}"
    test "$(wc -l < "${manifest}" | tr -d ' ')" = 14

    mkdir "${WORKSPACE}/data"
    echo "DATA_STREAM_START $(date -u +%FT%TZ)"
    # GNU tar exits as soon as every requested member has been seen. That closes
    # the pipe before curl reaches the end of the 308 GB archive, so curl reports
    # CURLE_WRITE_ERROR (23). Accept that one status only when tar succeeded;
    # the file-count and non-empty checks below remain the final authority.
    set +e
    timeout 7200 curl --fail --location --progress-bar "${ARCHIVE_URL}" |
        tar -xzf - -C "${WORKSPACE}/data" --strip-components=1 \
            --occurrence=1 -T "${manifest}"
    stream_status=("${PIPESTATUS[@]}")
    set -e
    curl_status="${stream_status[0]}"
    tar_status="${stream_status[1]}"
    if (( tar_status != 0 )); then
        echo "DATA_TAR_EXIT=${tar_status}" >&2
        return "${tar_status}"
    fi
    if (( curl_status != 0 && curl_status != 23 )); then
        echo "DATA_CURL_EXIT=${curl_status}" >&2
        return "${curl_status}"
    fi
    echo "DATA_STREAM_EXIT curl=${curl_status} tar=${tar_status}"
    echo "DATA_STREAM_DONE $(date -u +%FT%TZ)"

    while IFS= read -r archive_member; do
        session_file="${WORKSPACE}/data/${archive_member#*/}"
        test -s "${session_file}"
    done < "${manifest}"
    test "$(find "${WORKSPACE}/data" -maxdepth 1 -type f -name '*.hdf5' | wc -l | tr -d ' ')" = 14
    du -sh "${WORKSPACE}/data"

    echo "EVAL_START $(date -u +%FT%TZ)"
    mkdir -p "${WORKSPACE}/results/user0-generic-greedy"
    python -m emg2qwerty.train \
        user=user0 \
        checkpoint="${checkpoint}" \
        train=False \
        decoder=ctc_greedy \
        dataset.root="${WORKSPACE}/data" \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        num_workers=4 \
        hydra.run.dir="${WORKSPACE}/results/user0-generic-greedy" \
        2>&1 | tee "${WORKSPACE}/results/user0-generic-greedy/console.log"

    echo "EXPECTED_CER val=60.07 test=61.48"
    echo "M3_PASS $(date -u +%FT%TZ)"
)

workflow
workflow_exit=$?
echo "M3_WORKFLOW_EXIT=${workflow_exit} $(date -u +%FT%TZ)"
echo "M3_HOLD_FOR_LOG_COLLECTION"
sleep 10800
exit "${workflow_exit}"
