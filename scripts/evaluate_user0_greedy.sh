#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_COMMIT="3200d91eeb952cbed1f278e47d0cc56928334fd1"
CHECKPOINT_SHA256="338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 UPSTREAM_DIRECTORY DATA_DIRECTORY CHECKPOINT OUTPUT_DIRECTORY" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
manifest="${project_dir}/manifests/user0-sessions.txt"
upstream_dir="$(cd "$1" && pwd)"
data_dir="$(cd "$2" && pwd)"
checkpoint="$3"
output_dir="$4"
python_bin="${PYTHON_BIN:-python}"
accelerator="${M3_ACCELERATOR:-cpu}"

actual_commit="$(git -C "${upstream_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "Expected upstream ${UPSTREAM_COMMIT}; found ${actual_commit}" >&2
    exit 1
fi

printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${checkpoint}" | sha256sum --check -

while IFS= read -r archive_member; do
    [[ -n "${archive_member}" ]] || continue
    session_file="${data_dir}/${archive_member#*/}"
    if [[ ! -s "${session_file}" ]]; then
        echo "Missing or empty session file: ${session_file}" >&2
        exit 1
    fi
done < "${manifest}"

mkdir -p "${output_dir}"
cd "${upstream_dir}"
"${python_bin}" -m emg2qwerty.train \
    user=user0 \
    checkpoint="${checkpoint}" \
    train=False \
    decoder=ctc_greedy \
    dataset.root="${data_dir}" \
    trainer.accelerator="${accelerator}" \
    trainer.devices=1 \
    num_workers=4 \
    hydra.run.dir="${output_dir}" \
    2>&1 | tee "${output_dir}/console.log"

echo "Expected upstream user0 greedy CER: val=60.07%, test=61.48%"
