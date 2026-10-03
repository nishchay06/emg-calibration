#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_COMMIT="3200d91eeb952cbed1f278e47d0cc56928334fd1"
CHECKPOINT_SHA256="338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81"

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 USER UPSTREAM_DIRECTORY DATA_DIRECTORY CHECKPOINT OUTPUT_DIRECTORY" >&2
    exit 2
fi

user="$1"
if [[ ! "${user}" =~ ^user[0-7]$ ]]; then
    echo "Invalid user '${user}'; expected user0 through user7." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
manifest="${project_dir}/manifests/${user}-sessions.txt"

if [[ ! -s "${manifest}" ]]; then
    echo "Missing or empty manifest: ${manifest}" >&2
    exit 1
fi
if [[ ! -d "$2" ]]; then
    echo "Upstream directory does not exist: $2" >&2
    exit 1
fi
if [[ ! -d "$3" ]]; then
    echo "Data directory does not exist: $3" >&2
    exit 1
fi
if [[ ! -f "$4" ]]; then
    echo "Checkpoint does not exist: $4" >&2
    exit 1
fi
if [[ -e "$5" ]]; then
    echo "Refusing to overwrite output path: $5" >&2
    exit 1
fi

upstream_dir="$(cd "$2" && pwd)"
data_dir="$(cd "$3" && pwd)"
checkpoint_dir="$(cd "$(dirname "$4")" && pwd)"
checkpoint="${checkpoint_dir}/$(basename "$4")"
python_bin="${PYTHON_BIN:-python}"
accelerator="${EMG_ACCELERATOR:-${M3_ACCELERATOR:-cpu}}"
num_workers="${EMG_NUM_WORKERS:-4}"

actual_commit="$(git -C "${upstream_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "Expected upstream ${UPSTREAM_COMMIT}; found ${actual_commit}." >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${checkpoint}" | sha256sum --check -
elif command -v shasum >/dev/null 2>&1; then
    printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${checkpoint}" | shasum -a 256 --check -
else
    echo "sha256sum or shasum is required." >&2
    exit 1
fi

while IFS= read -r archive_member; do
    [[ -n "${archive_member}" ]] || continue
    session_file="${data_dir}/${archive_member#*/}"
    if [[ ! -s "${session_file}" ]]; then
        echo "Missing or empty session file: ${session_file}" >&2
        exit 1
    fi
done < "${manifest}"

# Upstream scripts/experimental_results.py at UPSTREAM_COMMIT, Generic / No LM.
case "${user}" in
    user0) expected_val="60.07"; expected_test="61.48" ;;
    user1) expected_val="55.59"; expected_test="59.96" ;;
    user2) expected_val="47.38"; expected_test="48.00" ;;
    user3) expected_val="59.03"; expected_test="54.69" ;;
    user4) expected_val="58.93"; expected_test="58.24" ;;
    user5) expected_val="56.01"; expected_test="53.86" ;;
    user6) expected_val="58.08"; expected_test="54.66" ;;
    user7) expected_val="49.45"; expected_test="52.17" ;;
esac

mkdir -p "$5"
output_dir="$(cd "$5" && pwd)"
cd "${upstream_dir}"
"${python_bin}" -m emg2qwerty.train \
    user="${user}" \
    checkpoint="${checkpoint}" \
    train=False \
    decoder=ctc_greedy \
    dataset.root="${data_dir}" \
    trainer.accelerator="${accelerator}" \
    trainer.devices=1 \
    num_workers="${num_workers}" \
    hydra.run.dir="${output_dir}" \
    2>&1 | tee "${output_dir}/console.log"

echo "Expected upstream ${user} greedy CER: val=${expected_val}%, test=${expected_test}%"
