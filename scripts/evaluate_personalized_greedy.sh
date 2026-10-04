#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_COMMIT="3200d91eeb952cbed1f278e47d0cc56928334fd1"

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 BENCHMARK USER UPSTREAM_DIRECTORY DATA_DIRECTORY CHECKPOINT OUTPUT_DIRECTORY" >&2
    echo "BENCHMARK must be randominit or finetuned; USER must be user0 through user7." >&2
    exit 2
fi

benchmark="$1"
user="$2"
case "${benchmark}" in
    randominit|finetuned) ;;
    *)
        echo "Invalid benchmark '${benchmark}'; expected randominit or finetuned." >&2
        exit 2
        ;;
esac
if [[ ! "${user}" =~ ^user[0-7]$ ]]; then
    echo "Invalid user '${user}'; expected user0 through user7." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
manifest="${project_dir}/manifests/${user}-sessions.txt"
upstream_input="$3"
data_input="$4"
checkpoint_input="$5"
output_input="$6"

if [[ ! -s "${manifest}" ]]; then
    echo "Missing or empty manifest: ${manifest}" >&2
    exit 1
fi
if [[ ! -d "${upstream_input}" ]]; then
    echo "Upstream directory does not exist: ${upstream_input}" >&2
    exit 1
fi
if [[ ! -d "${data_input}" ]]; then
    echo "Data directory does not exist: ${data_input}" >&2
    exit 1
fi
if [[ ! -f "${checkpoint_input}" ]]; then
    echo "Checkpoint does not exist: ${checkpoint_input}" >&2
    exit 1
fi
if [[ -e "${output_input}" ]]; then
    echo "Refusing to overwrite output path: ${output_input}" >&2
    exit 1
fi

upstream_dir="$(cd "${upstream_input}" && pwd)"
data_dir="$(cd "${data_input}" && pwd)"
checkpoint_dir="$(cd "$(dirname "${checkpoint_input}")" && pwd)"
checkpoint="${checkpoint_dir}/$(basename "${checkpoint_input}")"
python_bin="${PYTHON_BIN:-python}"
accelerator="${EMG_ACCELERATOR:-cpu}"
num_workers="${EMG_NUM_WORKERS:-4}"

actual_commit="$(git -C "${upstream_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "Expected upstream ${UPSTREAM_COMMIT}; found ${actual_commit}." >&2
    exit 1
fi
"${python_bin}" "${script_dir}/verify_personalized_checkpoint.py" \
    "${benchmark}" "${user}" "${checkpoint}"

while IFS= read -r archive_member; do
    [[ -n "${archive_member}" ]] || continue
    session_file="${data_dir}/${archive_member#*/}"
    if [[ ! -s "${session_file}" ]]; then
        echo "Missing or empty session file: ${session_file}" >&2
        exit 1
    fi
done < "${manifest}"

mkdir -p "${output_input}"
output_dir="$(cd "${output_input}" && pwd)"
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
