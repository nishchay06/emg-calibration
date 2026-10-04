#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_COMMIT="3200d91eeb952cbed1f278e47d0cc56928334fd1"
BASE_URL="https://media.githubusercontent.com/media/facebookresearch/emg2qwerty/${UPSTREAM_COMMIT}/models"
RUN_MODE="--download"
DRY_RUN_MODE="--dry-run"

usage() {
    echo "Usage: $0 {${RUN_MODE}|${DRY_RUN_MODE}} BENCHMARK DESTINATION_DIRECTORY USER [USER ...]" >&2
    echo "BENCHMARK must be randominit or finetuned; USER must be explicit and unique." >&2
}

if (( $# < 4 )); then
    usage
    exit 2
fi

mode="$1"
benchmark="$2"
destination="$3"
shift 3

if [[ "${mode}" != "${RUN_MODE}" && "${mode}" != "${DRY_RUN_MODE}" ]]; then
    usage
    exit 2
fi
case "${benchmark}" in
    randominit) upstream_family="personalized-randominit" ;;
    finetuned) upstream_family="personalized-finetuned" ;;
    *)
        echo "Invalid benchmark '${benchmark}'; expected randominit or finetuned." >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
reference="${project_dir}/references/personalized-greedy.json"
python_bin="${PYTHON_BIN:-python3}"
users=("$@")
selected_users=" "
selected_csv=""

for user in "${users[@]}"; do
    if [[ ! "${user}" =~ ^user[0-7]$ ]]; then
        echo "Invalid user '${user}'; expected user0 through user7." >&2
        exit 2
    fi
    if [[ "${selected_users}" == *" ${user} "* ]]; then
        echo "Duplicate user '${user}'." >&2
        exit 2
    fi
    selected_users="${selected_users}${user} "
    if [[ -n "${selected_csv}" ]]; then
        selected_csv="${selected_csv},"
    fi
    selected_csv="${selected_csv}${user}"
done

total_bytes="$(
    "${python_bin}" -c '
import json
import sys
reference = json.load(open(sys.argv[1], encoding="utf-8"))
users = reference["benchmarks"][sys.argv[2]]["users"]
print(sum(int(users[user]["checkpoint"]["size_bytes"]) for user in sys.argv[3:]))
' "${reference}" "${benchmark}" "${users[@]}"
)"

echo "Benchmark: ${benchmark}"
echo "Selected users: ${selected_csv}"
echo "Checkpoint count: ${#users[@]}"
echo "Expected bytes: ${total_bytes}"
for user in "${users[@]}"; do
    echo "${BASE_URL}/${upstream_family}/${user}.ckpt"
done

if [[ "${mode}" == "${DRY_RUN_MODE}" ]]; then
    echo "Dry run only; no directories were created and no checkpoints were downloaded."
    exit 0
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required." >&2
    exit 1
fi

partial="${destination}.partial"
if [[ -e "${destination}" || -e "${partial}" ]]; then
    echo "Refusing to overwrite ${destination} or ${partial}." >&2
    exit 1
fi
mkdir -p "$(dirname "${destination}")"
mkdir "${partial}"

for user in "${users[@]}"; do
    checkpoint="${partial}/${user}.ckpt"
    url="${BASE_URL}/${upstream_family}/${user}.ckpt"
    echo "Downloading ${benchmark}/${user}"
    curl --fail --location --output "${checkpoint}" "${url}"
    "${python_bin}" "${script_dir}/verify_personalized_checkpoint.py" \
        "${benchmark}" "${user}" "${checkpoint}"
done

actual_count="$(find "${partial}" -maxdepth 1 -type f -name 'user*.ckpt' | wc -l | tr -d ' ')"
if [[ "${actual_count}" != "${#users[@]}" ]]; then
    echo "Expected ${#users[@]} checkpoints; found ${actual_count}." >&2
    exit 1
fi
mv "${partial}" "${destination}"
echo "Staged ${benchmark} checkpoints at ${destination}."
