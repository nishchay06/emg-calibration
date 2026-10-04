#!/usr/bin/env bash
set -euo pipefail

RUN_MODE="--run"
DRY_RUN_MODE="--dry-run"

usage() {
    echo "Usage: $0 {${RUN_MODE}|${DRY_RUN_MODE}} BENCHMARK UPSTREAM_DIRECTORY DATA_DIRECTORY CHECKPOINT_DIRECTORY OUTPUT_ROOT USER [USER ...]" >&2
    echo "BENCHMARK must be randominit or finetuned; USER must be explicit and unique." >&2
}

if (( $# < 7 )); then
    usage
    exit 2
fi

mode="$1"
benchmark="$2"
upstream_dir="$3"
data_dir="$4"
checkpoint_dir="$5"
output_root="$6"
shift 6

if [[ "${mode}" != "${RUN_MODE}" && "${mode}" != "${DRY_RUN_MODE}" ]]; then
    usage
    exit 2
fi
case "${benchmark}" in
    randominit|finetuned) ;;
    *)
        echo "Invalid benchmark '${benchmark}'; expected randominit or finetuned." >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

echo "Benchmark: ${benchmark}"
echo "Selected users: ${selected_csv}"
echo "Evaluation count: ${#users[@]}"
for index in "${!users[@]}"; do
    user="${users[$index]}"
    checkpoint="${checkpoint_dir}/${user}.ckpt"
    output_dir="${output_root}/${user}-personalized-${benchmark}-greedy"
    printf '[%d/%d] %s checkpoint=%s output=%s\n' \
        "$((index + 1))" "${#users[@]}" "${user}" "${checkpoint}" "${output_dir}"
done

if [[ "${mode}" == "${DRY_RUN_MODE}" ]]; then
    echo "Dry run only; no directories were created and no evaluations were run."
    exit 0
fi

if [[ ! -d "${upstream_dir}" ]]; then
    echo "Upstream directory does not exist: ${upstream_dir}" >&2
    exit 1
fi
if [[ ! -d "${data_dir}" ]]; then
    echo "Data directory does not exist: ${data_dir}" >&2
    exit 1
fi
if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "Checkpoint directory does not exist: ${checkpoint_dir}" >&2
    exit 1
fi
if [[ -e "${output_root}" && ! -d "${output_root}" ]]; then
    echo "Output root exists but is not a directory: ${output_root}" >&2
    exit 1
fi

for user in "${users[@]}"; do
    checkpoint="${checkpoint_dir}/${user}.ckpt"
    output_dir="${output_root}/${user}-personalized-${benchmark}-greedy"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Checkpoint does not exist: ${checkpoint}" >&2
        exit 1
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing to overwrite output path: ${output_dir}" >&2
        exit 1
    fi
done

mkdir -p "${output_root}"
for index in "${!users[@]}"; do
    user="${users[$index]}"
    checkpoint="${checkpoint_dir}/${user}.ckpt"
    output_dir="${output_root}/${user}-personalized-${benchmark}-greedy"
    echo "Running $((index + 1))/${#users[@]}: ${benchmark}/${user}"
    "${script_dir}/evaluate_personalized_greedy.sh" \
        "${benchmark}" \
        "${user}" \
        "${upstream_dir}" \
        "${data_dir}" \
        "${checkpoint}" \
        "${output_dir}"
    "${script_dir}/capture_personalized_result.py" \
        "${benchmark}" \
        "${user}" \
        "${output_dir}/console.log" \
        "${output_dir}/result.json"
done

echo "Completed ${#users[@]} ${benchmark} personalized greedy evaluations under ${output_root}."
