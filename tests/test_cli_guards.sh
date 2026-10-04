#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
temporary_dir="$(mktemp -d)"

cleanup() {
    rm -rf "${temporary_dir}"
}
trap cleanup EXIT

expect_exit() {
    expected_status="$1"
    shift
    set +e
    "$@" > "${temporary_dir}/stdout" 2> "${temporary_dir}/stderr"
    actual_status="$?"
    set -e
    if [[ "${actual_status}" != "${expected_status}" ]]; then
        echo "Expected exit ${expected_status}, got ${actual_status}: $*" >&2
        cat "${temporary_dir}/stdout" >&2
        cat "${temporary_dir}/stderr" >&2
        exit 1
    fi
}

expect_exit 2 "${project_dir}/scripts/stage_test_users_data.sh"
expect_exit 2 "${project_dir}/scripts/stage_test_users_data.sh" \
    --dry-run /unused user8
expect_exit 2 "${project_dir}/scripts/stage_test_users_data.sh" \
    --dry-run /unused user0 user0
expect_exit 2 "${project_dir}/scripts/stage_test_users_data.sh" \
    --archive-file /unused /unused
expect_exit 2 env EMG_STAGE_TIMEOUT_SECONDS=invalid \
    "${project_dir}/scripts/stage_test_users_data.sh" \
    --dry-run /unused user0
expect_exit 2 "${project_dir}/scripts/evaluate_generic_greedy.sh" \
    user8 /unused /unused /unused /unused
expect_exit 2 "${project_dir}/scripts/evaluate_user0_greedy.sh"
expect_exit 2 "${project_dir}/scripts/evaluate_generic_sweep.sh"
expect_exit 2 "${project_dir}/scripts/evaluate_generic_sweep.sh" \
    --dry-run /unused /unused /unused /unused user8
expect_exit 2 "${project_dir}/scripts/evaluate_generic_sweep.sh" \
    --dry-run /unused /unused /unused /unused user0 user0

mkdir "${temporary_dir}/not-upstream" "${temporary_dir}/data"
touch "${temporary_dir}/checkpoint"
expect_exit 128 "${project_dir}/scripts/evaluate_generic_greedy.sh" \
    user0 \
    "${temporary_dir}/not-upstream" \
    "${temporary_dir}/data" \
    "${temporary_dir}/checkpoint" \
    "${temporary_dir}/output"
[[ ! -e "${temporary_dir}/output" ]]

user0_wrapper_plan="$(
    "${project_dir}/scripts/stage_user0_data.sh" --dry-run /unused
)"
[[ "${user0_wrapper_plan}" == *"Selected users: user0"* ]]
[[ "${user0_wrapper_plan}" == *"Required sessions: 14"* ]]

two_user_plan="$(
    "${project_dir}/scripts/stage_test_users_data.sh" \
        --dry-run /unused user0 user1
)"
[[ "${two_user_plan}" == *"Selected users: user0,user1"* ]]
[[ "${two_user_plan}" == *"Required sessions: 27"* ]]
[[ "${two_user_plan}" == *"Minimum free space: 18 GiB"* ]]

all_user_plan="$(
    "${project_dir}/scripts/stage_test_users_data.sh" \
        --dry-run /unused user0 user1 user2 user3 user4 user5 user6 user7
)"
[[ "${all_user_plan}" == *"Required sessions: 100"* ]]
[[ "${all_user_plan}" == *"Minimum free space: 48 GiB"* ]]
[[ "${all_user_plan}" == *"Stream timeout: 7200 seconds"* ]]

long_stream_plan="$(
    EMG_STAGE_TIMEOUT_SECONDS=14400 \
        "${project_dir}/scripts/stage_test_users_data.sh" \
        --dry-run /unused user3 user4 user5 user6 user7
)"
[[ "${long_stream_plan}" == *"Required sessions: 60"* ]]
[[ "${long_stream_plan}" == *"Minimum free space: 33 GiB"* ]]
[[ "${long_stream_plan}" == *"Stream timeout: 14400 seconds"* ]]

archive_destination="${temporary_dir}/archive-data"
expect_exit 1 "${project_dir}/scripts/stage_test_users_data.sh" \
    --archive-file "${temporary_dir}/missing.tar.gz" \
    "${archive_destination}" user0
[[ "$(cat "${temporary_dir}/stderr")" == *"Archive file does not exist"* ]]
[[ ! -e "${archive_destination}" ]]

touch "${temporary_dir}/wrong-size.tar.gz"
expect_exit 1 "${project_dir}/scripts/stage_test_users_data.sh" \
    --archive-file "${temporary_dir}/wrong-size.tar.gz" \
    "${archive_destination}" user0
[[ "$(cat "${temporary_dir}/stderr")" == *"expected 308382645571"* ]]
[[ ! -e "${archive_destination}" ]]

sweep_output_root="${temporary_dir}/dry-run-output"
three_user_sweep_plan="$(
    "${project_dir}/scripts/evaluate_generic_sweep.sh" \
        --dry-run \
        /workspace/emg2qwerty \
        /workspace/data \
        /workspace/emg2qwerty/models/generic.ckpt \
        "${sweep_output_root}" \
        user0 user1 user2
)"
[[ "${three_user_sweep_plan}" == *"Selected users: user0,user1,user2"* ]]
[[ "${three_user_sweep_plan}" == *"Evaluation count: 3"* ]]
[[ "${three_user_sweep_plan}" == *"[3/3] user2 -> ${sweep_output_root}/user2-generic-greedy"* ]]
[[ "${three_user_sweep_plan}" == *"Dry run only; no directories were created"* ]]
[[ ! -e "${sweep_output_root}" ]]

missing_input_output_root="${temporary_dir}/missing-input-output"
expect_exit 1 "${project_dir}/scripts/evaluate_generic_sweep.sh" \
    --run \
    "${temporary_dir}/missing-upstream" \
    "${temporary_dir}/missing-data" \
    "${temporary_dir}/missing-checkpoint" \
    "${missing_input_output_root}" \
    user0 user1
[[ ! -e "${missing_input_output_root}" ]]

echo "CLI guard tests passed"
