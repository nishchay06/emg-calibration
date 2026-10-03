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
expect_exit 2 "${project_dir}/scripts/evaluate_generic_greedy.sh" \
    user8 /unused /unused /unused /unused
expect_exit 2 "${project_dir}/scripts/evaluate_user0_greedy.sh"

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

echo "CLI guard tests passed"
