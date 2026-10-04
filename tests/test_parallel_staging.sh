#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
temporary_dir="$(mktemp -d)"
rapidgzip_bin="${EMG_RAPIDGZIP_BIN:-rapidgzip}"

cleanup() {
    rm -rf "${temporary_dir}"
}
trap cleanup EXIT

expect_failure() {
    set +e
    "$@" > "${temporary_dir}/stdout" 2> "${temporary_dir}/stderr"
    status="$?"
    set -e
    if (( status == 0 )); then
        echo "Expected failure, got exit 0: $*" >&2
        exit 1
    fi
}

archive_root="${temporary_dir}/archive/emg2qwerty-data-2021-08"
mkdir -p "${archive_root}"
printf 'selected-a\n' > "${archive_root}/selected-a.hdf5"
printf 'selected-b\n' > "${archive_root}/selected-b.hdf5"
dd if=/dev/zero of="${archive_root}/ignored.bin" bs=1024 count=1024 status=none

good_archive="${temporary_dir}/good.tar.gz"
tar -czf "${good_archive}" -C "${temporary_dir}/archive" emg2qwerty-data-2021-08

selection_manifest="${temporary_dir}/selection.txt"
printf '%s\n' \
    'emg2qwerty-data-2021-08/selected-a.hdf5' \
    'emg2qwerty-data-2021-08/selected-b.hdf5' \
    > "${selection_manifest}"

success_output="${temporary_dir}/success"
mkdir "${success_output}"
EMG_RAPIDGZIP_BIN="${rapidgzip_bin}" \
    "${project_dir}/scripts/extract_selected_archive.sh" \
    "${good_archive}" "${selection_manifest}" "${success_output}"
[[ "$(cat "${success_output}/selected-a.hdf5")" == "selected-a" ]]
[[ "$(cat "${success_output}/selected-b.hdf5")" == "selected-b" ]]
[[ ! -e "${success_output}/ignored.bin" ]]

missing_manifest="${temporary_dir}/missing-selection.txt"
printf '%s\n' 'emg2qwerty-data-2021-08/missing.hdf5' > "${missing_manifest}"
missing_output="${temporary_dir}/missing-output"
mkdir "${missing_output}"
expect_failure env EMG_RAPIDGZIP_BIN="${rapidgzip_bin}" \
    "${project_dir}/scripts/extract_selected_archive.sh" \
    "${good_archive}" "${missing_manifest}" "${missing_output}"

archive_bytes="$(wc -c < "${good_archive}" | tr -d ' ')"
truncated_archive="${temporary_dir}/truncated.tar.gz"
dd if="${good_archive}" of="${truncated_archive}" bs=1 \
    count="$((archive_bytes - 8))" status=none
corrupt_output="${temporary_dir}/corrupt-output"
mkdir "${corrupt_output}"
expect_failure env EMG_RAPIDGZIP_BIN="${rapidgzip_bin}" \
    "${project_dir}/scripts/extract_selected_archive.sh" \
    "${truncated_archive}" "${selection_manifest}" "${corrupt_output}"
[[ "$(cat "${temporary_dir}/stderr")" == *"Archive integrity/decompression failed"* ]]

expect_failure env \
    EMG_RAPIDGZIP_BIN="${rapidgzip_bin}" \
    EMG_RAPIDGZIP_PARALLELISM=invalid \
    "${project_dir}/scripts/extract_selected_archive.sh" \
    "${good_archive}" "${selection_manifest}" "${temporary_dir}/invalid-output"

echo "Parallel staging tests passed"
