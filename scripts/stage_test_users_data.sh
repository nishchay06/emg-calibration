#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_URL="https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"
ARCHIVE_BYTES="308382645571"
ACKNOWLEDGEMENT="--ack-stream-308gb"
DRY_RUN="--dry-run"

usage() {
    echo "Usage: $0 {${ACKNOWLEDGEMENT}|${DRY_RUN}} DESTINATION_DIRECTORY USER [USER ...]" >&2
    echo "USER must be an explicit, unique value from user0 through user7." >&2
}

if (( $# < 3 )); then
    usage
    exit 2
fi

mode="$1"
destination="$2"
shift 2

if [[ "${mode}" != "${ACKNOWLEDGEMENT}" && "${mode}" != "${DRY_RUN}" ]]; then
    usage
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
users=("$@")
selected_users=" "
selected_csv=""
expected_count=0

for user in "${users[@]}"; do
    if [[ ! "${user}" =~ ^user[0-7]$ ]]; then
        echo "Invalid user '${user}'; expected user0 through user7." >&2
        exit 2
    fi
    if [[ "${selected_users}" == *" ${user} "* ]]; then
        echo "Duplicate user '${user}'." >&2
        exit 2
    fi

    manifest="${project_dir}/manifests/${user}-sessions.txt"
    if [[ ! -s "${manifest}" ]]; then
        echo "Missing or empty manifest: ${manifest}" >&2
        exit 1
    fi

    manifest_count="$(awk 'NF {count++} END {print count+0}' "${manifest}")"
    expected_count=$((expected_count + manifest_count))
    selected_users="${selected_users}${user} "
    if [[ -n "${selected_csv}" ]]; then
        selected_csv="${selected_csv},"
    fi
    selected_csv="${selected_csv}${user}"
done

user_count="${#users[@]}"
minimum_free_gib=$((8 + (5 * user_count)))

echo "Selected users: ${selected_csv}"
echo "Required sessions: ${expected_count}"
echo "Minimum free space: ${minimum_free_gib} GiB"
echo "Source archive: ${ARCHIVE_BYTES} bytes"

if [[ "${mode}" == "${DRY_RUN}" ]]; then
    echo "Dry run only; no directories were created and no data was downloaded."
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required." >&2
    exit 1
fi
if ! command -v tar >/dev/null 2>&1; then
    echo "tar is required." >&2
    exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "GNU timeout is required." >&2
    exit 1
fi
tar_version="$(tar --version 2>&1 || true)"
if [[ "${tar_version}" != *"GNU tar"* ]]; then
    echo "GNU tar is required for selective early exit with --occurrence." >&2
    exit 1
fi

destination_parent="$(dirname "${destination}")"
partial="${destination}.partial"
mkdir -p "${destination_parent}"

if [[ -e "${destination}" || -e "${partial}" ]]; then
    echo "Refusing to overwrite ${destination} or ${partial}." >&2
    exit 1
fi

minimum_free_kib=$((minimum_free_gib * 1024 * 1024))
free_kib="$(df -Pk "${destination_parent}" | awk 'NR == 2 {print $4}')"
if (( free_kib < minimum_free_kib )); then
    echo "At least ${minimum_free_gib} GiB free is required; found ${free_kib} KiB." >&2
    exit 1
fi

selection_manifest="$(mktemp "${destination_parent}/.emg2qwerty-selected.XXXXXX")"
cleanup() {
    rm -f "${selection_manifest}"
}
trap cleanup EXIT

for user in "${users[@]}"; do
    manifest="${project_dir}/manifests/${user}-sessions.txt"
    while IFS= read -r archive_member; do
        [[ -n "${archive_member}" ]] || continue
        printf '%s\n' "${archive_member}" >> "${selection_manifest}"
    done < "${manifest}"
done
sort -u "${selection_manifest}" -o "${selection_manifest}"

unique_count="$(awk 'NF {count++} END {print count+0}' "${selection_manifest}")"
if [[ "${unique_count}" != "${expected_count}" ]]; then
    echo "Expected ${expected_count} unique sessions; found ${unique_count}." >&2
    exit 1
fi

mkdir "${partial}"
echo "Streaming until GNU tar finds all ${expected_count} requested members."
set +e
timeout 7200 curl --fail --location --progress-bar "${ARCHIVE_URL}" |
    tar -xzf - -C "${partial}" --strip-components=1 \
        --occurrence=1 -T "${selection_manifest}"
stream_status=("${PIPESTATUS[@]}")
set -e
curl_status="${stream_status[0]}"
tar_status="${stream_status[1]}"

if (( tar_status != 0 )); then
    echo "Selective extraction failed: tar exit ${tar_status}. Partial data remains at ${partial}." >&2
    exit "${tar_status}"
fi
if (( curl_status != 0 && curl_status != 23 )); then
    echo "Archive stream failed: curl exit ${curl_status}. Partial data remains at ${partial}." >&2
    exit "${curl_status}"
fi
echo "Archive pipeline completed: curl=${curl_status} tar=${tar_status}."

while IFS= read -r archive_member; do
    [[ -n "${archive_member}" ]] || continue
    extracted_file="${partial}/${archive_member#*/}"
    if [[ ! -s "${extracted_file}" ]]; then
        echo "Missing or empty extracted file: ${extracted_file}" >&2
        exit 1
    fi
done < "${selection_manifest}"

actual_count="$(find "${partial}" -maxdepth 1 -type f -name '*.hdf5' | wc -l | tr -d ' ')"
if [[ "${actual_count}" != "${expected_count}" ]]; then
    echo "Expected ${expected_count} HDF5 files; found ${actual_count}." >&2
    exit 1
fi

mv "${partial}" "${destination}"
du -sh "${destination}"
echo "Staged ${selected_csv} data at ${destination}."
