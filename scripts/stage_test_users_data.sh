#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_URL="https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"
ARCHIVE_BYTES="308382645571"
ACKNOWLEDGEMENT="--ack-stream-308gb"
DRY_RUN="--dry-run"
ARCHIVE_FILE_MODE="--archive-file"
STREAM_TIMEOUT_SECONDS="${EMG_STAGE_TIMEOUT_SECONDS:-7200}"

usage() {
    echo "Usage: $0 {${ACKNOWLEDGEMENT}|${DRY_RUN}} DESTINATION_DIRECTORY USER [USER ...]" >&2
    echo "       $0 ${ARCHIVE_FILE_MODE} ARCHIVE_FILE DESTINATION_DIRECTORY USER [USER ...]" >&2
    echo "USER must be an explicit, unique value from user0 through user7." >&2
    echo "Set EMG_STAGE_TIMEOUT_SECONDS to a positive integer to override the 7200-second stream guard." >&2
}

if (( $# < 3 )); then
    usage
    exit 2
fi

mode="$1"
archive_file=""
case "${mode}" in
    "${ACKNOWLEDGEMENT}"|"${DRY_RUN}")
        destination="$2"
        shift 2
        ;;
    "${ARCHIVE_FILE_MODE}")
        if (( $# < 4 )); then
            usage
            exit 2
        fi
        archive_file="$2"
        destination="$3"
        shift 3
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ "${mode}" != "${ARCHIVE_FILE_MODE}" && ! "${STREAM_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EMG_STAGE_TIMEOUT_SECONDS must be a positive integer; got '${STREAM_TIMEOUT_SECONDS}'." >&2
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
if [[ "${mode}" == "${ARCHIVE_FILE_MODE}" ]]; then
    echo "Archive file: ${archive_file}"
else
    echo "Stream timeout: ${STREAM_TIMEOUT_SECONDS} seconds"
fi

if [[ "${mode}" == "${DRY_RUN}" ]]; then
    echo "Dry run only; no directories were created and no data was downloaded."
    exit 0
fi

if [[ "${mode}" == "${ARCHIVE_FILE_MODE}" ]]; then
    if [[ ! -f "${archive_file}" ]]; then
        echo "Archive file does not exist or is not a regular file: ${archive_file}" >&2
        exit 1
    fi
    archive_file_bytes="$(wc -c < "${archive_file}" | tr -d ' ')"
    if [[ "${archive_file_bytes}" != "${ARCHIVE_BYTES}" ]]; then
        echo "Archive file has ${archive_file_bytes} bytes; expected ${ARCHIVE_BYTES}: ${archive_file}" >&2
        exit 1
    fi
    echo "Archive file size validated: ${archive_file_bytes} bytes."
fi
if ! command -v tar >/dev/null 2>&1; then
    echo "tar is required." >&2
    exit 1
fi
tar_version="$(tar --version 2>&1 || true)"
if [[ "${tar_version}" != *"GNU tar"* ]]; then
    echo "GNU tar is required for selective early exit with --occurrence." >&2
    exit 1
fi
if [[ "${mode}" == "${ACKNOWLEDGEMENT}" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required." >&2
        exit 1
    fi
    if ! command -v timeout >/dev/null 2>&1; then
        echo "GNU timeout is required." >&2
        exit 1
    fi
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
if [[ "${mode}" == "${ARCHIVE_FILE_MODE}" ]]; then
    echo "Reading and verifying the complete local archive while extracting ${expected_count} requested members."
    set +e
    "${script_dir}/extract_selected_archive.sh" \
        "${archive_file}" "${selection_manifest}" "${partial}"
    archive_status="$?"
    set -e
    tar_status="0"
    curl_status="0"
else
    echo "Streaming until GNU tar finds all ${expected_count} requested members."
    set +e
    timeout "${STREAM_TIMEOUT_SECONDS}" curl --fail --location --progress-bar "${ARCHIVE_URL}" |
        tar -xzf - -C "${partial}" --strip-components=1 \
            --occurrence=1 -T "${selection_manifest}"
    stream_status=("${PIPESTATUS[@]}")
    set -e
    curl_status="${stream_status[0]}"
    tar_status="${stream_status[1]}"
fi

if [[ "${mode}" == "${ARCHIVE_FILE_MODE}" ]] && (( archive_status != 0 )); then
    echo "Verified local-archive extraction failed: exit ${archive_status}. Partial data remains at ${partial}." >&2
    exit "${archive_status}"
fi
if (( tar_status != 0 )); then
    echo "Selective extraction failed: tar exit ${tar_status}. Partial data remains at ${partial}." >&2
    exit "${tar_status}"
fi
if [[ "${mode}" == "${ACKNOWLEDGEMENT}" ]] && (( curl_status != 0 && curl_status != 23 )); then
    echo "Archive stream failed: curl exit ${curl_status}. Partial data remains at ${partial}." >&2
    exit "${curl_status}"
fi
if [[ "${mode}" == "${ARCHIVE_FILE_MODE}" ]]; then
    echo "Archive verification and extraction completed."
else
    echo "Archive pipeline completed: curl=${curl_status} tar=${tar_status}."
fi

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
