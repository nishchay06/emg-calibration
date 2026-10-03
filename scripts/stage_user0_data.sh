#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_URL="https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"
ARCHIVE_BYTES="308382645571"
ACKNOWLEDGEMENT="--ack-stream-308gb"
MIN_FREE_KIB=$((12 * 1024 * 1024))

if [[ $# -ne 2 || "$1" != "${ACKNOWLEDGEMENT}" ]]; then
    echo "Usage: $0 ${ACKNOWLEDGEMENT} DESTINATION_DIRECTORY" >&2
    echo "This streams the entire ${ARCHIVE_BYTES}-byte archive but retains only 14 user0 files." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
manifest="${project_dir}/manifests/user0-sessions.txt"
destination="$2"
destination_parent="$(dirname "${destination}")"
partial="${destination}.partial"

mkdir -p "${destination_parent}"
if [[ -e "${destination}" || -e "${partial}" ]]; then
    echo "Refusing to overwrite ${destination} or ${partial}" >&2
    exit 1
fi

free_kib="$(df -Pk "${destination_parent}" | awk 'NR == 2 {print $4}')"
if (( free_kib < MIN_FREE_KIB )); then
    echo "At least 12 GiB free is required before staging; found ${free_kib} KiB" >&2
    exit 1
fi

mkdir "${partial}"
echo "Streaming ${ARCHIVE_BYTES} bytes and extracting the 14 manifest entries."
curl --fail --location "${ARCHIVE_URL}" |
    tar -xzf - -C "${partial}" --strip-components=1 -T "${manifest}"

expected_count=0
while IFS= read -r archive_member; do
    [[ -n "${archive_member}" ]] || continue
    expected_count=$((expected_count + 1))
    extracted_file="${partial}/${archive_member#*/}"
    if [[ ! -s "${extracted_file}" ]]; then
        echo "Missing or empty extracted file: ${extracted_file}" >&2
        exit 1
    fi
done < "${manifest}"

actual_count="$(find "${partial}" -maxdepth 1 -type f -name '*.hdf5' | wc -l | tr -d ' ')"
if [[ "${actual_count}" != "${expected_count}" ]]; then
    echo "Expected ${expected_count} HDF5 files; found ${actual_count}" >&2
    exit 1
fi

mv "${partial}" "${destination}"
du -sh "${destination}"
echo "user0 data staged at ${destination}"
