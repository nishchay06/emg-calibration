#!/usr/bin/env bash
set -euo pipefail

RAPIDGZIP_REQUIRED_VERSION="0.16.0"
RAPIDGZIP_BIN="${EMG_RAPIDGZIP_BIN:-rapidgzip}"
TAR_BIN="${EMG_TAR_BIN:-tar}"
PARALLELISM="${EMG_RAPIDGZIP_PARALLELISM:-0}"

usage() {
    echo "Usage: $0 ARCHIVE_FILE SELECTION_MANIFEST OUTPUT_DIRECTORY" >&2
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

archive_file="$1"
selection_manifest="$2"
output_directory="$3"

if [[ ! "${PARALLELISM}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "EMG_RAPIDGZIP_PARALLELISM must be zero or a positive integer; got '${PARALLELISM}'." >&2
    exit 2
fi
if [[ ! -f "${archive_file}" ]]; then
    echo "Archive file does not exist or is not a regular file: ${archive_file}" >&2
    exit 1
fi
if [[ ! -s "${selection_manifest}" ]]; then
    echo "Selection manifest is missing or empty: ${selection_manifest}" >&2
    exit 1
fi
if [[ ! -d "${output_directory}" ]]; then
    echo "Output directory does not exist: ${output_directory}" >&2
    exit 1
fi
if ! command -v "${RAPIDGZIP_BIN}" >/dev/null 2>&1; then
    echo "rapidgzip ${RAPIDGZIP_REQUIRED_VERSION} is required; install requirements/m4-staging.txt." >&2
    exit 1
fi
if ! command -v "${TAR_BIN}" >/dev/null 2>&1; then
    echo "tar is required." >&2
    exit 1
fi

rapidgzip_version="$("${RAPIDGZIP_BIN}" --version 2>&1 || true)"
if [[ "${rapidgzip_version}" != *"version ${RAPIDGZIP_REQUIRED_VERSION}"* ]]; then
    echo "rapidgzip ${RAPIDGZIP_REQUIRED_VERSION} is required; found: ${rapidgzip_version}" >&2
    exit 1
fi

echo "Parallel archive decoder: rapidgzip ${RAPIDGZIP_REQUIRED_VERSION}; parallelism=${PARALLELISM} (zero means automatic)."
echo "Scanning the complete gzip stream so CRC32 verification reaches the archive trailer."

set +e
"${RAPIDGZIP_BIN}" --decompress --stdout --verify \
    --decoder-parallelism "${PARALLELISM}" "${archive_file}" |
    "${TAR_BIN}" -xf - -C "${output_directory}" --strip-components=1 \
        -T "${selection_manifest}"
pipeline_status=("${PIPESTATUS[@]}")
set -e

decoder_status="${pipeline_status[0]}"
tar_status="${pipeline_status[1]}"
if (( decoder_status != 0 )); then
    echo "Archive integrity/decompression failed: rapidgzip exit ${decoder_status}." >&2
    exit "${decoder_status}"
fi
if (( tar_status != 0 )); then
    echo "Selective extraction failed: tar exit ${tar_status}." >&2
    exit "${tar_status}"
fi

echo "Verified archive extraction completed: rapidgzip=${decoder_status} tar=${tar_status}."
