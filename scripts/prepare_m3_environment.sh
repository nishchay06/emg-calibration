#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_COMMIT="3200d91eeb952cbed1f278e47d0cc56928334fd1"
UPSTREAM_URL="https://github.com/facebookresearch/emg2qwerty.git"
CHECKPOINT_URL="https://media.githubusercontent.com/media/facebookresearch/emg2qwerty/${UPSTREAM_COMMIT}/models/generic.ckpt"
CHECKPOINT_SHA256="338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 UPSTREAM_DIRECTORY" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
upstream_dir="$1"
python_bin="${PYTHON_BIN:-python}"

if [[ -e "${upstream_dir}" ]]; then
    echo "Refusing to overwrite existing path: ${upstream_dir}" >&2
    exit 1
fi

python_version="$(${python_bin} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.10" ]]; then
    echo "Python 3.10 is required; found ${python_version}" >&2
    exit 1
fi

env GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "${UPSTREAM_URL}" "${upstream_dir}"
git -C "${upstream_dir}" checkout "${UPSTREAM_COMMIT}"
git -C "${upstream_dir}" apply --check "${project_dir}/patches/greedy-decoder-optional-kenlm.patch"
git -C "${upstream_dir}" apply "${project_dir}/patches/greedy-decoder-optional-kenlm.patch"

"${python_bin}" -m pip uninstall -y torchvision >/dev/null 2>&1 || true
"${python_bin}" -m pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
"${python_bin}" -m pip install --no-cache-dir \
    -r "${project_dir}/requirements/m3-greedy.txt"
"${python_bin}" -m pip install --no-deps -e "${upstream_dir}"

checkpoint_path="${upstream_dir}/models/generic.ckpt"
checkpoint_partial="${checkpoint_path}.partial"
curl --fail --location --output "${checkpoint_partial}" "${CHECKPOINT_URL}"
printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${checkpoint_partial}" | sha256sum --check -
mv "${checkpoint_partial}" "${checkpoint_path}"

"${python_bin}" -c \
    "import torch, torchaudio; from emg2qwerty.decoder import CTCGreedyDecoder; CTCGreedyDecoder(); print(torch.__version__, torchaudio.__version__, 'GREEDY_DECODER_OK')"

echo "M3 environment prepared at ${upstream_dir}"
