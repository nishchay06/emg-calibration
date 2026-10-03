# Project status

**Checked:** 2026-10-03

## M0 — Local preflight

The first preflight was intentionally read-only. No dataset, model checkpoint,
or external repository was downloaded.

| Check | Result |
|---|---|
| Working tree | Clean before this status file was added |
| Python | Available, but installed versions are 3.14 (Homebrew) and 3.13 (Conda base); upstream requires 3.10.13 |
| PyTorch | Not installed in either checked interpreter |
| Git LFS | Not installed |
| LaTeX compiler | Not installed (`pdflatex`) |
| Pandoc | Available |
| Local NVIDIA GPU | Not detected (`nvidia-smi` missing) |
| Free disk on Desktop volume | 22 GiB |

### Consequences

- The full benchmark cannot be staged locally on this machine yet: the plan
  estimates roughly 25–35 GB for the eight test users, in addition to the
  released checkpoints and working space.
- The next step is an upstream code audit and environment inspection, not a
  dataset download.
- Baseline training/evaluation will likely need a personal GPU machine or a
  rented GPU environment. Work accounts and work hardware remain out of scope.

## Next acceptance test

M1 is complete when we can identify the upstream commit, official evaluation
entry point, required checkpoint files, and the smallest test invocation—
without downloading the 308 GB archive.

## M1 — Upstream audit

**Status:** complete (2026-10-03)

- Upstream repository: [`facebookresearch/emg2qwerty`](https://github.com/facebookresearch/emg2qwerty)
- Audited commit: `3200d91eeb952cbed1f278e47d0cc56928334fd1`
- Default branch: `main`
- Repository status: archived upstream
- Upstream license: CC BY-NC-SA 4.0; check its terms before redistributing code,
  checkpoints, or derived artifacts.
- Required environment: Python 3.10.13, PyTorch 2.3.0, CUDA-oriented conda
  environment, and KenLM for the beam-search language model.
- Released checkpoints: `models/generic.ckpt` and the personalized checkpoint
  directories; checkpoint files are managed with Git LFS.
- Official evaluation entry point: `python -m emg2qwerty.train` with
  `train=False`, a checkpoint, a user config, and either `ctc_greedy` or
  `ctc_beam` decoding.
- Relevant upstream script: `scripts/experimental_results.py`.

The smallest useful run still requires both a released checkpoint and at least
one user's HDF5 session data, so we should not attempt it on this Mac before a
GPU/storage environment is chosen. The next acceptance test is M2: create the
environment and verify imports without downloading the full dataset.

## M2 — Runpod environment smoke test

**Status:** complete (2026-10-03)

- Tested upstream commit: `3200d91eeb952cbed1f278e47d0cc56928334fd1`
- Runpod Pod: `5di8ucuwf2ug8o` (`emg-smoke-test`), Secure Cloud in
  `EU-SE-1`
- GPU: one NVIDIA A40; CUDA was available and a CUDA tensor operation passed
- Base image: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- Verified runtime: Python 3.10, PyTorch `2.3.0+cu121`, Torchaudio
  `2.3.0+cu121`, and PyTorch Lightning `1.8.6`
- Verified imports: the `emg2qwerty` package plus `charset`, `data`, `metrics`,
  `modules`, and `transforms`
- Result marker: `SMOKE_PASS 2026-10-03T08:12:07Z`
- No dataset, checkpoint, or persistent volume was downloaded or created
- The Pod was terminated after the test, and the subsequent Pod list was empty

The first import attempt exposed `unidecode` as a required runtime dependency;
adding `unidecode==1.3.8` made the corrected run pass. Updating this ephemeral
Pod also replaced its container disk, so future repeatable runs should use a
script or custom image rather than relying on manual in-container state.

The Pod's listed price was `$0.49/hour`. It existed for under six minutes, so
the rate-based compute estimate is under `$0.05`; Runpod billing had not yet
posted the final charge when this status was written.

## M3 — One-user generic baseline

**Status:** complete (2026-10-03)

- Selected `user0` as the first reproducibility target.
- Expected greedy baseline: 60.07% validation CER and 61.48% test CER.
- Identified 14 required session files: 10 train, 2 validation, and 2 test.
- Recorded the generic checkpoint's 63,587,626-byte size and SHA-256 digest.
- Verified the archive prefix with an 8 MiB range request; the 308,382,645,571-
  byte dataset itself was not downloaded.
- Added guarded environment, data-staging, input-validation, and evaluation
  scripts; see [`M3_RUNBOOK.md`](M3_RUNBOOK.md).
- Selective streaming stopped after 15.8% of the 308 GB archive, once all 14
  requested files had been found. The extracted dataset occupied 4.4 GB.
- Reproduced `user0` greedy-decoder metrics on an RTX 4090:
  - validation CER: **60.082565%** (reference 60.07%; +0.012565 pp)
  - test CER: **61.509636%** (reference 61.48%; +0.029636 pp)
- Structured evidence: [`results/m3-user0-generic-greedy.json`](results/m3-user0-generic-greedy.json)
- The successful Pod (`jj5nmpysd7btkl`, Secure Cloud, `US-IL-1`) ran for
  1,351 seconds at `$0.74/hour`, for a rate-based estimate of `$0.2777`.
- A slower A40 attempt (`0we5bztyhpwqfu`, `EU-SE-1`) was stopped after 251
  seconds at `$0.49/hour`, estimated at `$0.0342`. Total estimated M3 compute
  was **$0.3119**; Runpod's billing records had not posted when checked.
- Both Pods were terminated, and the subsequent Pod list was empty.

The initial one-shot wrapper treated curl exit 23 as failure when selective
`tar` intentionally closed the pipe after finding every requested file. The
wrapper now accepts that status only when tar succeeds, then verifies all 14
files are present and non-empty before evaluation.

## Next acceptance test

M4 starts by making the selective staging path reusable for additional users,
then reproducing the generic baseline on a small multi-user subset before the
full eight-user sweep. Persistent storage should be considered before another
archive stream so the downloaded sessions can be reused.

### M4 multi-user generic baseline

**Status:** in progress (2026-10-03); three-user gate passed

- Audited upstream `user0` through `user7` configs at the pinned commit.
- Generated manifests for 100 unique sessions: 68 train, 16 validation, and
  16 test.
- Added deterministic regeneration with a read-only `--check` mode and local
  parser tests.
- Generalized selective staging to an explicit, duplicate-free user list and
  added a mutation-free dry-run plan.
- Generalized generic greedy evaluation to `user0` through `user7`; the old
  `user0` commands remain as compatibility wrappers.
- Added checked-in generic greedy references and automatic conversion of a
  completed Lightning console log into a structured, reference-compared result.
- Added a guarded sequential sweep with a mutation-free dry-run, per-user output
  isolation, complete overwrite preflight, and automatic result capture.
- Verified the three-user staging and evaluation dry-runs: 40 unique sessions,
  a 23 GiB minimum, and distinct sequential outputs for `user0` through `user2`.
- Added an explicit M4 acceptance guard: each validation and test CER must be
  within 0.10 percentage points of its pinned reference. A failed result remains
  captured as evidence and stops the sweep before the next user.
- Staged all 40 required `user0`-`user2` sessions. Selective extraction stopped
  after 16.9% of the archive and produced 13 GB of validated HDF5 data.
- Reproduced all six required validation/test CER values within the 0.10-point
  acceptance threshold:
  - `user0`: validation 60.082565% (+0.012565 pp), test 61.509636%
    (+0.029636 pp)
  - `user1`: validation 55.591190% (+0.001190 pp), test 59.945858%
    (-0.014142 pp)
  - `user2`: validation 47.390659% (+0.010659 pp), test 48.010944%
    (+0.010944 pp)
- The successful Secure Cloud RTX 4090 Pod (`etdelq2c8rhwbr`, `US-IL-1`)
  ran for 1,927 seconds at `$0.74/hour`, for a rate-based compute estimate of
  `$0.3961`.
- The observed Runpod balance decrease for M4 provisioning, including short
  failed EU CPU/GPU transfer attempts and temporary storage, was `$0.4607`.
- The CPU attempt in `EU-RO-1` transferred the archive at about 2.0 MB/s. A
  subsequent EU RTX 4090 single-stream probe reached 29.96 MB/s. Both were
  stopped without running evaluations. The successful run reused the M3-proven
  `US-IL-1` location rather than changing the data or evaluation procedure.
- Raw console logs and Hydra configurations were copied to the ignored local
  `artifacts/m4-runpod-2026-10-03/` directory before cleanup.
- Structured evidence is checked in under `results/m4-*.json`, with aggregate
  provenance in
  [`results/m4-three-user-gate-summary.json`](results/m4-three-user-gate-summary.json).
- All Pods and both temporary network volumes were deleted; live read-back
  returned zero Pods, zero network volumes, and `$0/hour` active spend.
- Verified the no-cost `user3`-`user7` expansion plan: 60 session files and a
  33 GiB minimum. Restaging all eight users for reuse resolves 100 files and a
  48 GiB minimum.
- Made the archive-stream timeout explicitly configurable and validated so the
  all-user run can use a four-hour termination guard; the original two-hour
  default remains unchanged.
- A 2026-10-04 all-user staging attempt on a Secure Cloud L4 in `EU-RO-1`
  failed the transfer-rate gate. Two measured windows reached 1,641,654 and
  1,668,363 bytes/second, versus the required 28,000,000 bytes/second. The
  stream was intentionally stopped after 11 minutes 42 seconds at 0.4% of the
  archive; tar consequently exited 2 and left only unvalidated partial data.
- The failed L4 Pod (`uevnc74dw12z7b`) was deleted after its raw logs were
  copied to ignored `artifacts/m4-runpod-2026-10-04/`. The 60 GB Standard
  network volume (`6jpri33smy`) remains in `EU-RO-1` at `$0.006/hour` pending
  the next staging decision. The observed balance decrease from preflight
  through cleanup was `$0.1580947138`.
- Read-only follow-up confirmed that the upstream archive is served from AWS
  `us-east-1` and supports byte-range requests. The NEMAR per-file copy uses a
  converted BDF/TSV representation rather than the upstream HDF5 inputs, so it
  is not an acceptable substitute for baseline reproduction.
- Next: approve a revised, guarded `US-IL-1` proposal that downloads the
  original archive in parallel to temporary Pod disk, selectively extracts the
  pinned HDF5 sessions to a colocated network volume, then evaluates `user3`
  through `user7` sequentially.
