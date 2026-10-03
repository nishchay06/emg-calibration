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

### M4 preparation

**Status:** in progress (2026-10-03); no paid resources created

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
- Next: verify the live Runpod balance and automatic-termination mechanism as
  the final paid-compute gate.
