# Data-Efficient Personalization of sEMG Typing Decoders

[![Validate repository](https://github.com/nishchay06/emg-calibration/actions/workflows/validate.yml/badge.svg)](https://github.com/nishchay06/emg-calibration/actions/workflows/validate.yml)

This repository is a reproducibility harness and experiment plan for measuring
how much labeled calibration data a new user needs before a surface-EMG typing
decoder becomes useful. It builds on Meta's archived
[`facebookresearch/emg2qwerty`](https://github.com/facebookresearch/emg2qwerty)
benchmark and pins all upstream-dependent work to commit
`3200d91eeb952cbed1f278e47d0cc56928334fd1`.

The long-term experiment compares full and parameter-efficient adaptation at
1, 2, 5, and 10 minutes of per-user calibration data. The immediate goal is to
reproduce the released generic and personalized baselines before introducing
new methods.

## Current result

The full eight-user generic, greedy-decoder baseline passed. Every reproduced
validation and test CER is within 0.10 percentage points of its pinned upstream
reference:

| User | Validation CER | Difference | Test CER | Difference |
|---|---:|---:|---:|---:|
| `user0` | 60.082565% | +0.012565 pp | 61.509636% | +0.029636 pp |
| `user1` | 55.591190% | +0.001190 pp | 59.945858% | -0.014142 pp |
| `user2` | 47.390659% | +0.010659 pp | 48.010944% | +0.010944 pp |
| `user3` | 59.027016% | -0.002984 pp | 54.689388% | -0.000612 pp |
| `user4` | 58.939510% | +0.009510 pp | 58.236763% | -0.003237 pp |
| `user5` | 56.035351% | +0.025351 pp | 53.847031% | -0.012969 pp |
| `user6` | 58.067543% | -0.012457 pp | 54.661217% | +0.001217 pp |
| `user7` | 49.451645% | +0.001645 pp | 52.170109% | +0.000109 pp |

Across all eight users, test CER is **55.383868% mean with 4.383906 sample
standard deviation**, reproducing the upstream aggregate of 55.38% ± 4.38.

Structured records are in [`results/`](results/), including the
[`all-user summary`](results/m4-all-user-generic-greedy-summary.json). The
released personalized baselines are the next reproduction gate.

No-cost M4 preparation now includes deterministic, pinned manifests for all
eight held-out users: 100 unique sessions (68 train, 16 validation, 16 test).
See [`manifests/README.md`](manifests/README.md) for regeneration and validation.

When a complete archive is already available locally, install the pinned M4
staging dependency before using `--archive-file` mode:

```bash
python -m pip install -r requirements/m4-staging.txt
```

That path uses `rapidgzip` parallel decompression, verifies the complete gzip
stream's CRC32, and only accepts the destination after every selected HDF5
member passes the manifest checks.

## Reproduce the one-user baseline

The scripts expect Linux, Python 3.10, and a CUDA-capable machine. Dataset and
checkpoint files are downloaded from the upstream project's public locations;
they are not committed here.

```bash
# Clone and pin upstream, install dependencies, and verify the checkpoint.
./scripts/prepare_m3_environment.sh /workspace/emg2qwerty

# Explicit acknowledgement is required because the source archive is 308 GB.
./scripts/stage_user0_data.sh --ack-stream-308gb /workspace/data

# CPU is the default; set M3_ACCELERATOR=gpu for CUDA evaluation.
./scripts/evaluate_user0_greedy.sh \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/user0-generic-greedy
```

[`M3_RUNBOOK.md`](M3_RUNBOOK.md) documents the audited inputs, acceptance
criteria, observed run, and cost evidence. [`STATUS.md`](STATUS.md) records each
completed milestone and known limitation.

## Plan a multi-user baseline

Multi-user staging requires an explicit user list. Dry-run mode validates the
selection and prints session/storage requirements without creating directories
or downloading data:

```bash
./scripts/stage_test_users_data.sh \
  --dry-run /workspace/data user0 user1 user2
```

The paid form uses the same user list and requires the 308 GB acknowledgement:

```bash
./scripts/stage_test_users_data.sh \
  --ack-stream-308gb /workspace/data user0 user1 user2

./scripts/evaluate_generic_greedy.sh \
  user1 \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/user1-generic-greedy
```

For the first three-user gate, preview the sequential evaluation and automatic
result capture without running anything. The paid sweep stops after recording
the first validation or test CER more than 0.10 percentage points from its
pinned reference:

```bash
./scripts/evaluate_generic_sweep.sh \
  --dry-run \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/generic-greedy \
  user0 user1 user2
```

See [`M4_RUNBOOK.md`](M4_RUNBOOK.md) for the staged rollout and spending gate.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/` | Environment, staging, evaluation, and one-shot Pod workflows |
| `manifests/` | Exact upstream archive members required by an experiment |
| `patches/` | Narrow compatibility changes applied to pinned upstream code |
| `requirements/` | Reproducibility dependency pins |
| `results/` | Small structured result artifacts; no raw participant data |
| `project-plan.md` | Research questions, hypotheses, scope, and experiment design |

## Roadmap

1. Reproduce the released personalized baselines.
2. Add seeded contiguous calibration windows at 1, 2, 5, and 10 minutes.
3. Compare full, final-layer-only, normalization-only, and low-rank adaptation.
4. Report per-user curves, updated parameter counts, and adaptation time.

## Data, attribution, and license

This repository does not redistribute the emg2qwerty dataset or released model
checkpoints. Upstream code, data, and model artifacts remain subject to the
upstream project's terms. This work is not affiliated with or endorsed by Meta.

The repository is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), matching
the upstream project's license. See [`LICENSE`](LICENSE). If you use this work,
please also cite the original emg2qwerty paper and repository.
