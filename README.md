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

The first generic, greedy-decoder baseline has been reproduced on upstream
`user0`:

| Split | Reproduced CER | Upstream reference | Difference |
|---|---:|---:|---:|
| Validation | 60.082565% | 60.07% | +0.012565 pp |
| Test | 61.509636% | 61.48% | +0.029636 pp |

The structured record is in
[`results/m3-user0-generic-greedy.json`](results/m3-user0-generic-greedy.json).
The remaining seven generic users and the personalized baselines are pending.

No-cost M4 preparation now includes deterministic, pinned manifests for all
eight held-out users: 100 unique sessions (68 train, 16 validation, 16 test).
See [`manifests/README.md`](manifests/README.md) for regeneration and validation.

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

1. Generalize selective staging and evaluation to `user0` through `user7`.
2. Reproduce all generic and released personalized baselines.
3. Add seeded contiguous calibration windows at 1, 2, 5, and 10 minutes.
4. Compare full, final-layer-only, normalization-only, and low-rank adaptation.
5. Report per-user curves, updated parameter counts, and adaptation time.

## Data, attribution, and license

This repository does not redistribute the emg2qwerty dataset or released model
checkpoints. Upstream code, data, and model artifacts remain subject to the
upstream project's terms. This work is not affiliated with or endorsed by Meta.

The repository is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), matching
the upstream project's license. See [`LICENSE`](LICENSE). If you use this work,
please also cite the original emg2qwerty paper and repository.
