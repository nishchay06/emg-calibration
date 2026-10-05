# Data-Efficient Personalization of sEMG Typing Decoders

[![Validate repository](https://github.com/nishchay06/emg-calibration/actions/workflows/validate.yml/badge.svg)](https://github.com/nishchay06/emg-calibration/actions/workflows/validate.yml)

This repository is a reproducibility harness and experiment plan for measuring
how much labeled calibration data a new user needs before a surface-EMG typing
decoder becomes useful. It builds on Meta's archived
[`facebookresearch/emg2qwerty`](https://github.com/facebookresearch/emg2qwerty)
benchmark and pins all upstream-dependent work to commit
`3200d91eeb952cbed1f278e47d0cc56928334fd1`.

The long-term experiment compares full and parameter-efficient adaptation at
1, 2, 5, 10, 30, and 60 minutes plus full per-user calibration data. The
released generic and personalized checkpoints have been reproduced. Our own
full-data fine-tuning also passed the two-user upstream reproduction gate. The
local calibration-budget sampler and fixed-update protocol have passed their
CPU acceptance checks. The next milestone is tuning and the first full-method
curve; CER-versus-minutes results and head/norm/LoRA comparisons remain future work. See
[`M5B_RUNBOOK.md`](M5B_RUNBOOK.md) for the recipe and training evidence.
See [`M6_RUNBOOK.md`](M6_RUNBOOK.md) for calibration allocation and fixed-step
training; actual recording coverage and a tuned/frozen research profile remain
required before participant experiments.
See [`M7_ANALYSIS_RUNBOOK.md`](M7_ANALYSIS_RUNBOOK.md) for offline receipt checks,
tables and figure exports; its synthetic demo establishes no calibration result.

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
released personalized checkpoint evaluation also passed for both families and
all eight users: all 32 validation/test CER checks are within 0.10 percentage
points of their pinned references.

| Personalized family | Reproduced mean test CER | Upstream mean test CER |
|---|---:|---:|
| Fine-tuned from generic | 11.273045% ± 4.758755% | 11.276875% ± 4.759960% |
| Trained from random initialization | 15.374904% ± 6.278189% | 15.379750% ± 6.283013% |

These are sample standard deviations across users. See the
[`M5 all-user summary`](results/m5-all-user-personalized-greedy-summary.json)
and [`M5_RUNBOOK.md`](M5_RUNBOOK.md) for per-user results, checkpoint
identities, raw-evidence provenance, and qualified experiment costs.

### Own full-data fine-tuning (M5b)

Both runs completed 150 epochs with the upstream recipe, generic initialization,
seed 1501 and greedy decoding. Test CER passed the 1.0 pp reference tolerance:

| User | Test CER | Reference | Difference | Training/evaluation |
|---|---:|---:|---:|---:|
| `user0` | 21.209850% | 20.570% | +0.639850 pp | 34.44 min |
| `user5` | 6.130137% | 5.811% | +0.319137 pp | 27.93 min |

See the [`M5b summary`](results/m5b-full-upstream-summary.json) for checkpoint
provenance and per-user records. An earlier user5 attempt stopped for runtime
rather than accuracy; its failed attempt and diagnostic measurements remain
published. Full-data reproduction selects the best validation checkpoint;
calibration-budget experiments will use fixed steps and final-checkpoint
selection to avoid spending uncounted validation labels. These are one-seed
reproduction results, not the completed calibration study.

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
criteria, observed run, and cost evidence. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) records each
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
| `scripts/` | Environment, staging, evaluation, adaptation and bounded diagnostics |
| `manifests/` | Exact upstream archive members required by an experiment |
| `patches/` | Narrow compatibility changes applied to pinned upstream code |
| `requirements/` | Reproducibility dependency pins |
| `configs/` | Versioned protocol examples and future frozen method profiles |
| `results/` | Small structured result artifacts; no raw participant data |
| `project-plan.md` | Research questions, hypotheses, scope, and experiment design |

## Roadmap

1. Completed: reproduce upstream full-data fine-tuning for two users.
2. Completed locally: seeded calibration windows and fixed-step/final-checkpoint protocol.
3. Verify real recording coverage, tune on user0/user1, freeze the profile and produce the first full-method curve.
4. Compare full, final-layer-only, normalization-only, and low-rank adaptation.
5. Report per-user curves, trainable parameter counts, adaptation time, and cost.

## Data, attribution, and license

This repository does not redistribute the emg2qwerty dataset or released model
checkpoints. Upstream code, data, and model artifacts remain subject to the
upstream project's terms. This work is not affiliated with or endorsed by Meta.

The repository is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), matching
the upstream project's license. See [`LICENSE`](LICENSE). If you use this work,
please also cite the original emg2qwerty paper and repository.

## Contributing and evidence

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for local validation commands and
change-review conventions. Publish numerical results, failed experiments,
recipe/version pins and hashes. Account balances, access coordinates and live
session records stay in ignored local files. Raw archives are retained locally;
published hashes identify them but do not provide public download access.
