# Research status

Upstream is pinned to `3200d91eeb952cbed1f278e47d0cc56928334fd1`.
Research evidence belongs in `results/`; raw data, checkpoints and machine-local
session records stay outside version control.

## Completed gates

- M3: user0 generic greedy-decoder baseline reproduced.
- M4: generic released checkpoint reproduced for all eight users; all
  validation/test CER checks within 0.10 percentage points of reference.
- M5a: both released personalized checkpoint families reproduced for all eight
  users, with the same 0.10 pp per-split tolerance.
- M5b: pinned adaptation harness and CPU checkpoint-restoration smoke passed.
  User0 full-data reproduction passed: 150 epochs, test CER 21.209850%,
  +0.639850 pp from reference, 34.44 minutes. User5 also passed: 150 epochs,
  test CER 6.130137%, +0.319137 pp from reference, 27.93 minutes.

## Evidence policy

Publish metrics, unsuccessful experiments, recipe/version pins, checkpoint and
archive hashes, measured runtime, and clearly qualified experiment costs.
Keep account balances, access coordinates, resource identifiers and live-session
handoffs in ignored local records. Retaining a digest establishes identity;
it does not make a private raw archive publicly downloadable.

Raw participant data and model checkpoints are not redistributed. Refer to the
upstream project for permitted access and licensing. See `README.md` for the
research objective and `ROADMAP.md` for acceptance criteria.

## Interrupted user5 attempt

The first user5 attempt stopped after four epochs: 77–95 seconds per epoch
projected beyond the planned session. It produced no final test CER and has no
accuracy-gate verdict. Matching packages/recipes and a 6.842× slower synthetic
CUDA fit suggested an execution-environment regression; driver differences and
shared cuDNN warnings did not establish causality. See
`results/m5b-user5-interrupted-upstream.json` and
`results/m5b-user5-runtime-diagnosis.json`.

## Bounded performance diagnostic

All six synthetic/real train/validation phases completed without recipe changes.
The historical slowdown was not reproduced. User5 warmed training computation
was 79.08 ms/batch; loader wait varied. CPU quota/thread and stage measurements
are in `results/m5b-performance-diagnostic-20261005.json`. The exact original
cause remains unresolved. Short-batch projections are scenarios, not bounds.

## M5b completion and next gate

Both full-data runs passed the 1.0 pp test CER tolerance; selected and last
checkpoints were preserved and independently verified. All 150 epochs are
recorded for both users. `results/m5b-full-upstream-summary.json` consolidates
metrics and runtime. The full runs use upstream validation checkpoint selection
for reproduction; this must not be used for calibration-budget experiments.

M6's local deterministic calibration-window sampler, exact sample counts,
training-only windows, no validation/test overlap, and full-training-set identity
have now passed their local acceptance checks (see below).
Budget experiments use fixed optimizer steps and final-checkpoint evaluation;
hyperparameters are tuned only on user0/user1, then frozen. No calibration-minute
curve or head/norm/LoRA result is claimed yet.

## Incomplete storage migration

The 400 GB to 50 GB storage migration stopped at its file-transfer gate.
Source checks verified 100 expected HDF5 sessions (28,417,553,136 bytes), the
generic checkpoint digest, and a preservation inventory of 1,012 files
(30,539,725,849 bytes). The large dataset archive was excluded from the copy;
the original source was retained.

A bounded network-only probe passed, but all eight parallel rsync file streams
timed out. This does not establish a filesystem throughput fix; the cause
remains unresolved. The destination contains partial files and has not passed
digest, HDF5-readability or mounted-configuration verification. It must not be
used for training.

Both temporary CPU Pods were deleted; read-back at 2026-10-04 22:50:25 UTC
confirmed zero Pods and endpoints, with both volumes retained. CPU cost is an
estimate of $0.05189 through cleanup, with itemized billing pending. Raw evidence
was preserved locally and its archive digest verified. See
`results/storage-migration-attempt-20261005.json`. Training recipes and the
passed M5b gate were unchanged; M6 was the next gate at that time.

## M6 local acceptance

The deterministic sampler, bounded HDF5 dataset adapter and fixed-update runner
passed local tests. All seven budgets and eight pinned split configurations
were checked with synthetic lengths; padding, jitter and labels stay inside
selected ranges, and full-data loading delegates to upstream.

The real 5,293,315-parameter upstream model completed a synthetic CPU check:
exactly three optimizer updates, batches of 2/1/2, and learning rates
1e-8/1e-3/1e-6. No validation batches ran during fitting. Final checkpoint state
and evaluation outputs matched after reload; ordinary test and tuning-only
validation paths both completed. These checks use seeded random initialization,
not the released generic checkpoint, and establish no participant CER result.

All 52 local unit tests passed with no skips, along with shell syntax, CLI guards
and parallel archive-staging checks. Ordinary calibration runs require a frozen
profile supported by user0/user1 final-checkpoint tuning receipts; no research
profile has been tuned or frozen yet. See `M6_RUNBOOK.md` and
`results/m6-local-acceptance.json`.

Next: prepare the bounded M7 tuning/first-curve proposal. Before real training,
verify actual session coverage/allocation, generic initialization and CUDA runtime,
then tune only on user0/user1 and freeze the selected full-method profile. The
storage migration remains incomplete; the original source remains authoritative.
