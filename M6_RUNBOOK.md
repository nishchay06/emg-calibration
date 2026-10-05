# M6 runbook: calibration allocation and fixed-update training

## Acceptance and scope

M6 is a local implementation gate: exact allocations, deterministic selection,
training-only access, full-data compatibility, and final-checkpoint training.
Budgets are 1, 2, 5, 10, 30, 60 minutes and full. One minute is 120,000 samples
at the nominal 2 kHz rate; gaps between sessions do not count. The local gate
uses synthetic HDF5 fixtures and pinned upstream code, not participant CER.

Actual recording coverage, generic initialization and Linux/CUDA readiness must
pass before M7 training. No step count or learning rate has been tuned yet.
`configs/m6-protocol-example.json` is a five-update CLI illustration, not a
research configuration. Ordinary runs reject its draft status.

## Deterministic allocation

`scripts/calibration_sampler.py` selects ranges from official training splits.
If a budget fits one session, it selects an eligible session uniformly and a
valid offset uniformly. Otherwise it chooses a feasible starting session/offset
and continues through training sessions ordered by their filename recording
timestamps. It never wraps, repeats samples or silently reduces a budget.
`full` preserves every training session and upstream ordering.

Ranges use start-inclusive, stop-exclusive indices. Records include counts,
split/length digests, seed and algorithm version. Selection depends on user,
budget and seed, independently of model/method and training RNG. Budgets need
not be nested. Dataset views constrain padding and jitter to selected ranges;
supervised labels retain upstream unpadded-window timestamp conventions.
Sessions are windowed separately. Short fragments yield no complete windows.

Log nominal window capacity and remainders separately from allocated samples.
Context and jitter can change which allocated samples are read, and repeated
training does not create additional calibration data. The production batch cap
is 32, with smaller batches retained and repeated shuffled passes until S updates.

## Local checks

Use the pinned Python 3.10 CPU training environment and upstream checkout
at `3200d91eeb952cbed1f278e47d0cc56928334fd1`. Install the existing training
requirements with the pinned Torch trio; CPU fixtures need no released checkpoint.
Apply the audited optional-KenLM patch when KenLM is absent.

```bash
python -m unittest discover -s tests -v
python scripts/adapt.py --check-config --user user0 \
  --select fixed --budget-minutes 1 \
  --protocol configs/m6-protocol-example.json \
  --upstream-dir upstream/emg2qwerty --data-dir data \
  --output-dir artifacts/m6-config-preview
```

The tests exercise all eight user configurations and seven budgets. CPU tests
skip when training dependencies are unavailable; skipped tests do not establish
the complete local gate. Run them with the full CPU environment. To preserve
the synthetic integration checkpoint/config/proof, set `EMG_M6_EVIDENCE_DIR`
to a fresh ignored directory when running `tests/test_fixed_adaptation.py`.

## Mounted-data preflight before M7

Read headers from the complete official 100-session directory, then check all
56 user/budget allocations. Output records contain counts, not participant signals.

```bash
python scripts/index_calibration_sessions.py \
  --upstream-dir /workspace/emg2qwerty --data-dir /workspace/data \
  > /workspace/session-index.json
python scripts/check_calibration_coverage.py \
  --upstream-dir /workspace/emg2qwerty \
  --session-index /workspace/session-index.json \
  > /workspace/calibration-coverage.json
```

This checks schema, identity, split membership and lengths. It does not replace
file-digest or signal-integrity verification. The runner reindexes the supplied
directory before training and rejects stale supplied allocations. Data files
remain directly under the supplied directory, without the archive prefix.

Preview a concrete allocation by adding `--session-index` to the fixed-mode
`adapt.py --dry-run` command. Without an index, a dry run explicitly records
data preflight as pending and creates no files. `--check-config` checks composition,
not mounted data or generic-checkpoint availability.

## Tuning and final-checkpoint execution

A method profile specifies S, learning rate, warmup updates, warmup start LR
and minimum LR. The scheduler advances once per optimizer update, regardless
of dataset epochs. Update indices are 0 through S-1. Warmup ends at the peak;
cosine decay reaches minimum LR on the final update when S exceeds one.

`--select fixed --run --tune` is allowed only for user0/user1 with an explicit
candidate protocol. Fitting has no validation batches, sanity validation,
early stopping or best-checkpoint selection. After exactly S updates, the final
checkpoint is saved, verified and reloaded. Tuning evaluates only validation;
ordinary adaptation evaluates only test. Tune candidates in a separately
approved bounded session, with candidate sets and budget choices declared first.

Freeze the chosen candidate using its completed tuning records:

```bash
python scripts/freeze_calibration_protocol.py \
  --candidate /workspace/chosen-candidate.json \
  --tuning-result /workspace/user0-tuning/result.json \
  --tuning-result /workspace/user1-tuning/result.json \
  > /workspace/frozen-protocol.json
```

Freezing requires matching final-checkpoint receipts for both tuning users and
no test metrics. Ordinary `--run` requires a frozen protocol; `--steps` cannot
override its method profile. Fresh output is required; implicit resume is disabled.
Raw checkpoints, console and result records go under ignored `artifacts/` when
inside this repository. No CER acceptance tolerance from M5b is applied to a
calibration result. Keep M5b's upstream-selection reference distinct from the
fixed-protocol full-data point. Head/norm/LoRA remain gated until M8.
