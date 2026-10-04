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
  +0.639850 pp from reference, 34.44 minutes. User5 is the remaining gate.

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
