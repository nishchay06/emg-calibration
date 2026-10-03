# M4 runbook — multi-user baseline preparation

## Goal

Reproduce generic greedy-decoder baselines on a small held-out-user subset,
then expand to all eight users only after the subset matches upstream values.
No paid resource is provisioned by the commands in the planning section.

All inputs remain pinned to upstream commit
`3200d91eeb952cbed1f278e47d0cc56928334fd1`.

## Inventory

The eight official held-out-user configs require 100 unique sessions:

- 68 training sessions
- 16 validation sessions
- 16 test sessions

Counts and provenance are recorded in
[`manifests/test-users.json`](manifests/test-users.json). Regeneration and
validation are documented in [`manifests/README.md`](manifests/README.md).

## No-cost planning

Plan the first three-user subset without creating a directory or starting a
download:

```bash
./scripts/stage_test_users_data.sh \
  --dry-run /workspace/data user0 user1 user2
```

Expected plan:

```text
Selected users: user0,user1,user2
Required sessions: 40
Minimum free space: 23 GiB
Source archive: 308382645571 bytes
Dry run only; no directories were created and no data was downloaded.
```

Plan all eight users with the explicit list:

```bash
./scripts/stage_test_users_data.sh \
  --dry-run /workspace/data \
  user0 user1 user2 user3 user4 user5 user6 user7
```

That plan resolves 100 sessions and requires at least 48 GiB free space.

## Paid staging command

Run this only after live compute and storage prices have been checked and a
hard spending ceiling has been agreed:

```bash
./scripts/stage_test_users_data.sh \
  --ack-stream-308gb /workspace/data user0 user1 user2
```

The script requires GNU tar and GNU timeout. It builds a unique manifest,
streams the public archive until every requested member is found, accepts curl
exit 23 only when tar completed successfully, validates every non-empty HDF5
file, and refuses to overwrite an existing destination.

## Per-user evaluation

```bash
EMG_ACCELERATOR=gpu ./scripts/evaluate_generic_greedy.sh \
  user0 \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/user0-generic-greedy
```

Use a distinct output directory for each user. The evaluator verifies the
upstream commit, checkpoint digest, complete per-user manifest, and output-path
nonexistence before invoking the official test path.

After a run completes, convert its console output into a small structured
artifact while preserving the raw log:

```bash
./scripts/capture_generic_result.py \
  user0 \
  /workspace/results/user0-generic-greedy/console.log \
  /workspace/results/user0-generic-greedy/result.json
```

The capture command requires all five validation and test metrics, refuses to
overwrite an existing result, and calculates CER differences from the pinned
references in [`references/generic-greedy.json`](references/generic-greedy.json).
It records whether each absolute CER difference is at most **0.10 percentage
points**. A failed comparison is still written to `result.json` as evidence,
then the command exits nonzero.

## Three-user sweep

Preview the complete evaluation plan without checking inputs, creating output
directories, or starting an evaluation:

```bash
./scripts/evaluate_generic_sweep.sh \
  --dry-run \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/generic-greedy \
  user0 user1 user2
```

After staging and input verification, replace `--dry-run` with `--run`. The
sweep evaluates users sequentially, gives each user a separate output
directory, captures `result.json` immediately after each successful run, and
stops on the first execution, capture, or CER-acceptance failure. It performs a
full overwrite preflight before creating the output root.

The first-subset greedy references from upstream
`scripts/experimental_results.py` are:

| User | Validation CER | Test CER |
|---|---:|---:|
| `user0` | 60.07% | 61.48% |
| `user1` | 55.59% | 59.96% |
| `user2` | 47.38% | 48.00% |

## Acceptance gate

1. The three-user sweep dry-run resolves three distinct output directories
   without creating them.
2. `user0`, `user1`, and `user2` validation/test runs complete without training.
3. Each observed CER is captured in `result.json`; both validation and test
   must be within 0.10 percentage points of their pinned upstream per-user
   values. The sweep stops after recording the first failed comparison.
4. Raw console logs and Hydra configs are copied out before ephemeral compute is
   terminated.
5. The all-user sweep is not started until the three-user subset passes.
6. Every paid Pod and unused persistent volume is explicitly deleted and then
   verified absent.

## Observed three-user gate

The gate passed on 2026-10-03 using the pinned upstream commit and generic
checkpoint on a Secure Cloud RTX 4090 in `US-IL-1`.

| User | Validation CER | Reference | Test CER | Reference | Accepted |
|---|---:|---:|---:|---:|---|
| `user0` | 60.082565% | 60.07% | 61.509636% | 61.48% | yes |
| `user1` | 55.591190% | 55.59% | 59.945858% | 59.96% | yes |
| `user2` | 47.390659% | 47.38% | 48.010944% | 48.00% | yes |

Selective staging found all 40 sessions after streaming 16.9% of the source
archive; the extracted data occupied 13 GB. The successful Pod ran for 1,927
seconds at `$0.74/hour`, an estimated `$0.3961` in compute. The total observed
Runpod balance decrease for this milestone was `$0.4607`, including short
failed transfer attempts and temporary storage.

Raw logs and Hydra configurations were copied locally before cleanup. The
checked-in results are listed by
[`results/m4-three-user-gate-summary.json`](results/m4-three-user-gate-summary.json).
Post-cleanup read-back returned zero Pods and zero network volumes.
