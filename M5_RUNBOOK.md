# M5 runbook — released personalized baselines

M5 reproduces the released personalized greedy-decoder baselines before any
calibration-budget or novel adaptation experiment. All upstream-dependent work
is pinned to commit `3200d91eeb952cbed1f278e47d0cc56928334fd1`.

## Scope and current status

The no-cost audit found two released personalized checkpoint families:

| Harness name | Upstream benchmark | Checkpoint directory | Checkpoints | Total bytes |
|---|---|---|---:|---:|
| `randominit` | Personalized (random-init) | `models/personalized-randominit` | 8 | 508,702,544 |
| `finetuned` | Personalized (finetuned) | `models/personalized-finetuned` | 8 | 508,877,392 |

The fine-tuned family is the primary baseline for later calibration work.
Upstream's personalized training example initializes from `models/generic.ckpt`;
its testing example evaluates `personalized-finetuned/${user}.ckpt` with
`train=False` and `decoder=ctc_greedy`. The random-init family is separately
named and reported in `scripts/experimental_results.py`; this harness evaluates
it through the same official testing entry point by changing only the released
checkpoint.

Both released checkpoint families passed for all eight users; the observed results are recorded below and under `results/`.

## Audited checkpoint identities

The SHA-256 values and sizes below come from the Git LFS pointers committed at
the pinned upstream revision. The machine-readable source of truth is
[`references/personalized-greedy.json`](references/personalized-greedy.json).

| User | Random-init SHA-256 | Bytes | Fine-tuned SHA-256 | Bytes |
|---|---|---:|---|---:|
| `user0` | `e22ec3be7488e02751efc007386bb1f98e7e6885a9141ddcd2f10943844fb0b4` | 63,587,818 | `b2335e1da5b3eeaf8693e4431b806e8d4dda553c37cf844414bf53da249a1474` | 63,610,538 |
| `user1` | `c96a920290520936a74367e6936edfb9828be1e6dc6006849d0f43c9e7a23b6a` | 63,587,818 | `b2db4590e6ce84d55ae3c2c86ccbe6558bb1710cd558e13968dfad2fda9c8d93` | 63,609,962 |
| `user2` | `2ac618c0e4a463ffdcfcb2df83be68fcc1807ac13d3a8d87d1ac4dfd427cd5e3` | 63,587,818 | `189c1c66677fd43968c12f717e6b9ed548e4e5aa514a836459d8d7aa8c65eeb5` | 63,609,962 |
| `user3` | `4976be783b845a7eba4053967ec3cdbd5c80df3553ca33995880e93849e3a1b3` | 63,587,818 | `e3cc48ebef8157dfc073f9fe8f6e6f62a4daba85d6e4d9ff3f7ad78a4cbb7ff6` | 63,609,386 |
| `user4` | `9af470db9beabfece9d7671361eec4f66f7bbc8dc4c9a58f634f216f7d16634d` | 63,587,818 | `d3c16cab0c855ff1260401ee47b2ccb351a6bc0adc0d219af6218b031e89f64c` | 63,609,386 |
| `user5` | `2f952c545543590ffe7c70fb7d8113071d168eab14904bff0e7671debefdbe22` | 63,587,818 | `f881b44a483c3f93584e715271303889c3142c68dfc21e9f5060a90e065bcbec` | 63,609,386 |
| `user6` | `d221780cf274579882f7475c7dd9a20c46d86ee6ec78628459797bf8e5a6ea43` | 63,587,818 | `06c85fb9998d38f839b9da5663f5534239c4dbd19f7fe60de56c8bea2244d02e` | 63,609,386 |
| `user7` | `cda34cd76c7ac5494c9fec16e451b547c128d672798ab9a7b3e2d7b3be2f8079` | 63,587,818 | `a1ef5af0e0c7ab14abf0ec0562695aa29395c52cc4239c99383e1e9b3da2a0a2` | 63,609,386 |

The official GitHub media URL pattern is:

```text
https://media.githubusercontent.com/media/facebookresearch/emg2qwerty/3200d91eeb952cbed1f278e47d0cc56928334fd1/models/personalized-{randominit|finetuned}/userN.ckpt
```

Read-only HEAD checks for both `user0` URLs returned HTTP 200, the expected
content lengths, and ETags equal to the pinned SHA-256 values. Every downloaded
file must still pass the local full-file verifier before evaluation.

## Pinned greedy references

These values are the `No LM` rows in upstream
`scripts/experimental_results.py`:

| User | Random-init val | Random-init test | Fine-tuned val | Fine-tuned test |
|---|---:|---:|---:|---:|
| `user0` | 24.13% | 26.60% | 17.96% | 20.57% |
| `user1` | 11.31% | 14.15% | 8.392% | 10.32% |
| `user2` | 10.05% | 11.12% | 8.165% | 8.409% |
| `user3` | 14.61% | 13.25% | 9.543% | 8.928% |
| `user4` | 10.64% | 10.51% | 7.575% | 7.907% |
| `user5` | 9.363% | 7.648% | 7.148% | 5.811% |
| `user6` | 22.20% | 18.82% | 17.17% | 14.21% |
| `user7` | 22.91% | 20.94% | 15.19% | 14.06% |

Regenerate or verify all identities and references without downloading a
checkpoint:

```bash
./scripts/generate_personalized_reference.py \
  /workspace/emg2qwerty \
  --check
```

## Mutation-free preflight

Preview the three-user checkpoint downloads. The two families require a
combined 381,593,916 bytes for `user0` through `user2`:

```bash
./scripts/stage_personalized_checkpoints.sh \
  --dry-run finetuned \
  /workspace/checkpoints/personalized-finetuned \
  user0 user1 user2

./scripts/stage_personalized_checkpoints.sh \
  --dry-run randominit \
  /workspace/checkpoints/personalized-randominit \
  user0 user1 user2
```

Preview the sequential output plan without checking remote inputs or creating
directories:

```bash
./scripts/evaluate_personalized_sweep.sh \
  --dry-run finetuned \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/checkpoints/personalized-finetuned \
  /workspace/results/personalized-finetuned \
  user0 user1 user2

./scripts/evaluate_personalized_sweep.sh \
  --dry-run randominit \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/checkpoints/personalized-randominit \
  /workspace/results/personalized-randominit \
  user0 user1 user2
```

## Paid execution gate

Before provisioning, recheck Runpod authentication, balance, active resources,
live GPU/storage prices, stock, and data-center compatibility. Reuse the
retained `US-IL-1` volume only after confirming the 100-file dataset and pinned
upstream checkout are intact. Present the exact resource, runtime guard,
worst-case cost, and cleanup procedure for approval.

After approval, download and verify the six gate checkpoints:

```bash
./scripts/stage_personalized_checkpoints.sh \
  --download finetuned \
  /workspace/checkpoints/personalized-finetuned \
  user0 user1 user2

./scripts/stage_personalized_checkpoints.sh \
  --download randominit \
  /workspace/checkpoints/personalized-randominit \
  user0 user1 user2
```

Then run the two sweeps with `--run` in place of `--dry-run`. Set
`EMG_ACCELERATOR=gpu` to use CUDA. Each sweep validates the upstream commit,
checkpoint filename, byte size, full SHA-256, and every manifest-selected data
file before invoking the official entry point. It captures `result.json`
immediately and stops on the first execution, capture, or acceptance failure.

## Acceptance criteria

1. Both three-user dry runs resolve distinct checkpoint and output paths and
   create nothing.
2. Every downloaded checkpoint matches its pinned filename, size, and SHA-256.
3. Both checkpoint families complete for `user0`, `user1`, and `user2`.
4. Every validation and test CER is within **0.10 percentage points** of its
   pinned family/user reference.
5. Raw console logs, Hydra configurations, and structured JSON are copied
   locally before compute deletion.
6. The Pod is deleted immediately after evidence verification; active Pods and
   unexpected volumes are checked afterward.
7. Do not expand to users 3–7 until all six gate evaluations pass.

Passing released-checkpoint evaluation does not prove that this repository can
reproduce personalized training. Full-data fine-tuning from the generic
checkpoint remains the following gate before calibration-budget experiments.

## Observed three-user gate completion

The paid `user0`–`user2` gate completed on 2026-10-04 using Secure RTX 4090 Pod
`osxf0akjj1suup` in `US-IL-1`. The Pod reused the retained 400 GB Standard
volume, and the runtime reported PyTorch and Torchaudio `2.3.0+cu121` plus the
`GREEDY_DECODER_OK` smoke marker. All six checkpoints passed filename, exact
byte-size, and full SHA-256 verification.

| Family | User | Validation CER | Reference | Test CER | Reference | Accepted |
|---|---|---:|---:|---:|---:|---|
| finetuned | `user0` | 17.963375% | 17.96% | 20.567451% | 20.57% | yes |
| finetuned | `user1` | 8.392132% | 8.392% | 10.319437% | 10.32% | yes |
| finetuned | `user2` | 8.164897% | 8.165% | 8.408756% | 8.409% | yes |
| randominit | `user0` | 24.155817% | 24.13% | 26.595289% | 26.60% | yes |
| randominit | `user1` | 11.310669% | 11.31% | 14.152680% | 14.15% | yes |
| randominit | `user2` | 10.049104% | 10.05% | 11.123974% | 11.12% | yes |

All 12 CER checks passed the 0.10-percentage-point acceptance threshold. The
ignored local evidence archive has SHA-256
`43f557a24e96b2bcd692d33264b902f544860860fb49cadd4adbd03b43c20e64`.
The Pod was deleted after local verification; read-back found zero Pods and
zero endpoints. Itemized Pod billing had not posted at the final audit; exact attributable cost remains pending.

Runpodctl 2.12 and later removed `--stop-after` and `--terminate-after` because
the backend accepted but did not enforce those deadlines. A detached watchdog
started from the local tool process was not later observable and must not be
treated as a durable guard. Future paid runs require active supervision or a
separately verified scheduler, followed by the same immediate deletion and
read-back procedure.

## Observed users 3–7 expansion completion

The expansion completed on 2026-10-04 using Secure RTX 4090 Pod
`ttd4el9vlaqfq7` in `US-IL-1`. The Pod reused the retained 400 GB Standard
volume. The ten additional checkpoints passed filename, exact byte-size, and
full SHA-256 verification.

| Family | User | Validation CER | Reference | Test CER | Reference | Accepted |
|---|---|---:|---:|---:|---:|---|
| finetuned | `user3` | 9.543159% | 9.543% | 8.916987% | 8.928% | yes |
| finetuned | `user4` | 7.563774% | 7.575% | 7.907294% | 7.907% | yes |
| finetuned | `user5` | 7.148450% | 7.148% | 5.810502% | 5.811% | yes |
| finetuned | `user6` | 17.166979% | 17.17% | 14.205607% | 14.21% | yes |
| finetuned | `user7` | 15.187770% | 15.19% | 14.048322% | 14.06% | yes |
| randominit | `user3` | 14.605755% | 14.61% | 13.249038% | 13.25% | yes |
| randominit | `user4` | 10.638298% | 10.64% | 10.508975% | 10.51% | yes |
| randominit | `user5` | 9.363463% | 9.363% | 7.648402% | 7.648% | yes |
| randominit | `user6` | 22.197468% | 22.20% | 18.785048% | 18.82% | yes |
| randominit | `user7` | 22.931206% | 22.91% | 20.935825% | 20.94% | yes |

All 20 expansion checks passed, completing all 32 M5a validation/test checks.
Across eight users, reproduced fine-tuned test CER is 11.273045% ± 4.758755%
and random-init test CER is 15.374904% ± 6.278189% (sample standard
deviations). The ignored expansion evidence archive has SHA-256
`7bf5868783af3186f8affbe7a6815fe09cdb310dc592d15379d32f50a8fca4f3`.

The Pod was deleted after local verification; read-back found zero Pods and
zero endpoints. The retained network volume remains. Itemized Pod billing had not posted at the final audit; exact attributable cost remains pending.

M5a is complete. The next gate is M5b: reproduce the upstream full-data,
full-fine-tuning recipe for `user0`, then `user5` if `user0` passes, with each
test CER within 1.0 percentage point of the released fine-tuned checkpoint.

The pinned training audit, local adaptation harness and separate training gates
are documented in [`M5B_RUNBOOK.md`](M5B_RUNBOOK.md).
