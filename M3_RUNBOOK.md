# M3 runbook — one-user generic baseline

## Goal

Evaluate the released generic checkpoint on upstream `user0` with greedy CTC
decoding. The upstream reference values are **60.07% validation CER** and
**61.48% test CER**.

This run is pinned to upstream commit
`3200d91eeb952cbed1f278e47d0cc56928334fd1`.

## Audited inputs

- Generic checkpoint: `models/generic.ckpt`
- Checkpoint size: 63,587,626 bytes
- Checkpoint SHA-256:
  `338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81`
- Dataset sessions: 14 files—10 train, 2 validation, and 2 test
- Session manifest: [`manifests/user0-sessions.txt`](manifests/user0-sessions.txt)
- Public archive size: 308,382,645,571 bytes (about 287.2 GiB)

The upstream data module constructs all three splits even when `train=False`,
so all 14 user0 files are required. Selective extraction saves disk space but
does **not** save bandwidth: the entire compressed archive must pass through
the staging machine because the publisher provides one gzip-compressed tarball.

## Prepared commands

On a fresh Python 3.10 machine:

```bash
./scripts/prepare_m3_environment.sh /workspace/emg2qwerty
```

The setup installs the versions validated in M2, fetches only the 63.6 MB
generic checkpoint, verifies its checksum, and applies a narrow compatibility
patch. The patch makes KenLM optional for greedy decoding; requesting a KenLM
language model still fails clearly unless KenLM is installed.

Dataset staging is intentionally guarded because it streams 308 GB:

```bash
./scripts/stage_user0_data.sh --ack-stream-308gb /workspace/data
```

Then run the baseline:

```bash
./scripts/evaluate_user0_greedy.sh \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/user0-generic-greedy
```

The official upstream testing path uses CPU. Set `M3_ACCELERATOR=gpu` only if
we deliberately decide to test GPU inference.

## Spending gate

Do not provision anything until live CPU, GPU, and persistent-storage prices
have been checked. Before launch, choose a hard dollar ceiling that includes
environment setup, the full 308 GB stream, evaluation, and a cleanup check.

## Acceptance criteria

1. All 14 manifest files and the checkpoint pass preflight validation.
2. Validation and test complete without training.
3. Captured metrics are compared with 60.07% validation CER and 61.48% test
   CER; any gap is reported rather than hidden.
4. The console log and Hydra configuration remain in the result directory.
5. All paid compute is terminated and verified absent after artifacts are
   copied out.
