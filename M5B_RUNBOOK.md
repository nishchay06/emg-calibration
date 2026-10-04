# M5b runbook — full-data training reproduction

## Recipe and acceptance

Pin upstream to `3200d91eeb952cbed1f278e47d0cc56928334fd1`. Initialize the
5,293,315-parameter model from `models/generic.ckpt` (SHA-256
`338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81`).
Use Python 3.10, PyTorch 2.3.0, Lightning 1.8.6, seed 1501, batch 32,
four loader workers, upstream transforms, Adam and warmup/cosine schedule,
150 epochs, and greedy CTC decoding. Dependency pins are in
`requirements/m5b-training.txt`; CUDA experiments used Torch 2.3.0+cu121.

Reproduce user0 first, then user5. Require `last.ckpt` epoch 149 and test CER
within 1.0 percentage point of 20.57% (user0) or 5.811% (user5). Validation
CER is diagnostic; upstream validation selection is intentional for this gate.
Calibration-budget experiments must instead use fixed steps and the final
checkpoint. The harness currently supports full/full/upstream reproduction;
other advertised budget/method choices are guarded until implemented.

## Prepare and run

Prepare the pinned upstream environment and generic checkpoint with
`scripts/prepare_m3_environment.sh`. Install `requirements/m5b-training.txt`
and the pinned Torch/torchaudio/torchvision CUDA trio (2.3.0/2.3.0/0.18.0).
Stage all manifest-listed training, validation and test sessions for each user;
the upstream data module constructs all three splits. Dataset/checkpoints are
not committed. Apply only the optional-KenLM greedy compatibility patch.

Preview the exact command, then validate composition without training:

```bash
python scripts/adapt.py --dry-run --user user0
python scripts/adapt.py --check-config --user user0 \
  --upstream-dir /workspace/emg2qwerty --data-dir /workspace/data \
  --checkpoint /workspace/emg2qwerty/models/generic.ckpt \
  --output-dir /workspace/results/m5b/user0 --accelerator gpu
```

`scripts/smoke_adaptation_cpu.py` checks finite updates, the learning-rate
schedule and checkpoint restoration using synthetic inputs. For the historical
CUDA workspace layout, `python scripts/smoke_adaptation_cuda.py "$PWD"
/workspace/emg2qwerty` checks one synthetic update from the generic checkpoint.
It is a runtime gate, not a CER result.

Use the checked arguments with `--run` replacing `--check-config` and explicit
`--budget-minutes full --method full --select upstream --seed 1501`. Output
must be fresh: the harness rejects existing directories and unexpected resumes.
For user5, add `--user0-result /workspace/results/m5b/user0/result.json`; this
must identify completed, passing user0 training with checkpoint provenance.

The shell launchers preserve the executed workspace layout and bounded setup/
work deadlines. Inspect their positional arguments and paths before using them
on another machine. They do not provision or delete infrastructure; a process
timeout does not stop GPU billing. Keep evidence-copy and teardown time separate
from training/evaluation runtime.

## Observed evidence

| User | Test CER | Reference | Delta | Runtime |
|---|---:|---:|---:|---:|
| user0 | 21.209850% | 20.570% | +0.639850 pp | 34.44 min |
| user5 | 6.130137% | 5.811% | +0.319137 pp | 27.93 min |

Both completed 150 epochs. Selected checkpoints were epoch 145 (user0) and
124 (user5); last checkpoints independently prove completion. All 15 user0 and
21 user5 evidence-file checksums matched their preserved local archives.
Per-user records contain metrics, pins, hashes, runtime and qualified costs;
`results/m5b-full-upstream-summary.json` consolidates the gate.

The first user5 attempt stopped after four slow epochs, without final test CER.
Its runtime diagnosis and subsequent bounded measurements are retained publicly;
see `PERFORMANCE_DIAGNOSTIC.md`. The old slowdown was not reproduced, its cause
remains unresolved, and no recipe fix was applied. Raw archives and checkpoints
remain local under ignored `artifacts/`, subject to upstream terms.
