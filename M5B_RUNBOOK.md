# M5b: reproduce upstream personalized training

M5a verified released-checkpoint evaluation. M5b tests whether training through
this repository reproduces full-data personalization. Calibration experiments
remain blocked until both training gates pass. No training result is claimed
by the local preparation checks.

## Pinned training audit

Upstream: `facebookresearch/emg2qwerty` at
`3200d91eeb952cbed1f278e47d0cc56928334fd1`. Paths below refer to that checkout.

| Behavior | Pinned evidence | Harness behavior |
|---|---|---|
| Entry point | `emg2qwerty/train.py:main`, README personalized training command | Launch the same `python -m emg2qwerty.train`, one user at a time |
| Composition | `config/base.yaml` defaults: user, transforms, model, Adam, warmup/cosine, greedy decoder, local cluster, `_self_` | Compose these files directly; no copied model or training implementation |
| Initialization | `train.py` calls `load_from_checkpoint` with current optimizer/scheduler/decoder config | Verify generic SHA-256, initialize weights, start fresh optimizer/scheduler; do not pass generic as `fit(ckpt_path=...)` |
| Seed/workers | `base.yaml`: 1501, four workers; `train.py`: `seed_everything(..., workers=True)` | Keep these defaults; record any explicit seed |
| Optimizer | `config/optimizer/adam.yaml`, `lightning.py:configure_optimizers`, `utils.py` | Adam, base LR 1e-3, default Adam betas/epsilon, no added regularizer |
| Schedule | `config/lr_scheduler/linear_warmup_cosine_annealing.yaml` | Bolts warmup/cosine, interval **epoch**, 10 warmup epochs, 150 total, start 1e-8, minimum 1e-6 |
| Training windows | `config/model/tds_conv_ctc.yaml`, `lightning.py:WindowedEMGDataModule` | 8,000 samples (4 seconds), padding 1,800 past / 200 future; training windows jittered, shuffled, batch 32; incomplete batch retained |
| Transforms | `config/transforms/log_spectrogram.yaml` | Preserve rotation, temporal jitter, log spectrogram and SpecAugment for training; validation/test use tensor + log spectrogram |
| Evaluation | `lightning.py` validation/test loaders | Validation uses fixed windows; test uses each complete session, no padding, batch 1 |
| Selection | `base.yaml` callbacks; `train.py` after `fit` | Minimize `val/CER`, keep best checkpoint and `last.ckpt`; reload best, then validate and test |
| Outputs | `base.yaml` Hydra output subdirectory and callback dirpath | Fresh output, `hydra_configs/`, `checkpoints/`, logger outputs, upstream log, console, plan/config/runtime/result evidence |
| Resume | `train.py`: search `Path.cwd()/checkpoints`; `utils.get_last_checkpoint`: newest **mtime**, not specifically `last.ckpt` | Refuse any existing output; `hydra.job.chdir=True` isolates the search; no resume interface yet |

The scheduler's base LR is 1e-3, but its constructor sets the starting LR to
1e-8. With upstream's epoch interval, the first epoch stays at that starting
rate. Preserve this even though the
[Bolts documentation recommends iteration stepping](https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html).
The [0.7.0 scheduler source](https://raw.githubusercontent.com/Lightning-Universe/lightning-bolts/0.7.0/src/pl_bolts/optimizers/lr_scheduler.py)
ramps linearly to the base rate, then decays with cosine annealing.

Loading module weights and resuming a Trainer have different semantics; see
[Lightning 1.8.6 checkpoint documentation](https://pytorch-lightning.readthedocs.io/en/1.8.6/common/checkpointing_basic.html).
[ModelCheckpoint 1.8.6](https://pytorch-lightning.readthedocs.io/en/1.8.6/api/pytorch_lightning.callbacks.ModelCheckpoint.html)
defaults to one best checkpoint; `save_last=True` also retains the latest state.
[Hydra 1.3](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/)
defaults to leaving the working directory unchanged. Explicitly changing it
prevents the upstream auto-resume search from seeing a different experiment.

## Small implementation scope

`scripts/adapt.py` accepts user, budget, method, seed, steps and selection, plus
explicit input/output paths and an execution mode. This increment implements
only `full/full/upstream`. Other budgets/methods and fixed-step execution fail
before mutation. Implement their semantics after M5b, under M6/M8. An upstream
run with a steps override is rejected because that would change the recipe.

- `--dry-run`: standard-library plan only; no input existence requirements,
  imports of training packages, output creation or downloads.
- `--check-config`: real Hydra composition, pinned upstream/compatibility-patch
  check, exact ordered split comparison and manifest consistency, without data,
  checkpoints or GPU. Does not instantiate the module or scheduler.
- `--run`: Python 3.10, pinned training packages, checkpoint digest and nonempty
  session checks, CPU module/optimizer/scheduler constructor probe, then the
  upstream subprocess. CPU is the default; GPU requires `--accelerator gpu`.

Training may run only `user0`, then `user5`. `user5` requires `--user0-result`
pointing to a passed M5b training result; M5a released-checkpoint evidence cannot
unlock it. A failed run returns nonzero and preserves its available evidence.
The wrapper never provisions or deletes infrastructure.

## Free local checks

Install only `requirements/m5b-config.txt` into the local virtual environment.
It contains Hydra 1.3.2, OmegaConf 2.3.0 and the upstream Submitit launcher.

```bash
.venv/bin/python scripts/adapt.py --dry-run \
  --user user0 --budget-minutes full --method full --select upstream \
  --upstream-dir upstream/emg2qwerty --data-dir /workspace/data \
  --output-dir /workspace/results/m5b/user0 --accelerator gpu

.venv/bin/python scripts/adapt.py --check-config \
  --user user0 --upstream-dir upstream/emg2qwerty --data-dir /workspace/data \
  --output-dir /workspace/results/m5b/user0 --accelerator gpu

.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/test_adapt_config.py -v
./tests/test_cli_guards.sh
EMG_RAPIDGZIP_BIN="$PWD/.venv/bin/rapidgzip" ./tests/test_parallel_staging.sh
```

Configuration tests cover all eight users and quoted paths with spaces and
Hydra punctuation. Additional guards cover leakage, step overrides, unsupported
methods, existing outputs, gate tolerance boundaries, incomplete training,
missing checkpoint provenance and accidental resume. CI explicitly runs the
composition checks after checking out pinned upstream.

## Training environment preflight still required

The local Python 3.13 environment has no Torch/Lightning. Preparation checks
therefore do **not** establish training-package import compatibility or CUDA
execution. Before the paid gate, test the pinned Python 3.10 environment,
scheduler constructor and LR trace, one synthetic optimizer update, checkpoint
save/reload, and module/decoder imports. Prefer personal CPU resources for this
remaining no-cost smoke test before provisioning.

Upstream `environment.yml` pins `lightning-bolts==0.7.0`; M3/M5a's evaluation
requirements did not install it. Bolts also
[requires torchvision and TensorBoard](https://raw.githubusercontent.com/Lightning-Universe/lightning-bolts/0.7.0/requirements/base.txt).
Use `requirements/m5b-training.txt`, which constrains the Torch trio and avoids
an unconstrained torchvision upgrade changing Torch. In the prepared CUDA
environment, install the matching trio before the training requirements:

```bash
python -m pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/m5b-training.txt
```

These are future environment instructions, not proof of an executed smoke test.
Do not substitute `prepare_m3_environment.sh` on an existing checkout: it
deliberately refuses overwrite. Keep only the audited optional-KenLM patch;
the adaptation preflight rejects other tracked upstream changes.

## Paid gate and evidence

Before provisioning, recheck authentication, balance, Pods/endpoints, volume
`ni0dpvtday`, `US-IL-1` compatibility, GPU availability and compute/storage
prices. Confirm the staged files and generic checkpoint are intact. Present
the exact GPU, attachment, runtime estimate, worst-case cost, active supervision
or verified durable termination guard, copy/cleanup plan, and obtain explicit
approval. Retained-volume storage continues even with zero Pods.

After approval and runtime smoke verification, use the same checked command
with `--run` replacing `--check-config`. Stop if `user0` fails; only then is
`user5` eligible with `--user0-result /workspace/results/m5b/user0/result.json`.

Test-only acceptance is within 1.0 pp of the pinned fine-tuned reference:
`user0` 20.57%, `user5` **5.811%** (the orientation rounds this to 5.81%).
Record validation metrics as diagnostics; do not impose the M5a 0.10 pp
validation gate on a stochastic training reproduction. The capture requires
the final checkpoint to have epoch 149, preserving evidence that all 150
epochs ran even if the selected best checkpoint is earlier.

Preserve console and upstream logs, Hydra YAML, resolved config, package freeze,
initial and selected checkpoint SHA-256, selected epoch/step, trainable/total
parameters, hardware, seed, and process wall time. `cost` starts as `null`:
supplement the committed evidence with measured balance delta/itemized billing,
Pod ID, location, prices, runtime and cleanup read-back. Do not call an estimate
an actual charge. Report CPU preflight/setup time separately from the measured
upstream subprocess time when projecting the full grid.

Copy and verify all raw evidence locally under ignored `artifacts/` before
authorized compute deletion. Check zero unexpected Pods/endpoints afterward;
retain the volume. Commit only small result records; never checkpoints, HDF5 or
raw participant data. Stop and diagnose the first failed gate. No M6 or budget
experiment may start until both M5b CER gates pass.
