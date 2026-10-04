#!/usr/bin/env python3
"""Exercise pinned upstream training on synthetic EMG, with no data downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import adapt


def check(condition, message):
    if not condition:
        raise ValueError(message)


def exercise(upstream: Path, output: Path, evidence: dict):
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(upstream))
    from emg2qwerty import transforms, utils
    from emg2qwerty.data import LabelData, WindowedEMGDataset
    from emg2qwerty.decoder import CTCGreedyDecoder

    check(sys.version_info[:3] == (3, 10, 13), "Expected Python 3.10.13")
    pins = {
        "torch": "2.3.0", "torchaudio": "2.3.0", "torchvision": "0.18.0",
        "pytorch-lightning": "1.8.6", "lightning-bolts": "0.7.0",
        "torchmetrics": "0.11.4", "hydra-core": "1.3.2", "omegaconf": "2.3.0",
        "numpy": "1.24.4", "setuptools": "69.5.1", "pip": "24.0",
    }
    for package, expected in pins.items():
        check(version(package).split("+")[0] == expected, f"Unexpected {package} version")
    evidence["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                               "architecture": platform.machine(), "packages": pins,
                               "accelerator_used": "cpu", "torch_num_threads": 2}
    torch.set_num_threads(2)
    pl.seed_everything(1501, workers=True)
    args = adapt.arguments([
        "--check-config", "--user", "user0", "--upstream-dir", str(upstream),
        "--data-dir", str(output / "unused-data"), "--output-dir", str(output / "config-plan"),
    ])
    config = adapt.compose_config(args)
    (output / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    evidence["checks"]["pinned_composition_and_imports"] = True

    # Exercise the real scheduler constructor/step implementation independently
    # of model training. These optimizer calls have no gradient or weight update.
    dummy = torch.nn.Parameter(torch.ones(1))
    setup = utils.instantiate_optimizer_and_scheduler(
        iter([dummy]), OmegaConf.create(config["optimizer"]), OmegaConf.create(config["lr_scheduler"])
    )
    optimizer = setup["optimizer"]
    scheduler = setup["lr_scheduler"]["scheduler"]
    trace = [optimizer.param_groups[0]["lr"]]
    for _ in range(150):
        optimizer.step()
        scheduler.step()
        trace.append(optimizer.param_groups[0]["lr"])
    expected = [
        1e-8 + epoch * (1e-3 - 1e-8) / 9 if epoch < 10 else
        1e-6 + 0.5 * (1e-3 - 1e-6) * (1 + math.cos(math.pi * (epoch - 10) / 140))
        for epoch in range(151)
    ]
    maximum_error = max(abs(actual - target) for actual, target in zip(trace, expected))
    check(maximum_error < 1e-12, f"Unexpected LR trace: max error {maximum_error}")
    check(setup["lr_scheduler"]["interval"] == "epoch", "Scheduler must step per epoch")
    evidence["learning_rate_schedule"] = {
        "interval": "epoch", "indices": "scheduler index 0 before first epoch through 150",
        "trace": trace, "maximum_absolute_error": maximum_error,
    }
    evidence["checks"]["warmup_cosine_trace"] = True

    cfg = OmegaConf.create(config)
    module = instantiate(cfg.module, optimizer=cfg.optimizer, lr_scheduler=cfg.lr_scheduler,
                         decoder=cfg.decoder, _recursive_=False)
    check(isinstance(module.decoder, CTCGreedyDecoder), "Expected upstream greedy decoder")
    evidence["parameters"] = {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
    }
    check(evidence["parameters"]["total"] == evidence["parameters"]["trainable"], "Full model must be trainable")
    initial_head = module.model[4].weight.detach().clone()

    # Match the upstream raw-input fields and nominal 4-second window plus
    # context. No participant data or generic checkpoint is used in this probe.
    transform_sets = {
        phase: transforms.Compose([instantiate(item) for item in cfg.transforms[phase]])
        for phase in ("train", "val")
    }
    rng = np.random.default_rng(1501)
    targets = torch.tensor(LabelData.from_str("emgtest").labels, dtype=torch.int64)
    samples = {"train": [], "val": []}
    raw_length = cfg.datamodule.window_length + sum(cfg.datamodule.padding)
    for _ in range(2):
        raw = np.zeros(raw_length, dtype=[("emg_left", np.float32, (16,)),
                                         ("emg_right", np.float32, (16,)), ("time", np.float64)])
        raw["emg_left"] = rng.standard_normal((raw_length, 16), dtype=np.float32)
        raw["emg_right"] = rng.standard_normal((raw_length, 16), dtype=np.float32)
        raw["time"] = np.arange(raw_length) / 2000
        for phase in samples:
            samples[phase].append((transform_sets[phase](raw.copy()), targets.clone()))
    loaders = {phase: DataLoader(items, batch_size=2, num_workers=0, collate_fn=WindowedEMGDataset.collate)
               for phase, items in samples.items()}
    probe_batch = next(iter(loaders["train"]))
    evidence["synthetic_input"] = {
        "raw_samples_per_item": raw_length, "batch_size": 2,
        "model_input_shape": list(probe_batch["inputs"].shape),
        "target_length_per_item": len(targets), "dataloader_workers": 0,
        "initialization": "seeded random weights; no released checkpoint",
    }

    class Capture(pl.Callback):
        def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx):
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            check(gradients and all(torch.isfinite(gradient).all() for gradient in gradients), "Nonfinite gradients")
            evidence["checks"]["finite_gradients"] = True
            evidence["training"]["gradient_norm"] = float(torch.sqrt(sum(gradient.square().sum() for gradient in gradients)))
            evidence["training"]["learning_rate_at_update"] = optimizer.param_groups[0]["lr"]

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            evidence["training"]["loss"] = float(loss.detach())
            check(math.isfinite(evidence["training"]["loss"]), "Nonfinite CTC loss")
            evidence["checks"]["finite_loss"] = True

    callbacks = [instantiate(item) for item in cfg.callbacks]
    capture = Capture()
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=150, max_steps=1,
        callbacks=[*callbacks, capture], default_root_dir=str(output),
        num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False,
        log_every_n_steps=1,
    )
    evidence["training"] = {}
    start = time.monotonic()
    trainer.fit(module, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    evidence["training"]["fit_wall_clock_seconds"] = time.monotonic() - start
    evidence["training"]["optimizer_steps"] = trainer.global_step
    check(trainer.global_step == 1, "Expected exactly one optimizer update")
    changed = int(torch.count_nonzero(module.model[4].weight.detach() != initial_head))
    check(changed > 0, "Optimizer update did not change head weights")
    evidence["training"]["changed_head_weight_elements"] = changed
    evidence["checks"]["weights_updated"] = True

    checkpoint_path = output / "synthetic-roundtrip.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))
    loaded = type(module).load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    check(all(torch.equal(value, loaded.state_dict()[name]) for name, value in module.state_dict().items()),
          "Reloaded model state differs")
    module.eval()
    loaded.eval()
    with torch.no_grad():
        difference = float((module(probe_batch["inputs"]) - loaded(probe_batch["inputs"])).abs().max())
    check(difference == 0, "Checkpoint changed evaluation output")
    state = torch.load(checkpoint_path, map_location="cpu")
    restored = loaded.configure_optimizers()
    restored["optimizer"].load_state_dict(state["optimizer_states"][0])
    restored["lr_scheduler"]["scheduler"].load_state_dict(state["lr_schedulers"][0])
    check(restored["optimizer"].param_groups[0]["lr"] == trainer.optimizers[0].param_groups[0]["lr"],
          "Optimizer LR did not restore")
    original_states = trainer.optimizers[0].state_dict()["state"]
    restored_states = restored["optimizer"].state_dict()["state"]
    check(original_states.keys() == restored_states.keys(), "Optimizer state keys differ")
    for index, original in original_states.items():
        for name, value in original.items():
            check(torch.equal(value, restored_states[index][name]) if torch.is_tensor(value)
                  else value == restored_states[index][name], "Adam state did not restore")
    check(restored["lr_scheduler"]["scheduler"].state_dict() == trainer.lr_scheduler_configs[0].scheduler.state_dict(),
          "Scheduler state did not restore")
    evidence["checkpoint_roundtrip"] = {
        "filename": checkpoint_path.name, "sha256": adapt.sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size, "global_step": state["global_step"],
        "model_state_equal": True, "evaluation_output_maximum_absolute_difference": difference,
        "adam_state_equal": True, "scheduler_state_equal": True,
        "best_checkpoint_exists": Path(trainer.checkpoint_callback.best_model_path).is_file(),
    }
    evidence["checks"]["checkpoint_roundtrip"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    check(not output.exists(), f"Refusing to reuse {output}")
    check(output.is_relative_to(adapt.PROJECT / "artifacts"), "Smoke outputs must be under ignored artifacts/")
    adapt.verify_upstream(args.upstream_dir.resolve())
    output.mkdir(parents=True)
    evidence = {
        "schema_version": 1, "milestone": "M5b", "phase": "synthetic-cpu-runtime-smoke",
        "upstream_commit": adapt.UPSTREAM_COMMIT, "seed": 1501,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {}, "acceptance": {"passed": False},
        "real_data_used": False, "released_checkpoint_used": False,
        "m5b_CER_gate_passed": None, "incremental_gpu_cost_usd": 0,
        "limitations": ["Synthetic CPU test does not establish CER reproduction or CUDA compatibility"],
    }
    start = time.monotonic()
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output / "environment.txt").write_text(freeze)
        evidence["environment_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
        exercise(args.upstream_dir.resolve(), output, evidence)
        evidence["acceptance"]["passed"] = all(evidence["checks"].values())
    except Exception as error:
        evidence["failure"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    finally:
        evidence["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        evidence["wall_clock_seconds"] = time.monotonic() - start
        (output / "result.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
