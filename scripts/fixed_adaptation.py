"""Fixed-update adaptation using upstream model/transforms, never best selection."""
from __future__ import annotations

import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from calibration_protocol import load_protocol
from calibration_sampler import digest, select_calibration, verify_selection, window_accounting
from generate_test_user_manifests import UPSTREAM_COMMIT, parse_user_config


def verify_m5b_gate():
    import adapt

    adapt.verify_user0_gate(adapt.PROJECT / "results/m5b-user0-full-upstream.json")
    result = json.loads((adapt.PROJECT / "results/m5b-user5-full-upstream.json").read_text())
    if (result.get("milestone"), result.get("user"), result.get("budget_minutes"), result.get("method"),
            result.get("selection"), result.get("upstream_commit"), result.get("training_executed"),
            result.get("process_exit_code"), result.get("last_checkpoint", {}).get("epoch")) != (
            "M5b", "user5", "full", "full", "upstream", UPSTREAM_COMMIT, True, 0, 149):
        raise ValueError("M5b user5 reproduction is incomplete")
    if (not result.get("acceptance", {}).get("passed") or
            not result.get("trained_checkpoint", {}).get("sha256") or
            not adapt.test_acceptance(result["metrics"]["test"]["CER"], 5.811)["passed"]):
        raise ValueError("M5b user5 accuracy/provenance gate failed")


def indexed_lengths(index, manifest):
    if index.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Session-index upstream mismatch")
    content = {k: v for k, v in index.items() if k != "index_digest"}
    if index.get("index_digest") != digest(content):
        raise ValueError("Session-index digest mismatch")
    lengths = {}
    for split, names in manifest.splits.items():
        for name in names:
            item = index.get("sessions", {}).get(name, {})
            if (item.get("user"), item.get("split")) != (manifest.user, split):
                raise ValueError("Session index differs from official splits")
            lengths[name] = item.get("samples")
    return lengths


def plan_fixed(args):
    import adapt

    verify_m5b_gate()
    protocol, profile = load_protocol(args.protocol, require_frozen=args.run and not args.tune)
    if args.steps is not None and args.steps != profile["steps"]:
        raise ValueError("Step override differs from the shared method profile")
    selection = None
    if args.session_index:
        manifest = parse_user_config(args.upstream_dir / "config/user" / f"{args.user}.yaml", args.user)
        index = json.loads(args.session_index.read_text())
        selection = select_calibration(manifest, indexed_lengths(index, manifest), args.budget_minutes, args.seed)
    return {"schema_version": 1, "pipeline": "fixed-step-calibration", "implemented_in": "M6",
            "phase": "tuning" if args.tune else "adaptation",
            "user": args.user, "budget_minutes": args.budget_minutes, "method": args.method,
            "selection": "fixed", "seed": args.seed, "upstream_commit": UPSTREAM_COMMIT,
            "protocol_digest": digest(protocol), "protocol_status": protocol["status"],
            "profile": profile, "profile_digest": digest(profile),
            "calibration_selection": selection,
            "data_preflight": "pending" if selection is None else "index supplied; mounted checks still required",
            "initial_checkpoint": {"sha256": adapt.GENERIC_SHA256},
            "training_executed": False, "test_evaluated": False, "completed": False,
            "resume": "disabled; fresh output required", "cost": None}


def compose_fixed(args, record):
    import adapt

    base = copy.copy(args)
    base.select, base.budget_minutes, base.steps = "upstream", "full", None
    config = adapt.compose_config(base)
    profile = record["profile"]
    config["optimizer"] = {"_target_": "torch.optim.Adam", "lr": profile["learning_rate"]}
    config["lr_scheduler"] = {"scheduler": {"_target_": "step_schedule.UpdateWarmupCosine", "profile": profile},
                              "interval": "step", "frequency": 1}
    config["trainer"].update(max_epochs=-1, max_steps=profile["steps"], limit_val_batches=0,
                            num_sanity_val_steps=0, accumulate_grad_batches=1)
    config["callbacks"] = []
    config["checkpoint_selection"] = "explicit final checkpoint after exactly max_steps"
    return config


def make_datamodule(args, record, config, *, num_workers=4):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from calibration_data import CalibrationDataModule
    from emg2qwerty.transforms import Compose

    cfg = OmegaConf.create(config)
    transforms = {phase: Compose([instantiate(item) for item in cfg.transforms[phase]])
                  for phase in ("train", "val", "test")}
    return CalibrationDataModule(
        data_dir=args.data_dir, selection=record["calibration_selection"], allow_validation=args.tune,
        window_length=8000, padding=(1800, 200), batch_size=32, num_workers=num_workers,
        **{f"{split}_sessions": [args.data_dir / f"{item['session']}.hdf5" for item in config["dataset"][split]]
           for split in ("train", "val", "test")},
        **{f"{phase}_transform": transform for phase, transform in transforms.items()})


def fit_fixed(module, datamodule, profile, output, *, accelerator="cpu", stats=None):
    """Shared CPU-smoke/production fitting path; returns final checkpoint proof."""
    import pytorch_lightning as pl
    import torch
    import adapt

    if stats is None:
        stats = {}
    stats.update(learning_rates_at_updates=[], batch_sizes=[], finite_losses=True,
                 finite_gradients=True, validation_batches_during_fit=0, optimizer_steps=0)

    class Capture(pl.Callback):
        def on_before_backward(self, trainer, model, loss):
            if not torch.isfinite(loss).item():
                stats["finite_losses"] = False
                raise ValueError("Nonfinite calibration loss")

        def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx):
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                stats["finite_gradients"] = False
                raise ValueError("Missing or nonfinite calibration gradients")
            stats["learning_rates_at_updates"].append(optimizer.param_groups[0]["lr"])

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            stats["batch_sizes"].append(len(batch["input_lengths"]))
            stats["optimizer_steps"] = trainer.global_step

        def on_validation_batch_start(self, *unused):
            stats["validation_batches_during_fit"] += 1
            raise ValueError("Unexpected validation during fixed adaptation")

    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_steps=profile["steps"], max_epochs=-1,
                         accumulate_grad_batches=1, limit_val_batches=0, num_sanity_val_steps=0,
                         enable_checkpointing=False, logger=False, enable_progress_bar=False,
                         enable_model_summary=False, default_root_dir=str(output), callbacks=[Capture()])
    fit_started = time.monotonic()
    trainer.fit(module, datamodule=datamodule, ckpt_path=None)
    stats["fit_wall_clock_seconds"] = time.monotonic() - fit_started
    if trainer.global_step != profile["steps"] or len(stats["learning_rates_at_updates"]) != profile["steps"]:
        raise ValueError("Fixed optimizer-step count mismatch")
    if not all(torch.isfinite(value).all().item() for value in module.state_dict().values()):
        raise ValueError("Nonfinite final model state")
    checkpoint_dir = Path(output) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    final = checkpoint_dir / "final.ckpt"
    trainer.save_checkpoint(str(final))
    state = torch.load(final, map_location="cpu")
    if state["global_step"] != profile["steps"]:
        raise ValueError("Saved final checkpoint step mismatch")
    stats.update(optimizer_steps=trainer.global_step,
                 trained_checkpoint={"filename": final.name, "sha256": adapt.sha256(final),
                                     "size_bytes": final.stat().st_size, "epoch": state["epoch"],
                                     "global_step": state["global_step"]})
    return trainer, final, stats


def run_fixed(args, record, config):
    import adapt
    from index_calibration_sessions import index_sessions

    if sys.version_info[:2] != (3, 10):
        raise ValueError("Fixed training requires audited Python 3.10")
    if adapt.sha256(args.checkpoint) != adapt.GENERIC_SHA256:
        raise ValueError("Generic initialization SHA-256 mismatch")
    manifest = parse_user_config(args.upstream_dir / "config/user" / f"{args.user}.yaml", args.user)
    index = index_sessions(args.upstream_dir, args.data_dir)
    lengths = indexed_lengths(index, manifest)
    selection = select_calibration(manifest, lengths, args.budget_minutes, args.seed)
    if record["calibration_selection"] is not None:
        verify_selection(record["calibration_selection"], manifest, lengths)
    record.update(calibration_selection=selection, data_preflight="passed", session_index_digest=index["index_digest"],
                  window_accounting=window_accounting(selection))
    if record["window_accounting"]["training_windows"] == 0:
        raise ValueError("Calibration allocation has no complete windows")
    relative = None
    try:
        relative = args.output_dir.resolve().relative_to(adapt.PROJECT)
    except ValueError:
        pass
    if relative is not None and (not relative.parts or relative.parts[0] != "artifacts"):
        raise ValueError("Output inside the repository must be under ignored artifacts/")
    from importlib.metadata import version
    pins = {"torch": "2.3.0", "torchaudio": "2.3.0", "torchvision": "0.18.0", "pytorch-lightning": "1.8.6",
            "h5py": "3.11.0", "numpy": "1.24.4", "hydra-core": "1.3.2", "omegaconf": "2.3.0",
            "torchmetrics": "0.11.4", "setuptools": "69.5.1", "pip": "24.0"}
    for package, expected in pins.items():
        if version(package).split("+")[0] != expected:
            raise ValueError(f"Unexpected {package} version")
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf

    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise ValueError("GPU requested but CUDA is unavailable")
    sys.path.insert(0, str(args.upstream_dir))
    from emg2qwerty.lightning import TDSConvCTCModule

    pl.seed_everything(args.seed, workers=True)
    module = TDSConvCTCModule.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", optimizer=OmegaConf.create(config["optimizer"]),
        lr_scheduler=OmegaConf.create(config["lr_scheduler"]), decoder=OmegaConf.create(config["decoder"]))
    datamodule = make_datamodule(args, record, config)
    datamodule.setup("fit")  # Validate allocation before creating output.
    record["parameters"] = {"total": sum(p.numel() for p in module.parameters()),
                             "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
    record["runtime"] = {"python": platform.python_version(), "packages": pins, "accelerator": args.accelerator}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output_dir / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    record["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    stack = ExitStack()
    console = stack.enter_context((args.output_dir / "console.log").open("w"))

    class Tee:
        def __init__(self, original):
            self.original = original

        def write(self, text):
            console.write(text)
            console.flush()
            return self.original.write(text)

        def flush(self):
            console.flush()
            self.original.flush()

        def __getattr__(self, name):
            return getattr(self.original, name)

    stack.enter_context(redirect_stdout(Tee(sys.stdout)))
    stack.enter_context(redirect_stderr(Tee(sys.stderr)))
    record["fit_diagnostics"] = {}
    try:
        record["training_executed"] = True
        trainer, final, stats = fit_fixed(module, datamodule, record["profile"], args.output_dir,
                                          accelerator=args.accelerator, stats=record["fit_diagnostics"])
        record.update(stats, training_executed=True)
        reloaded = TDSConvCTCModule.load_from_checkpoint(str(final), map_location="cpu")
        if not all(torch.equal(value.cpu(), reloaded.state_dict()[key]) for key, value in module.state_dict().items()):
            raise ValueError("Final checkpoint reload differs from fitted model")
        record.update(evaluate_final(trainer, reloaded, datamodule,
                                     tune=args.tune, accelerator=args.accelerator))
        record.update(completed=True, process_exit_code=0,
                      evaluation_checkpoint_sha256=adapt.sha256(final))
    except BaseException as error:
        record.update(failure=str(error), process_exit_code=1, interrupted=not isinstance(error, Exception))
        traceback.print_exc()
        if not isinstance(error, Exception):
            raise
    finally:
        stack.close()
        record.update(wall_clock_seconds=time.monotonic() - start,
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        (args.output_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["completed"] else 1


def evaluate_final(trainer, reloaded, datamodule, *, tune, accelerator):
    if tune:
        import pytorch_lightning as pl
        # Capture forbids validation during fitting; use a fresh evaluator.
        evaluator = pl.Trainer(accelerator=accelerator, devices=1, logger=False,
                               enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
        metrics = evaluator.validate(reloaded, datamodule=datamodule)[0]
        return {"metrics": {"validation": normalized_metrics(metrics, "val")}, "test_evaluated": False}
    metrics = trainer.test(reloaded, datamodule=datamodule)[0]
    return {"metrics": {"test": normalized_metrics(metrics, "test")}, "test_evaluated": True}


def normalized_metrics(metrics, split):
    output = {}
    for name in ("loss", "CER", "IER", "DER", "SER"):
        value = float(metrics[f"{split}/{name}"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid final-checkpoint metrics")
        output[name] = value
    return output
