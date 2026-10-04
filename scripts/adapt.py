#!/usr/bin/env python3
"""Plan, check, or run the M5b full-data upstream personalization recipe.

Budget sampling and parameter-efficient methods are deliberately unavailable
until their roadmap gates. A dry run needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from capture_generic_result import parse_metrics
from generate_test_user_manifests import UPSTREAM_COMMIT, parse_user_config


PROJECT = Path(__file__).resolve().parent.parent
GENERIC_SHA256 = "338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest"
        ) else _digest_stream(stream)


def _digest_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def hydra_string(value: Path) -> str:
    # JSON double-quoted strings are valid Hydra quoted values. Disallow OmegaConf
    # interpolation, which is evaluated even inside quoted strings.
    if "${" in str(value):
        raise ValueError("Paths must not contain OmegaConf interpolation")
    return json.dumps(str(value))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, choices=[f"user{i}" for i in range(8)])
    parser.add_argument("--budget-minutes", choices=["1", "2", "5", "10", "30", "60", "full"], default="full")
    parser.add_argument("--method", choices=["full", "head", "norm", "lora"], default="full")
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--select", choices=["upstream", "fixed"], default="upstream")
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--user0-result", type=Path, help="Required passed M5b evidence before running user5")
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("seed must be in [0, 2**32 - 1]")
    if args.steps is not None and args.steps <= 0:
        parser.error("steps must be positive")
    if args.select == "upstream" and args.budget_minutes != "full":
        parser.error("upstream validation selection is allowed only for full data")
    if (args.budget_minutes, args.method, args.select) != ("full", "full", "upstream"):
        parser.error("M5b supports full/full/upstream only; M6/M8 are gated")
    if args.steps is not None:
        parser.error("M5b uses 150 epochs; a step override changes the reproduction")
    if args.run and args.user not in ("user0", "user5"):
        parser.error("The M5b paid gate runs user0, then user5 only")
    if args.run and args.user == "user5" and args.user0_result is None:
        parser.error("user5 requires --user0-result from a passed M5b training run")
    for name in ("upstream_dir", "data_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    args.checkpoint = (args.checkpoint or args.upstream_dir / "models/generic.ckpt").expanduser().absolute()
    for path in (args.upstream_dir, args.data_dir, args.output_dir, args.checkpoint):
        if "${" in str(path):
            parser.error("paths must not contain OmegaConf interpolation")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Refusing to reuse output path: {args.output_dir}")
    return args


def overrides(args):
    return [
        f"user={args.user}", "train=True", "decoder=ctc_greedy",
        f"checkpoint={hydra_string(args.checkpoint)}",
        f"dataset.root={hydra_string(args.data_dir)}", f"seed={args.seed}",
        "batch_size=32", "num_workers=4", "trainer.max_epochs=150",
        f"trainer.accelerator={args.accelerator}", "trainer.devices=1",
        f"hydra.run.dir={hydra_string(args.output_dir)}", "hydra.job.chdir=True",
    ]


def plan(args):
    reference = json.loads((PROJECT / "references/personalized-greedy.json").read_text())
    return {
        "milestone": "M5b", "user": args.user, "budget_minutes": "full",
        "method": "full", "selection": "upstream", "seed": args.seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "initial_checkpoint": {"path": str(args.checkpoint), "sha256": GENERIC_SHA256},
        "resume": "disabled; fresh output required",
        "command": [sys.executable, "-m", "emg2qwerty.train", *overrides(args)],
        "cwd": str(args.upstream_dir), "output_directory": str(args.output_dir),
        "acceptance": {"test_CER_reference": reference["benchmarks"]["finetuned"]["users"][args.user]["test_CER"],
                       "maximum_absolute_delta_percentage_points": 1.0,
                       "validation_is_diagnostic_only": True},
        "cost": None, "training_executed": False,
    }


def verify_upstream(path):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}; found {actual}")
    # Permit exactly the already-audited optional-KenLM compatibility patch.
    changed = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--name-only"], text=True).splitlines()
    if changed:
        if changed != ["emg2qwerty/decoder.py"]:
            raise ValueError(f"Unexpected upstream changes: {changed}")
        patch = PROJECT / "patches/greedy-decoder-optional-kenlm.patch"
        expected = patch.read_text()
        actual_diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--", changed[0]], text=True)
        def changes(diff):
            return [line for line in diff.splitlines() if line.startswith(("+", "-"))
                    and not line.startswith(("+++", "---"))]
        if changes(actual_diff) != changes(expected):
            raise ValueError("Decoder differs from the approved optional-KenLM patch")
        subprocess.run(["git", "-C", str(path), "apply", "--reverse", "--check", str(patch)], check=True)


def compose_config(args):
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    verify_upstream(args.upstream_dir)
    with initialize_config_dir(version_base=None, config_dir=str(args.upstream_dir / "config")):
        config = compose(config_name="base", overrides=overrides(args), return_hydra_config=True)
        # Compose API does not create a job or resolve runtime output_dir itself.
        config.hydra.runtime.output_dir = str(args.output_dir)
        HydraConfig.instance().set_config(config)
        try:
            job = OmegaConf.to_container(config, resolve=False)
            del job["hydra"]
            resolved = OmegaConf.to_container(OmegaConf.create(job), resolve=True)
        finally:
            HydraConfig.instance().cfg = None
    validate_config(resolved, args)
    return resolved


def validate_config(config, args):
    scheduler = config["lr_scheduler"]
    if config["optimizer"] != {"_target_": "torch.optim.Adam", "lr": 0.001}:
        raise ValueError("Optimizer differs from the pinned Adam recipe")
    if scheduler != {"scheduler": {"_target_": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
                                   "warmup_epochs": 10, "max_epochs": 150,
                                   "warmup_start_lr": 1e-8, "eta_min": 1e-6}, "interval": "epoch"}:
        raise ValueError("Scheduler differs from the pinned epoch-based recipe")
    datamodule = config["datamodule"]
    if datamodule["window_length"] != 8000 or datamodule["padding"] != [1800, 200]:
        raise ValueError("Window/padding differs from upstream")
    checkpoint = config["callbacks"][1]
    if (checkpoint["monitor"], checkpoint["mode"], checkpoint["save_last"], checkpoint["dirpath"]) != (
            "val/CER", "min", True, str(args.output_dir / "checkpoints")):
        raise ValueError("Checkpoint selection/output differs from upstream")
    manifest = parse_user_config(args.upstream_dir / "config/user" / f"{args.user}.yaml", args.user)
    committed = set((PROJECT / "manifests" / f"{args.user}-sessions.txt").read_text().splitlines())
    expected = {f"emg2qwerty-data-2021-08/{session}.hdf5" for session in manifest.sessions}
    if committed != expected:
        raise ValueError("Committed manifest differs from the pinned user splits")
    for split, sessions in manifest.splits.items():
        if [entry["session"] for entry in config["dataset"][split]] != list(sessions):
            raise ValueError(f"Composed {split} differs from upstream")


def run(args, record, config):
    if args.user == "user5":
        verify_user0_gate(args.user0_result)
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Training requires the audited Python 3.10 environment")
    if sha256(args.checkpoint) != GENERIC_SHA256:
        raise ValueError("Generic initialization SHA-256 mismatch")
    for split in ("train", "val", "test"):
        for item in config["dataset"][split]:
            path = args.data_dir / f"{item['session']}.hdf5"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty session: {path}")
    # Keep all model artifacts out of the public repository by construction.
    try:
        relative = args.output_dir.resolve().relative_to(PROJECT)
    except ValueError:
        relative = None
    if relative is not None and (not relative.parts or relative.parts[0] != "artifacts"):
        raise ValueError("Training output inside this repo must be under ignored artifacts/")

    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(args.upstream_dir))
    from emg2qwerty.lightning import TDSConvCTCModule
    # This CPU probe checks constructor, optimizer, and scheduler availability
    # before creating output. Actual training seeds and loads via upstream.
    module = TDSConvCTCModule.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu",
        optimizer=OmegaConf.create(config["optimizer"]),
        lr_scheduler=OmegaConf.create(config["lr_scheduler"]),
        decoder=OmegaConf.create(config["decoder"]),
    )
    module.configure_optimizers()
    record["parameters"] = {"total": sum(p.numel() for p in module.parameters()),
                             "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
    del module
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise ValueError("GPU requested but CUDA is unavailable")
    from importlib.metadata import version
    expected_versions = {"pytorch-lightning": "1.8.6", "lightning-bolts": "0.7.0",
                         "hydra-core": "1.3.2", "omegaconf": "2.3.0", "torchmetrics": "0.11.4"}
    for package, expected in expected_versions.items():
        if version(package) != expected:
            raise ValueError(f"{package} must be {expected}")
    if torch.__version__.split("+")[0] != "2.3.0":
        raise ValueError("PyTorch must be 2.3.0")
    record["runtime"] = {"python": platform.python_version(), "pytorch": torch.__version__,
                         "packages": expected_versions,
                         "gpu": torch.cuda.get_device_name(0) if args.accelerator == "gpu" else None}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output_dir / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output_dir / "environment.txt").write_text(freeze)
    record["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    env = dict(os.environ, PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
    with (args.output_dir / "console.log").open("w") as log:
        process = subprocess.Popen(record["command"], cwd=args.upstream_dir, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    record.update(training_executed=True, process_exit_code=code,
                  wall_clock_seconds=time.monotonic() - start,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    record["acceptance"]["passed"] = False
    try:
        if code:
            raise ValueError(f"Upstream training exited {code}")
        capture_training_result(record, args.output_dir, lambda path: torch.load(path, map_location="cpu"))
    except Exception as error:
        record["failure"] = str(error)
    finally:
        (args.output_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0 if record["acceptance"]["passed"] else 1


def capture_training_result(record, output_dir, load_checkpoint):
    """Require completed metrics, best-checkpoint identity, and 150 epochs."""
    text = (output_dir / "console.log").read_text()
    if "Resuming training from checkpoint" in text:
        raise ValueError("Unexpected training resume")
    metrics = parse_metrics(text)
    matches = re.findall(r"['\"]best_checkpoint['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    if not matches:
        raise ValueError("Missing final best-checkpoint provenance")
    best = Path(matches[-1]).resolve()
    best.relative_to((output_dir / "checkpoints").resolve())
    state = load_checkpoint(best)
    last = load_checkpoint(output_dir / "checkpoints/last.ckpt")
    if last["epoch"] != 149:
        raise ValueError("Training did not complete all 150 epochs")
    record["trained_checkpoint"] = {"filename": best.name, "sha256": sha256(best),
                                    "size_bytes": best.stat().st_size,
                                    "epoch": state["epoch"], "global_step": state["global_step"]}
    record["last_checkpoint"] = {"epoch": last["epoch"], "global_step": last["global_step"]}
    record["metrics"] = {"validation": metrics["val"], "test": metrics["test"]}
    record["acceptance"].update(test_acceptance(metrics["test"]["CER"], record["acceptance"]["test_CER_reference"]))


def test_acceptance(test_cer, reference_cer):
    if not math.isfinite(test_cer) or test_cer < 0:
        raise ValueError("Invalid test CER")
    delta = test_cer - reference_cer
    return {"test_CER_delta_percentage_points": delta,
            "passed": abs(delta) <= 1.0 or math.isclose(abs(delta), 1.0, rel_tol=0, abs_tol=1e-12)}


def verify_user0_gate(path):
    evidence = json.loads(path.read_text())
    if (evidence.get("milestone"), evidence.get("user"), evidence.get("budget_minutes"),
            evidence.get("method"), evidence.get("selection"), evidence.get("upstream_commit")) != (
            "M5b", "user0", "full", "full", "upstream", UPSTREAM_COMMIT):
        raise ValueError("user0 gate requires M5b full/full/upstream training evidence")
    if not evidence.get("training_executed") or evidence.get("process_exit_code") != 0:
        raise ValueError("user0 training did not complete successfully")
    if not evidence.get("acceptance", {}).get("passed"):
        raise ValueError("user0 acceptance failed")
    if evidence.get("last_checkpoint", {}).get("epoch") != 149:
        raise ValueError("user0 did not complete 150 epochs")
    if not evidence.get("trained_checkpoint", {}).get("sha256"):
        raise ValueError("user0 trained-checkpoint provenance missing")
    if not test_acceptance(evidence["metrics"]["test"]["CER"], 20.57)["passed"]:
        raise ValueError("user0 test CER fails the gate")


def main(argv=None):
    args = arguments(argv)
    try:
        record = plan(args)
        if args.dry_run:
            print(json.dumps(record, indent=2))
            return 0
        config = compose_config(args)
        if args.check_config:
            record["resolved_configuration"] = config
            record["configuration_checked"] = True
            print(json.dumps(record, indent=2))
            return 0
        return run(args, record, config)
    except (ValueError, OSError, subprocess.CalledProcessError, ImportError) as error:
        print(f"Adaptation preflight failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
