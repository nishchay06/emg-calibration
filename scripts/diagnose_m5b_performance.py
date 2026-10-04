#!/usr/bin/env python3
"""Bounded, disposable M5b timing probe; never trains an accepted checkpoint."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import adapt


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return {"unavailable": str(error)}


def cpu_snapshot():
    """Resolve the process cgroup using mount roots, including container namespaces."""
    result = {"affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "logical_cpus": os.cpu_count(), "pressure": read_optional("/proc/pressure/cpu"),
              "cgroup_membership": read_optional("/proc/self/cgroup"), "cgroups": []}
    membership = result["cgroup_membership"]
    mounts = read_optional("/proc/self/mountinfo")
    if not isinstance(membership, str) or not isinstance(mounts, str):
        return result
    for line in membership.splitlines():
        _, controllers, location = line.split(":", 2)
        for mount in mounts.splitlines():
            before, after = mount.split(" - ", 1)
            fields, tail = before.split(), after.split()
            kind = tail[0]
            if kind not in ("cgroup", "cgroup2"):
                continue
            if kind == "cgroup" and not set(controllers.split(",")).intersection(tail[2].split(",")):
                continue
            if kind == "cgroup2" and controllers:
                continue
            root, point = fields[3:5]
            relative = location[len(root):] if location.startswith(root) else location
            resolved = Path(point) / relative.lstrip("/")
            # Container cgroup namespaces may expose membership '/' at a mounted root.
            if not resolved.exists():
                resolved = Path(point)
            filenames = ("cpu.max", "cpu.stat", "cpuset.cpus.effective") if kind == "cgroup2" else (
                "cpu.cfs_quota_us", "cpu.cfs_period_us", "cpu.stat", "cpuset.cpus")
            result["cgroups"].append({"version": kind, "mount_root": root,
                "resolved_path": str(resolved), "files": {name: read_optional(resolved / name) for name in filenames}})
    return result


def summarize(rows):
    result = {}
    for key in rows[0]:
        values = sorted(row[key] for row in rows)
        result[key] = {"median": statistics.median(values),
                       "p90": values[math.ceil(0.9 * len(values)) - 1], "count": len(values)}
    return result


def throttling_delta(before, after):
    def values(snapshot):
        result = {}
        for group in snapshot["cgroups"]:
            raw = group["files"].get("cpu.stat")
            if isinstance(raw, str):
                result[group["resolved_path"]] = {key: int(value) for key, value in
                    (line.split() for line in raw.splitlines()) if value.isdigit()}
        return result
    left, right = values(before), values(after)
    return {path: {key: value - left[path][key] for key, value in counts.items() if key in left[path]}
            for path, counts in right.items() if path in left}


def validate_output(output):
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing existing output: {output}")
    try:
        relative = output.resolve().relative_to(adapt.PROJECT)
    except ValueError:
        relative = None
    if relative is not None and (not relative.parts or relative.parts[0] != "artifacts"):
        raise ValueError("Repository outputs must be under ignored artifacts/")


def supervise(command, log, deadline):
    remaining = deadline - time.time()
    if remaining <= 0 or remaining > 660:
        raise ValueError("Diagnostic deadline must be in the next 660 seconds")
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            code = 124
        finally:
            # Clean any loader/sampler descendants, even if the worker already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                return {"worker_exit_code": 125, "timed_out": timed_out,
                        "process_group": process.pid, "worker_reaped": False,
                        "error": "Worker did not reap after KILL; inspect process/GPU state",
                        "pod_billing_stopped": False}
    return {"worker_exit_code": code, "timed_out": timed_out,
            "process_group": process.pid, "worker_reaped": True,
            "pod_billing_stopped": False}


def worker(args):
    output = args.output_dir
    result = {"schema_version": 1, "milestone": "M5b", "phase": "performance-diagnostic",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "phases": {},
              "accuracy_gate_evaluated": False, "test_sessions_read": False,
              "checkpoints_written": False, "completed": False,
              "diagnostic_differences": ["manual bounded updates without Trainer/logging/checkpointing",
                  "synthetic batches cached in RAM; batch 2 is a historical smoke control",
                  "profiled pass synchronizes stages and is excluded from throughput estimates"]}
    sampler = None
    save(output / "timings.json", result)
    try:
        start = time.monotonic()
        import faulthandler
        faulthandler.dump_traceback_later(45)
        print("IMPORT_START", flush=True)
        import numpy as np
        print("NUMPY_IMPORTED", flush=True)
        import torch
        print("TORCH_IMPORTED", flush=True)
        import pytorch_lightning as pl
        print("LIGHTNING_IMPORTED", flush=True)
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from torch.utils.data import ConcatDataset, DataLoader
        from pytorch_lightning.utilities.seed import pl_worker_init_function
        sys.path.insert(0, str(args.upstream_dir))
        from emg2qwerty import transforms
        from emg2qwerty.data import LabelData, WindowedEMGDataset
        from emg2qwerty.lightning import TDSConvCTCModule, WindowedEMGDataModule
        faulthandler.cancel_dump_traceback_later()
        print("UPSTREAM_IMPORTED", flush=True)
        result["import_seconds"] = time.monotonic() - start
        device = "cpu" if args.cpu_smoke else "cuda"
        if not args.cpu_smoke:
            if sys.version_info[:2] != (3, 10) or not torch.cuda.is_available():
                raise ValueError("Expected Python 3.10 and CUDA")
            pins = {"torch": "2.3.0", "torchaudio": "2.3.0", "torchvision": "0.18.0",
                    "pytorch-lightning": "1.8.6", "lightning-bolts": "0.7.0",
                    "torchmetrics": "0.11.4", "hydra-core": "1.3.2", "omegaconf": "2.3.0"}
            pins.update({"numpy": "1.24.4", "setuptools": "69.5.1", "pip": "24.0"})
            for name, expected in pins.items():
                if version(name).split("+")[0] != expected:
                    raise ValueError(f"Package mismatch: {name}")
            if adapt.sha256(args.checkpoint) != adapt.GENERIC_SHA256:
                raise ValueError("Generic checkpoint digest mismatch")
            gpu_log = (output / "gpu-telemetry.csv").open("w")
            sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,pcie.link.gen.current,pcie.link.width.current", "--format=csv", "--loop-ms=1000"], stdout=gpu_log, stderr=subprocess.STDOUT)
        result["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
            "device": device, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        (output / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        configs = {}
        for user in ("user0", "user5"):
            plan = adapt.arguments(["--check-config", "--user", user, "--upstream-dir", str(args.upstream_dir),
                "--data-dir", str(args.data_dir), "--output-dir", str(output / (user + "-unused-plan")),
                "--checkpoint", str(args.checkpoint), "--accelerator", "gpu"])
            configs[user] = adapt.compose_config(plan)
        save(output / "resolved-configs.json", configs)

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        def load_module(cfg):
            start = time.monotonic()
            if args.cpu_smoke:
                module = instantiate(cfg.module, optimizer=cfg.optimizer, lr_scheduler=cfg.lr_scheduler,
                                     decoder=cfg.decoder, _recursive_=False)
            else:
                module = TDSConvCTCModule.load_from_checkpoint(str(args.checkpoint), map_location="cpu",
                    optimizer=cfg.optimizer, lr_scheduler=cfg.lr_scheduler, decoder=cfg.decoder)
            module.to(device)
            module.log = lambda *a, **kw: None
            optimizer = module.configure_optimizers()["optimizer"]
            if optimizer.param_groups[0]["lr"] != 1e-8:
                raise ValueError("Unexpected initial learning rate")
            sync()
            return module, optimizer, time.monotonic() - start

        def iteration(module, optimizer, batch, phase, profile=False, check_gradients=False):
            timings = {}
            originals = []
            original_cpu = torch.Tensor.cpu
            def measure(name, function):
                def wrapped(*a, **kw):
                    sync()
                    begin = time.monotonic()
                    value = function(*a, **kw)
                    sync()
                    timings[name] = time.monotonic() - begin
                    return value
                return wrapped
            if profile:
                def timed_cpu(tensor, *a, **kw):
                    sync(); begin = time.monotonic()
                    value = original_cpu(tensor, *a, **kw)
                    sync()
                    timings["device_host_copy_s"] = timings.get("device_host_copy_s", 0) + time.monotonic() - begin
                    return value
                torch.Tensor.cpu = timed_cpu
                for owner, attribute, name in ((module, "forward", "forward_s"),
                        (module.ctc_loss, "forward", "ctc_s"), (module.decoder, "decode_batch", "decode_s"),
                        (module.metrics[f"{phase}_metrics"], "update", "metrics_s")):
                    original = getattr(owner, attribute)
                    originals.append((owner, attribute, original))
                    # Metrics is called once per item: accumulate instead of overwriting.
                    def wrapped(*a, fn=original, key=name, **kw):
                        sync(); begin = time.monotonic()
                        value = fn(*a, **kw)
                        sync(); timings[key] = timings.get(key, 0) + time.monotonic() - begin
                        return value
                    setattr(owner, attribute, wrapped)
            module.train(phase == "train")
            optimizer.zero_grad()
            sync()
            begin = time.monotonic()
            events = None
            if device == "cuda":
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
            copy_start = time.monotonic()
            gpu_batch = {name: value.to(device) for name, value in batch.items()}
            if profile:
                sync(); timings["host_device_copy_s"] = time.monotonic() - copy_start
            try:
                with torch.set_grad_enabled(phase == "train"):
                    loss = module._step(phase, gpu_batch)
                    torch.Tensor.cpu = original_cpu
                    if phase == "train":
                        if profile:
                            measure("backward_s", loss.backward)()
                            measure("optimizer_s", optimizer.step)()
                        else:
                            loss.backward()
                            optimizer.step()
                if events:
                    events[1].record()
                sync()
                timings["iteration_wall_s"] = time.monotonic() - begin
                if events:
                    timings["cuda_elapsed_s"] = events[0].elapsed_time(events[1]) / 1000
                # Health checks are outside measured whole-iteration timing.
                if not math.isfinite(float(loss.detach().cpu())):
                    raise ValueError("Nonfinite CTC loss")
                if check_gradients and phase == "train" and not all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None):
                    raise ValueError("Nonfinite gradient")
            finally:
                torch.Tensor.cpu = original_cpu
                for owner, attribute, original in originals:
                    setattr(owner, attribute, original)
            return timings

        def phase_run(name, cfg, loader, phase, warmup, count, profile_count):
            pl.seed_everything(1501, workers=True)
            before = cpu_snapshot()
            module, optimizer, initialization = load_module(cfg)
            initial_head = module.model[4].weight.detach().clone()
            loader_start = time.monotonic()
            iterator = iter(loader)
            iterator_startup = time.monotonic() - loader_start
            rows, profile_rows, first = [], [], None
            for index in range(warmup + count + profile_count):
                begin = time.monotonic()
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                wait = time.monotonic() - begin
                row = iteration(module, optimizer, batch, phase, index >= warmup + count,
                                index in (0, warmup + count + profile_count - 1))
                row["loader_wait_s"] = wait
                if index == 0:
                    first = row
                if warmup <= index < warmup + count:
                    rows.append(row)
                elif index >= warmup + count:
                    profile_rows.append(row)
            changed = int(torch.count_nonzero(module.model[4].weight.detach() != initial_head))
            if phase == "train" and changed == 0:
                raise ValueError("No head weights changed")
            after = cpu_snapshot()
            result["phases"][name] = {"initialization_s": initialization,
                "iterator_startup_s": iterator_startup, "first_iteration": first,
                "unprofiled": summarize(rows), "profiled": summarize(profile_rows) if profile_rows else {},
                "measurements": rows, "profile_measurements": profile_rows,
                "changed_head_elements": changed, "cpu_before": before, "cpu_after": after,
                "cpu_stat_deltas": throttling_delta(before, after),
                "profile_stage_times_may_overlap": True}
            save(output / "timings.json", result)
            print("PHASE_DONE " + name, flush=True)
            if getattr(loader, "_iterator", None) is not None:
                loader._iterator._shutdown_workers()
            del iterator, module, optimizer, initial_head
            if device == "cuda":
                torch.cuda.empty_cache()

        cfg = OmegaConf.create(configs["user0"])
        for size in ((2,) if args.cpu_smoke else (2, 32)):
            pl.seed_everything(1501, workers=True)
            transform = transforms.Compose([instantiate(item) for item in cfg.transforms.train])
            rng = np.random.default_rng(1501)
            items = []
            transform_start = time.monotonic()
            for _ in range(size):
                raw = np.zeros(10000, dtype=[("emg_left", np.float32, (16,)),
                    ("emg_right", np.float32, (16,)), ("time", np.float64)])
                raw["emg_left"] = rng.standard_normal((10000, 16), dtype=np.float32)
                raw["emg_right"] = rng.standard_normal((10000, 16), dtype=np.float32)
                raw["time"] = np.arange(10000) / 2000
                items.append((transform(raw), torch.tensor(LabelData.from_str("emgtest").labels)))
            result.setdefault("synthetic_preparation_s", {})[str(size)] = time.monotonic() - transform_start
            batch = WindowedEMGDataset.collate(items)
            phase_run(f"synthetic_batch{size}", cfg, [batch], "train",
                      0 if args.cpu_smoke else 5, 1 if args.cpu_smoke else 20, 1 if args.cpu_smoke else 3)
        if not args.cpu_smoke:
            for user in ("user0", "user5"):
                cfg = OmegaConf.create(configs[user])
                pl.seed_everything(1501, workers=True)
                build = lambda configs: transforms.Compose([instantiate(item) for item in configs])
                # setup() also constructs test datasets, so build only train/val here.
                datamodule = WindowedEMGDataModule(window_length=8000, padding=(1800, 200),
                    batch_size=32, num_workers=4, train_sessions=[], val_sessions=[], test_sessions=[],
                    train_transform=build(cfg.transforms.train), val_transform=build(cfg.transforms.val),
                    test_transform=None)
                for phase in ("train", "val"):
                    dataset = ConcatDataset([WindowedEMGDataset(args.data_dir / f"{item.session}.hdf5",
                        transform=getattr(datamodule, phase + "_transform"), window_length=8000,
                        padding=(1800, 200), jitter=phase == "train") for item in cfg.dataset[phase]])
                    setattr(datamodule, phase + "_dataset", dataset)
                    loader = getattr(datamodule, phase + "_dataloader")()
                    loader.worker_init_fn = functools.partial(pl_worker_init_function, rank=0)
                    phase_run(user + "_" + phase, cfg, loader, phase, 3, 10, 3)
        result["completed"] = True
        result["cpu_smoke_only"] = args.cpu_smoke
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        if sampler:
            sampler.terminate()
            try:
                sampler.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sampler.kill(); sampler.wait()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "timings.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--cpu-smoke", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("upstream_dir", "data_dir", "checkpoint", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    if args.worker:
        worker(args)
        return
    validate_output(args.output_dir)
    if args.dry_run:
        print(json.dumps({"phase": "performance-diagnostic", "mutation": False,
            "users": ["user0", "user5"], "test_sessions_read": False,
            "full_training": False, "upstream_commit": adapt.UPSTREAM_COMMIT,
            "output": str(args.output_dir), "worker_seconds_max": 660}, indent=2))
        return
    adapt.verify_upstream(args.upstream_dir)
    if args.deadline is None or not 0 < args.deadline - time.time() <= 660:
        raise ValueError("Supply an absolute worker deadline within 660 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
        "--upstream-dir", str(args.upstream_dir), "--data-dir", str(args.data_dir),
        "--checkpoint", str(args.checkpoint), "--output-dir", str(args.output_dir)]
    if args.cpu_smoke:
        command.append("--cpu-smoke")
    state = supervise(command, args.output_dir / "console.log", args.deadline)
    save(args.output_dir / "supervisor.json", state)
    sys.exit(state["worker_exit_code"])


if __name__ == "__main__":
    main()
