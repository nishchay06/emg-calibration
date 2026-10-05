#!/usr/bin/env python3
"""Sequential approved M7 tuning; no test evaluation or implicit resume."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from calibration_protocol import freeze_protocol, validate_protocol
from calibration_sampler import digest
from generate_test_user_manifests import UPSTREAM_COMMIT

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def choose_candidate(matrix, protocols, results):
    """Require the complete declared grid; never consult a test metric."""
    expected = {run["id"]: run for run in matrix["tuning_runs"]}
    if len(expected) != 16 or set(results) != set(expected):
        raise ValueError("Require all 16 unique declared tuning runs")
    groups = {}
    allocations = {}
    for run_id, run in expected.items():
        record = results[run_id]
        profile = protocols[run["candidate"]]["methods"]["full"]
        if (record.get("phase"), record.get("user"), record.get("budget_minutes"),
                record.get("method"), record.get("seed"), record.get("selection"),
                record.get("completed"), record.get("test_evaluated")) != (
                "tuning", run["user"], run["budget_minutes"], "full", 1501,
                "fixed", True, False):
            raise ValueError("Tuning receipt identity or evaluation mismatch")
        if "test" in record.get("metrics", {}):
            raise ValueError("Test metrics cannot enter candidate selection")
        if (record.get("upstream_commit") != UPSTREAM_COMMIT or
                record.get("initial_checkpoint", {}).get("sha256") !=
                "338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81" or
                record.get("profile_digest") != digest(profile) or
                record.get("optimizer_steps") != profile["steps"] or
                record.get("validation_batches_during_fit") != 0 or
                not record.get("finite_losses") or not record.get("finite_gradients") or
                record.get("data_preflight") != "passed"):
            raise ValueError("Incomplete fixed-protocol training proof")
        checkpoint_sha = record.get("trained_checkpoint", {}).get("sha256")
        if not checkpoint_sha or checkpoint_sha != record.get("evaluation_checkpoint_sha256"):
            raise ValueError("Evaluation must use the saved final checkpoint")
        cer = record["metrics"]["validation"]["CER"]
        if isinstance(cer, bool) or not isinstance(cer, (int, float)) or not math.isfinite(cer) or cer < 0:
            raise ValueError("Invalid validation CER")
        allocation_key = (run["user"], run["budget_minutes"])
        allocation_digest = digest(record["calibration_selection"])
        if allocations.setdefault(allocation_key, allocation_digest) != allocation_digest:
            raise ValueError("Candidates used different calibration allocations")
        groups.setdefault(run["candidate"], []).append((run_id, cer))
    if len(groups) != 4 or any(len(items) != 4 for items in groups.values()):
        raise ValueError("Require four matched user/budget receipts per candidate")
    scores = {path: sum(cer for _, cer in items) / 4 for path, items in groups.items()}
    minimum = min(scores.values())
    eligible = [path for path, score in scores.items() if score <= minimum + 0.1]
    winner = min(eligible, key=lambda path: (
        protocols[path]["methods"]["full"]["steps"],
        protocols[path]["methods"]["full"]["learning_rate"], path))
    return winner, [{"candidate": path, "mean_validation_CER": score,
                     "run_ids": [run_id for run_id, _ in groups[path]]}
                    for path, score in sorted(scores.items())]


def projected_seconds(runs, rate, overhead):
    return sum(run["optimizer_steps"] * rate + overhead for run in runs)


def interrupt(signum, frame):
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
    raise KeyboardInterrupt(f"Worker interrupted by signal {signum}")


def run_command(command, log_path, deadline):
    global ACTIVE
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Worker deadline reached")
    started = time.monotonic()
    with log_path.open("w") as log:
        ACTIVE = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=PROJECT, start_new_session=True)
        try:
            code = ACTIVE.wait(timeout=remaining)
            if code:
                raise RuntimeError(f"Command failed ({code}): {command[1:3]}; see {log_path.name}")
        finally:
            if ACTIVE.poll() is None:
                os.killpg(ACTIVE.pid, signal.SIGTERM)
                try:
                    ACTIVE.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(ACTIVE.pid, signal.SIGKILL)
                    ACTIVE.wait()
            ACTIVE = None
    return time.monotonic() - started


def mounted_preflight(args, output):
    import adapt
    import torch
    from importlib.metadata import version
    from index_calibration_sessions import index_sessions
    from check_calibration_coverage import check_coverage
    from fixed_adaptation import verify_m5b_gate

    if sys.version_info[:2] != (3, 10):
        raise ValueError("Require pinned Python 3.10")
    adapt.verify_upstream(args.upstream_dir)
    verify_m5b_gate()
    pins = {"torch": "2.3.0", "torchaudio": "2.3.0", "torchvision": "0.18.0",
            "pytorch-lightning": "1.8.6", "h5py": "3.11.0", "numpy": "1.24.4",
            "hydra-core": "1.3.2", "omegaconf": "2.3.0", "torchmetrics": "0.11.4",
            "setuptools": "69.5.1", "pip": "24.0", "lightning-bolts": "0.7.0"}
    actual = {name: version(name) for name in pins}
    if any(actual[name].split("+")[0] != value for name, value in pins.items()):
        raise ValueError("Runtime package pins differ")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Require one CUDA GPU")
    if "4090" not in torch.cuda.get_device_name(0):
        raise ValueError("Approved GPU model mismatch")
    expected = [item for item in json.loads(args.source_inventory.read_text())["files"]
                if item["path"].startswith("data/") and item["path"].endswith(".hdf5")]
    if len(expected) != 100 or {p.name for p in args.data_dir.iterdir()} != {
            Path(item["path"]).name for item in expected}:
        raise ValueError("Require exactly 100 expected session files and no extra entries")

    def verify_file(item):
        path = args.data_dir / Path(item["path"]).name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Session inventory mismatch: {path.name}")
        return {"session": path.stem, "bytes": item["bytes"], "sha256": item["sha256"]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify_file, expected))
    if sha256(args.checkpoint) != adapt.GENERIC_SHA256:
        raise ValueError("Generic checkpoint mismatch")
    index = index_sessions(args.upstream_dir, args.data_dir)
    write_json(output / "session-index.json", index)
    coverage = check_coverage(args.upstream_dir, index)
    write_json(output / "calibration-coverage.json", coverage)
    write_json(output / "mounted-preflight.json", {
        "passed": True, "sessions": verified, "session_count": len(verified),
        "total_data_bytes": sum(item["bytes"] for item in verified),
        "generic_sha256": adapt.GENERIC_SHA256, "packages": actual,
        "GPU": torch.cuda.get_device_name(0), "CUDA": torch.version.cuda,
        "CPU_threads": torch.get_num_threads(), "CPU_interop_threads": torch.get_num_interop_threads(),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
                   if Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "session_index_digest": index["index_digest"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    state = {"milestone": "M7", "phase": "preflight", "completed": False,
             "test_evaluated": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "completed_run_ids": []}
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    rate, overhead = 0.25, 30.0
    matrix = json.loads((PROJECT / "configs/m7-proposed-matrix.json").read_text())
    protocols = {entry["path"]: json.loads((PROJECT / entry["path"]).read_text())
                 for entry in matrix["candidates"]}
    results = {}
    try:
        for protocol in protocols.values():
            validate_protocol(protocol)
        write_json(output / "session-state.json", state)
        mounted_preflight(args, output)
        smoke = json.loads(json.dumps(next(iter(protocols.values()))))
        smoke["purpose"] = "Unscored 100-update CUDA/data smoke, validation only"
        smoke["methods"]["full"].update(steps=100, learning_rate=0.001, warmup_steps=10)
        smoke_path = output / "smoke-protocol.json"
        write_json(smoke_path, smoke)

        def execute(run_id, user, budget, protocol_path):
            destination = output / "runs" / run_id
            command = [sys.executable, str(PROJECT / "scripts/adapt.py"), "--run", "--tune",
                       "--select", "fixed", "--method", "full", "--user", user,
                       "--budget-minutes", budget, "--seed", "1501", "--accelerator", "gpu",
                       "--protocol", str(protocol_path), "--upstream-dir", str(args.upstream_dir),
                       "--data-dir", str(args.data_dir), "--checkpoint", str(args.checkpoint),
                       "--session-index", str(output / "session-index.json"),
                       "--output-dir", str(destination)]
            elapsed = run_command(command, output / f"{run_id}.launcher.log", args.work_deadline_epoch)
            record = json.loads((destination / "result.json").read_text())
            if not record.get("completed") or record.get("test_evaluated"):
                raise ValueError("Run incomplete or test evaluated")
            return record, elapsed

        for budget in ("1", "full"):
            state.update(phase="smoke", current_run=f"smoke-user0-{budget}")
            write_json(output / "session-state.json", state)
            record, elapsed = execute(state["current_run"], "user0", budget, smoke_path)
            rate = max(rate, record["fit_wall_clock_seconds"] / 100)
            overhead = max(overhead, elapsed - record["fit_wall_clock_seconds"])
        remaining = matrix["tuning_runs"]
        for index, run in enumerate(remaining):
            estimate = projected_seconds(remaining[index:], rate, overhead)
            if time.time() + estimate > args.work_deadline_epoch:
                raise TimeoutError("Complete remaining tuning grid projects beyond approved work deadline")
            state.update(phase="tuning", current_run=run["id"],
                         projected_remaining_seconds=estimate, observed_seconds_per_update=rate,
                         observed_run_overhead_seconds=overhead)
            write_json(output / "session-state.json", state)
            record, elapsed = execute(run["id"], run["user"], run["budget_minutes"], PROJECT / run["candidate"])
            results[run["id"]] = record
            state["completed_run_ids"].append(run["id"])
            rate = max(rate, record["fit_wall_clock_seconds"] / run["optimizer_steps"])
            overhead = max(overhead, elapsed - record["fit_wall_clock_seconds"])
        winner, scores = choose_candidate(matrix, protocols, results)
        chosen_records = [results[run["id"]] for run in remaining if run["candidate"] == winner]
        selected = json.loads(json.dumps(protocols[winner]))
        selected["purpose"] = "Shared M7 full-method profile selected by the declared user0/user1 validation grid"
        frozen = freeze_protocol(selected, chosen_records)
        frozen["tuning_matrix_digest"] = digest(matrix)
        frozen["candidate_scores"] = scores
        write_json(output / "frozen-protocol.json", frozen)
        write_json(output / "tuning-summary.json", {"completed": True, "winner": winner,
                   "scores": scores, "selection": matrix["selection"], "test_evaluated": False,
                   "total_optimizer_updates": 32000, "unscored_smoke_updates": 200,
                   "frozen_protocol_digest": digest(frozen)})
        state.update(phase="complete", completed=True, winner=winner)
    except BaseException as error:
        state.update(phase="stopped", failure=str(error), failure_type=type(error).__name__)
        traceback.print_exc()
    finally:
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "session-final.json", state)
        write_json(output / "session-state.json", state)
    return 0 if state["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
