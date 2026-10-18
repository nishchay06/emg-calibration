#!/usr/bin/env python3
"""Sequential approved M7 tuning; no test evaluation or implicit resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from calibration_protocol import validate_protocol
from calibration_sampler import digest
from generate_test_user_manifests import UPSTREAM_COMMIT

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
    import torch
    count = sum(1 for path in args.data_dir.iterdir() if path.is_file() and path.suffix == ".hdf5")
    if count != 100:
        raise ValueError(f"Expected 100 HDF5 files in {args.data_dir}; found {count}")
    if not args.checkpoint.is_file():
        raise ValueError(f"Generic checkpoint is missing: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    write_json(output / "mounted-preflight.json", {
        "passed": True, "session_count": count, "checkpoint_exists": True,
        "CUDA": True, "GPU": torch.cuda.get_device_name(0)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        from index_calibration_sessions import index_sessions
        index = index_sessions(args.upstream_dir, args.data_dir)
        write_json(output / "session-index.json", index)

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
        write_json(output / "selection.json", {"completed": True, "winner": winner,
                   "scores": scores, "selection": matrix["selection"], "test_evaluated": False})
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
