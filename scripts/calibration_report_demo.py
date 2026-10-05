#!/usr/bin/env python3
"""Build explicitly fictional receipts to exercise reporting; never run participant training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration_protocol import freeze_protocol
from calibration_report import GENERIC_SHA256, SEED, collect_report, load_manifests, write_report
from calibration_sampler import BUDGETS, digest, select_calibration, window_accounting
from generate_test_user_manifests import UPSTREAM_COMMIT


def demo_inputs(manifests):
    """All lengths, checkpoints, timings and CER values below are invented fixtures."""
    lengths = {s: 2_400_000 for m in manifests.values() for s in m.sessions}
    sessions = {s: {"user": user, "split": split, "samples": lengths[s], "file_bytes": 1}
                for user, m in manifests.items() for split, names in m.splits.items() for s in names}
    index = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT, "synthetic_fixture": True,
             "source": "Fictional lengths; no participant recordings read", "sessions": sessions,
             "total_sessions": len(sessions)}
    index["index_digest"] = digest(index)
    profile = {"steps": 3000, "learning_rate": 0.001, "warmup_steps": 300,
               "warmup_start_lr": 1e-8, "minimum_lr": 1e-6}
    draft = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT, "synthetic_fixture": True,
             "status": "draft", "selection": "final", "schedule": "update-warmup-cosine-v1",
             "methods": {"full": profile}}
    tuning = [{"synthetic_fixture": True, "user": user, "budget_minutes": budget, "seed": SEED,
               "method": "full", "phase": "tuning", "selection": "fixed", "test_evaluated": False,
               "completed": True, "optimizer_steps": profile["steps"], "upstream_commit": UPSTREAM_COMMIT,
               "profile_digest": digest(profile), "trained_checkpoint": {"sha256": digest([user, budget, "fictional"])},
               "metrics": {"validation": {"CER": 30.0}}}
              for user in ("user0", "user1") for budget in ("5", "full")]
    protocol = freeze_protocol(draft, tuning)
    invented_cer = {"user0": [52, 49, 40, 33, 26, 23, 19.5],
                    "user1": [48, 45, 36, 29, 22, 20, 18],
                    "user2": [50, 47, 38, 31, 24, 21, 19], "user5": [8.5]}
    runs = []
    for user, values in invented_cer.items():
        for budget, cer in zip(BUDGETS if user != "user5" else ("full",), values):
            selection = select_calibration(manifests[user], lengths, budget, SEED)
            checkpoint = digest([user, budget, "fictional-final"])
            runs.append({"synthetic_fixture": True, "schema_version": 1, "pipeline": "fixed-step-calibration",
                         "phase": "adaptation", "user": user, "budget_minutes": budget, "method": "full",
                         "selection": "fixed", "seed": SEED, "upstream_commit": UPSTREAM_COMMIT,
                         "protocol_digest": digest(protocol), "protocol_status": "frozen", "profile": profile,
                         "profile_digest": digest(profile), "optimizer_steps": profile["steps"],
                         "completed": True, "training_executed": True, "process_exit_code": 0, "test_evaluated": True,
                         "data_preflight": "passed", "initial_checkpoint": {"sha256": GENERIC_SHA256},
                         "session_index_digest": index["index_digest"], "calibration_selection": selection,
                         "window_accounting": window_accounting(selection), "finite_losses": True,
                         "finite_gradients": True, "validation_batches_during_fit": 0,
                         "trained_checkpoint": {"filename": "final.ckpt", "sha256": checkpoint, "global_step": profile["steps"]},
                         "evaluation_checkpoint_sha256": checkpoint, "metrics": {"test": {"CER": cer}},
                         "parameters": {"total": 5293315, "trainable": 5293315},
                         "runtime": {"accelerator": "gpu", "note": "Fictional fixture; no GPU used"},
                         "fit_wall_clock_seconds": 150.0, "wall_clock_seconds": 160.0})
    baselines = [{"synthetic_fixture": True, "user": u, "upstream_commit": UPSTREAM_COMMIT,
                  "decoder": "ctc_greedy", "checkpoint": {"sha256": GENERIC_SHA256},
                  "acceptance": {"passed": True, "test": True}, "metrics": {"test": {"CER": cer}}}
                 for u, cer in zip(("user0", "user1", "user2"), (60, 55, 57))]
    references = [{"synthetic_fixture": True, "milestone": "M5b", "user": u, "budget_minutes": "full",
                   "method": "full", "selection": "upstream", "seed": SEED, "upstream_commit": UPSTREAM_COMMIT,
                   "training_executed": True, "process_exit_code": 0, "acceptance": {"passed": True},
                   "last_checkpoint": {"epoch": 149}, "trained_checkpoint": {"sha256": digest([u, "fictional-m5b"])},
                   "metrics": {"test": {"CER": cer}}} for u, cer in (("user0", 20), ("user5", 8))]
    return protocol, index, runs, baselines, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifests = load_manifests(args.upstream_dir)
    protocol, index, runs, baselines, references = demo_inputs(manifests)
    report = collect_report(protocol, index, manifests, runs, baselines, references, synthetic_demo=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = args.output_dir / "fictional-inputs"
    inputs.mkdir()
    for name, value in (("protocol", protocol), ("index", index), ("runs", runs),
                        ("baselines", baselines), ("m5b-references", references)):
        (inputs / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    write_report(report, args.output_dir / "report")
    print(report["warning"])


if __name__ == "__main__":
    main()
