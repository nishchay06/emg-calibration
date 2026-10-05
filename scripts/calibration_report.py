#!/usr/bin/env python3
"""Validate the complete M7 first curve and export auditable tables, without training."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from calibration_protocol import validate_protocol
from calibration_sampler import BUDGETS, SAMPLES_PER_MINUTE, digest, verify_selection, window_accounting
from generate_test_user_manifests import DEFAULT_USERS, UPSTREAM_COMMIT, parse_user_config, upstream_head

GENERIC_SHA256 = "338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81"
CURVE_USERS = ("user0", "user1", "user2")
SEED = 1501
DEMO_WARNING = "SYNTHETIC PIPELINE DEMO — NOT RESEARCH RESULTS"
CSV_FIELDS = ("provenance_kind", "user", "cohort", "budget_minutes", "selected_minutes", "selected_samples",
              "test_CER_percent", "generic_CER_percent", "optimizer_steps", "trainable_parameters",
              "fit_wall_clock_seconds", "wall_clock_seconds", "selection_digest", "result_digest")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid {name}")
    return value


def sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def spearman(values):
    """Correlation with increasing budget order, averaging tied CER ranks."""
    def rank(items):
        return [1 + sum(other < item for other in items)
                + (sum(other == item for other in items) - 1) / 2 for item in items]

    x, y = list(range(len(values))), rank(values)
    xm, ym = statistics.mean(x), statistics.mean(y)
    denominator = math.sqrt(sum((v - xm)**2 for v in x) * sum((v - ym)**2 for v in y))
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / denominator if denominator else None


def collect_report(protocol, index, manifests, runs, baselines, references, *, synthetic_demo=False):
    """Check receipts and replay allocations; this does not remeasure CER or read signals."""
    inputs = [protocol, index, *runs, *baselines, *references]
    require(all(item.get("synthetic_fixture") is True for item in inputs) if synthetic_demo
            else all("synthetic_fixture" not in item for item in inputs),
            "Synthetic inputs require an explicitly synthetic report; never mix measured and fictional inputs")
    profile = validate_protocol(protocol, require_frozen=True)
    require(set(protocol["methods"]) == {"full"}, "M7 first curve requires only the full method")
    receipts = protocol["tuning_evidence"]
    require(len(receipts) == 4 and all(r.get("method") == "full" and r.get("profile_digest") == digest(profile) for r in receipts)
            and {(r["user"], r.get("budget_minutes"), r.get("seed")) for r in receipts}
            == {(u, b, SEED) for u in ("user0", "user1") for b in ("5", "full")},
            "M7 requires all four winning user0/user1 tuning contexts")
    require(set(manifests) == set(DEFAULT_USERS), "Require all eight official user manifests")
    expected_sessions = {name: (user, split) for user, manifest in manifests.items()
                         for split, names in manifest.splits.items() for name in names}
    require(len(expected_sessions) == 100 and index.get("total_sessions") == 100
            and set(index.get("sessions", {})) == set(expected_sessions), "Require exactly 100 indexed sessions")
    require(index.get("schema_version") == 1 and index.get("upstream_commit") == UPSTREAM_COMMIT
            and index.get("index_digest") == digest({k: v for k, v in index.items() if k != "index_digest"}),
            "Session-index provenance mismatch")
    lengths = {}
    for name, (user, split) in expected_sessions.items():
        item = index["sessions"][name]
        require((item.get("user"), item.get("split")) == (user, split), "Index differs from official splits")
        require(type(item.get("samples")) is int and item["samples"] > 0, "Invalid indexed sample length")
        lengths[name] = item["samples"]

    generic = {}
    for record in baselines:
        user = record.get("user")
        require(user in CURVE_USERS and user not in generic, "Unexpected or duplicate generic baseline")
        require(record.get("upstream_commit") == UPSTREAM_COMMIT and record.get("decoder") == "ctc_greedy"
                and record.get("checkpoint", {}).get("sha256") == GENERIC_SHA256
                and record.get("acceptance", {}).get("passed") is True
                and record["acceptance"].get("test") is True, "Generic baseline provenance mismatch")
        generic[user] = {"test_CER_percent": nonnegative(record["metrics"]["test"]["CER"], "generic test CER"),
                         "result_digest": digest(record)}
    require(set(generic) == set(CURVE_USERS), "Require the three accepted M4 generic baselines")

    m5b = {}
    for record in references:
        user = record.get("user")
        require(user in ("user0", "user5") and user not in m5b, "Unexpected or duplicate M5b reference")
        require((record.get("milestone"), record.get("upstream_commit"), record.get("method"),
                 record.get("selection"), record.get("budget_minutes"), record.get("seed"),
                 record.get("training_executed"), record.get("process_exit_code"))
                == ("M5b", UPSTREAM_COMMIT, "full", "upstream", "full", SEED, True, 0)
                and record.get("acceptance", {}).get("passed") is True
                and record.get("last_checkpoint", {}).get("epoch") == 149
                and sha256(record.get("trained_checkpoint", {}).get("sha256")), "Incomplete M5b reference")
        cer = nonnegative(record["metrics"]["test"]["CER"], "M5b test CER")
        if not synthetic_demo:
            require(abs(cer - {"user0": 20.57, "user5": 5.811}[user]) <= 1.0, "M5b reference accuracy failed")
        m5b[user] = {"test_CER_percent": cer, "result_digest": digest(record)}
    require(set(m5b) == {"user0", "user5"}, "Require both accepted M5b references")

    expected = {(u, b) for u in CURVE_USERS for b in BUDGETS} | {("user5", "full")}
    rows = {}
    for record in runs:
        user, budget = record.get("user"), record.get("budget_minutes")
        identity = (user, budget)
        require(identity in expected and identity not in rows, "Unexpected or duplicate curve run")
        require((record.get("schema_version"), record.get("pipeline"), record.get("phase"),
                 record.get("method"), record.get("selection"), record.get("seed"), record.get("upstream_commit"),
                 record.get("completed"), record.get("training_executed"), record.get("process_exit_code"),
                 record.get("test_evaluated"), record.get("protocol_status"))
                == (1, "fixed-step-calibration", "adaptation", "full", "fixed", SEED, UPSTREAM_COMMIT,
                    True, True, 0, True, "frozen"), "Require completed final-checkpoint adaptation tests")
        require(record.get("protocol_digest") == digest(protocol) and record.get("profile") == profile
                and record.get("profile_digest") == digest(profile)
                and record.get("optimizer_steps") == profile["steps"], "Shared fixed protocol mismatch")
        require(record.get("initial_checkpoint", {}).get("sha256") == GENERIC_SHA256
                and record.get("data_preflight") == "passed"
                and record.get("session_index_digest") == index["index_digest"], "Data/initial checkpoint mismatch")
        checkpoint = record.get("trained_checkpoint", {})
        require(checkpoint.get("filename") == "final.ckpt" and checkpoint.get("global_step") == profile["steps"]
                and sha256(checkpoint.get("sha256"))
                and record.get("evaluation_checkpoint_sha256") == checkpoint["sha256"], "Final checkpoint reload mismatch")
        require(record.get("finite_losses") is True and record.get("finite_gradients") is True
                and record.get("validation_batches_during_fit") == 0, "Invalid fixed-training proof")
        require(record.get("runtime", {}).get("accelerator") == "gpu" and "failure" not in record,
                "M7 participant results require successful GPU training")
        selection = record.get("calibration_selection", {})
        require((selection.get("user"), selection.get("budget_minutes"), selection.get("selection_seed"))
                == (user, budget, SEED), "Run/selection identity mismatch")
        verify_selection(selection, manifests[user], lengths)
        accounting = window_accounting(selection)
        require(record.get("window_accounting") == accounting and accounting["training_windows"] > 0,
                "Training-window accounting mismatch")
        require(set(record.get("metrics", {})) == {"test"}, "Curve records must contain test metrics only")
        cer = nonnegative(record["metrics"]["test"]["CER"], "test CER")
        parameters = record.get("parameters", {})
        require(type(parameters.get("total")) is int and parameters["total"] > 0
                and parameters.get("trainable") == parameters["total"], "Full adaptation must train all parameters")
        fit = nonnegative(record.get("fit_wall_clock_seconds"), "fit time")
        wall = nonnegative(record.get("wall_clock_seconds"), "wall time")
        require(fit <= wall, "Fit duration cannot exceed total run duration")
        rows[identity] = {"user": user, "cohort": "tuning" if user in ("user0", "user1") else "untouched",
                          "budget_minutes": budget, "selected_minutes": selection["selected_samples"] / SAMPLES_PER_MINUTE,
                          "selected_samples": selection["selected_samples"], "test_CER_percent": cer,
                          "generic_CER_percent": generic[user]["test_CER_percent"] if user in generic else None,
                          "optimizer_steps": profile["steps"], "trainable_parameters": parameters["trainable"],
                          "fit_wall_clock_seconds": fit, "wall_clock_seconds": wall,
                          "selection_digest": selection["selection_digest"], "result_digest": digest(record)}
    require(set(rows) == expected, "Require all 21 curve points and the user5 full sentinel")
    curve = [rows[(user, budget)] for user in CURVE_USERS for budget in BUDGETS]
    compatibility = {u: {"absolute_delta_pp": abs(rows[(u, "full")]["test_CER_percent"] - m5b[u]["test_CER_percent"])}
                     for u in ("user0", "user5")}
    for result in compatibility.values():
        result["passed"] = result["absolute_delta_pp"] <= 1.0
    trends, inversions = {}, []
    for user in CURVE_USERS:
        points = [rows[(user, b)] for b in BUDGETS]
        require(points[-1]["selected_minutes"] >= 60, "Full data must cover the largest numeric budget")
        rho = spearman([r["test_CER_percent"] for r in points])
        trends[user] = {"spearman_rho": rho, "passed": rho is not None and rho < 0}
        for first, second in zip(points, points[1:]):
            if second["test_CER_percent"] > first["test_CER_percent"]:
                inversions.append({"user": user, "from_budget": first["budget_minutes"],
                                   "to_budget": second["budget_minutes"],
                                   "increase_pp": second["test_CER_percent"] - first["test_CER_percent"]})
    median1 = statistics.median(rows[(u, "1")]["test_CER_percent"] for u in CURVE_USERS)
    median60 = statistics.median(rows[(u, "60")]["test_CER_percent"] for u in CURVE_USERS)
    passed = all(r["passed"] for r in compatibility.values()) and all(r["passed"] for r in trends.values()) and median60 < median1
    report = {"schema_version": 1, "kind": "synthetic_pipeline_demo" if synthetic_demo else "measured_m7_first_curve",
            "warning": DEMO_WARNING if synthetic_demo else None,
            "upstream_commit": UPSTREAM_COMMIT, "method": "full", "seed": SEED,
            "protocol_digest": digest(protocol), "profile": profile, "session_index_digest": index["index_digest"],
            "generic_baselines": generic, "m5b_references": m5b, "curve": curve, "sentinel": rows[("user5", "full")],
            "scientific_gate": {"passed": passed, "full_compatibility": compatibility, "trends": trends,
                                "median_1_minute_CER": median1, "median_60_minute_CER": median60,
                                "median_improved": median60 < median1, "adjacent_inversions": inversions},
            "limitations": ["One seed; exploratory descriptive results, no uncertainty or significance claims.",
                            "Users0/1 participated in tuning; user2 is the sole untouched curve user.",
                            "Generic baselines reuse accepted M4 results; zero minutes is a reference line on log axes.",
                            "Full x positions use selected sample counts at 2 kHz, not a nominal budget label.",
                            "Receipt checks replay allocations; they do not re-read signals/checkpoints or remeasure CER.",
                            "Scientific gates are analysis checks, not a complete paid-session/milestone acceptance record."]}
    report["report_digest"] = digest(report)
    return report


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "curve.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({"provenance_kind": report["kind"], **row} for row in report["curve"])
    lines = ["# M7 first-curve report", "", report["warning"] or "Measured adaptation results", "",
             f"Scientific gate: **{'PASS' if report['scientific_gate']['passed'] else 'FAIL'}**", "",
             "| User | Cohort | Budget | Selected minutes | Test CER (%) | Generic CER (%) |",
             "|---|---|---|---:|---:|---:|"]
    for r in report["curve"]:
        lines.append(f"| {r['user']} | {r['cohort']} | {r['budget_minutes']} | {r['selected_minutes']:.3f} | "
                     f"{r['test_CER_percent']:.6f} | {r['generic_CER_percent']:.6f} |")
    lines += ["", "## Gates and sentinel", "", "```json",
              json.dumps({"scientific_gate": report["scientific_gate"], "user5_sentinel": report["sentinel"]}, indent=2),
              "```", "", "## Interpretation", "", *[f"- {note}" for note in report["limitations"]], ""]
    (output / "table.md").write_text("\n".join(lines))


def load_manifests(upstream):
    require(upstream_head(upstream) == UPSTREAM_COMMIT, "Upstream checkout pin mismatch")
    require(subprocess.run(["git", "-C", str(upstream), "diff", "--quiet", UPSTREAM_COMMIT, "--", "config/user"],
                           capture_output=True).returncode == 0, "Official user configurations were modified")
    return {u: parse_user_config(upstream / "config/user" / f"{u}.yaml", u) for u in DEFAULT_USERS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--session-index", required=True, type=Path)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--baselines", required=True, nargs="+", type=Path)
    parser.add_argument("--m5b-references", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args(argv)
    try:
        read = lambda p: json.loads(p.read_text())
        report = collect_report(read(args.protocol), read(args.session_index), load_manifests(args.upstream_dir),
                                [read(p) for p in args.results], [read(p) for p in args.baselines],
                                [read(p) for p in args.m5b_references], synthetic_demo=args.synthetic_demo)
        write_report(report, args.output_dir)
        print(json.dumps({"kind": report["kind"], "curve_points": len(report["curve"]),
                          "scientific_gate": report["scientific_gate"]["passed"]}))
        return 0 if report["scientific_gate"]["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"M7 report refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
