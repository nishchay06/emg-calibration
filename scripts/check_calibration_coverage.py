#!/usr/bin/env python3
"""Preview all eight users' budgets from an indexed dataset, without training."""
import argparse
import json
import sys
from pathlib import Path

from calibration_sampler import BUDGETS, digest, select_calibration, window_accounting
from fixed_adaptation import indexed_lengths
from generate_test_user_manifests import DEFAULT_USERS, UPSTREAM_COMMIT, parse_user_config, upstream_head


def check_coverage(upstream, index):
    if upstream_head(upstream) != UPSTREAM_COMMIT:
        raise ValueError("Upstream pin mismatch")
    selections = []
    for user in DEFAULT_USERS:
        manifest = parse_user_config(upstream / "config/user" / f"{user}.yaml", user)
        lengths = indexed_lengths(index, manifest)
        for budget in BUDGETS:
            selection = select_calibration(manifest, lengths, budget, 1501)
            accounting = window_accounting(selection)
            if not accounting["training_windows"]:
                raise ValueError(f"No complete training windows: {user}, {budget}")
            selections.append({"selection": selection, "window_accounting": accounting})
    result = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
              "session_index_digest": index["index_digest"], "selection_seed": 1501,
              "passed": True, "selections_checked": len(selections), "selections": selections,
              "limitation": "Allocation from indexed lengths only; not a signal-integrity or training check."}
    result["coverage_digest"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--session-index", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(check_coverage(args.upstream_dir, json.loads(args.session_index.read_text())),
                         indent=2, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Calibration coverage failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
