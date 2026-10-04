#!/usr/bin/env python3
"""Convert a personalized greedy console log into a result JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capture_generic_result import (
    MAX_CER_DELTA_PERCENTAGE_POINTS,
    CaptureError,
    cer_delta_is_accepted,
    parse_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = PROJECT_DIR / "references" / "personalized-greedy.json"
BENCHMARKS = ("randominit", "finetuned")
USERS = tuple(f"user{index}" for index in range(8))


def load_reference(path: Path, benchmark: str, user: str) -> dict[str, Any]:
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
        benchmark_reference = reference["benchmarks"][benchmark]
        user_reference = benchmark_reference["users"][user]
        validation_cer = float(user_reference["validation_CER"])
        test_cer = float(user_reference["test_CER"])
        checkpoint = user_reference["checkpoint"]
        source = reference["source"]
        decoder = reference["decoder"]
        upstream_label = benchmark_reference["upstream_label"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise CaptureError(
            f"Reference file {path} has no valid {benchmark}/{user} entry"
        ) from error
    return {
        "checkpoint": checkpoint,
        "decoder": decoder,
        "source": source,
        "test_CER": test_cer,
        "upstream_label": upstream_label,
        "validation_CER": validation_cer,
    }


def build_result(
    benchmark: str,
    user: str,
    console_log: Path,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
) -> dict[str, Any]:
    try:
        console_text = console_log.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise CaptureError(f"Cannot read console log {console_log}: {error}") from error

    parsed = parse_metrics(console_text)
    reference = load_reference(reference_path, benchmark, user)
    validation_cer = parsed["val"]["CER"]
    test_cer = parsed["test"]["CER"]
    validation_delta = validation_cer - reference["validation_CER"]
    test_delta = test_cer - reference["test_CER"]
    validation_accepted = cer_delta_is_accepted(validation_delta)
    test_accepted = cer_delta_is_accepted(test_delta)

    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "upstream_commit": reference["source"]["upstream_commit"],
        "benchmark": benchmark,
        "upstream_benchmark": reference["upstream_label"],
        "user": user,
        "decoder": reference["decoder"],
        "checkpoint": reference["checkpoint"],
        "source_log": console_log.name,
        "metrics": {
            "validation": parsed["val"],
            "test": parsed["test"],
        },
        "reference_CER": {
            "validation": reference["validation_CER"],
            "test": reference["test_CER"],
        },
        "CER_delta_percentage_points": {
            "validation": validation_delta,
            "test": test_delta,
        },
        "acceptance": {
            "maximum_absolute_CER_delta_percentage_points": (
                MAX_CER_DELTA_PERCENTAGE_POINTS
            ),
            "validation": validation_accepted,
            "test": test_accepted,
            "passed": validation_accepted and test_accepted,
        },
        "reference_source": {
            "repository": reference["source"]["repository"],
            "path": reference["source"]["path"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("user", choices=USERS)
    parser.add_argument("console_log", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Structured upstream references",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_json.exists():
        print(f"Refusing to overwrite output file: {args.output_json}", file=sys.stderr)
        return 1
    try:
        result = build_result(
            args.benchmark,
            args.user,
            args.console_log,
            args.reference_file,
        )
    except CaptureError as error:
        print(f"Result capture failed: {error}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Captured {args.benchmark}/{args.user}: "
        f"val CER={result['metrics']['validation']['CER']:.6f}% "
        f"test CER={result['metrics']['test']['CER']:.6f}%"
    )
    print(f"Wrote {args.output_json}")
    if not result["acceptance"]["passed"]:
        failed_splits = [
            split
            for split in ("validation", "test")
            if not result["acceptance"][split]
        ]
        failure_details = ", ".join(
            f"{split} |delta|="
            f"{abs(result['CER_delta_percentage_points'][split]):.6f} pp"
            for split in failed_splits
        )
        print(
            f"Acceptance failed for {args.benchmark}/{args.user}: "
            f"{failure_details}; required <= "
            f"{MAX_CER_DELTA_PERCENTAGE_POINTS:.2f} pp.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
