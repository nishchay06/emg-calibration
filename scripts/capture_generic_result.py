#!/usr/bin/env python3
"""Convert a completed generic-greedy console log into a result JSON file."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = PROJECT_DIR / "references" / "generic-greedy.json"
METRICS = ("loss", "CER", "IER", "DER", "SER")
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
QUOTED_METRIC_PATTERN = re.compile(
    rf"['\"](?P<split>val|test)/(?P<metric>loss|CER|IER|DER|SER)['\"]"
    rf"\s*:\s*(?P<value>{NUMBER_PATTERN})"
)
TABLE_METRIC_PATTERN = re.compile(
    rf"^\s*(?P<split>val|test)/(?P<metric>loss|CER|IER|DER|SER)"
    rf"\s+(?P<value>{NUMBER_PATTERN})\s*$",
    re.MULTILINE,
)


class CaptureError(ValueError):
    """Raised when a log or reference file is not a complete valid input."""


def parse_metrics(console_text: str) -> dict[str, dict[str, float]]:
    """Return the last complete validation and test metrics in a console log."""
    matches = [
        match
        for pattern in (TABLE_METRIC_PATTERN, QUOTED_METRIC_PATTERN)
        for match in pattern.finditer(console_text)
    ]
    matches.sort(key=lambda match: match.start())

    parsed: dict[str, dict[str, float]] = {"val": {}, "test": {}}
    for match in matches:
        value = float(match.group("value"))
        if not math.isfinite(value):
            raise CaptureError(
                f"Non-finite {match.group('split')}/{match.group('metric')} value"
            )
        parsed[match.group("split")][match.group("metric")] = value

    missing = [
        f"{split}/{metric}"
        for split in ("val", "test")
        for metric in METRICS
        if metric not in parsed[split]
    ]
    if missing:
        raise CaptureError("Missing completed metrics: " + ", ".join(missing))
    return parsed


def load_reference(path: Path, user: str) -> dict[str, Any]:
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureError(f"Cannot read reference file {path}: {error}") from error

    try:
        user_reference = reference["users"][user]
        validation_cer = float(user_reference["validation_CER"])
        test_cer = float(user_reference["test_CER"])
        checkpoint = reference["checkpoint"]
        source = reference["source"]
        decoder = reference["decoder"]
    except (KeyError, TypeError, ValueError) as error:
        raise CaptureError(
            f"Reference file {path} has no valid entry for {user}"
        ) from error

    return {
        "checkpoint": checkpoint,
        "decoder": decoder,
        "source": source,
        "validation_CER": validation_cer,
        "test_CER": test_cer,
    }


def build_result(
    user: str,
    console_log: Path,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
) -> dict[str, Any]:
    try:
        console_text = console_log.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise CaptureError(f"Cannot read console log {console_log}: {error}") from error

    parsed = parse_metrics(console_text)
    reference = load_reference(reference_path, user)
    validation_cer = parsed["val"]["CER"]
    test_cer = parsed["test"]["CER"]

    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "upstream_commit": reference["source"]["upstream_commit"],
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
            "validation": validation_cer - reference["validation_CER"],
            "test": test_cer - reference["test_CER"],
        },
        "reference_source": {
            "repository": reference["source"]["repository"],
            "path": reference["source"]["path"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user", choices=[f"user{index}" for index in range(8)])
    parser.add_argument("console_log", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Structured upstream references (default: repository reference file)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_json.exists():
        print(f"Refusing to overwrite output file: {args.output_json}", file=sys.stderr)
        return 1

    try:
        result = build_result(args.user, args.console_log, args.reference_file)
    except CaptureError as error:
        print(f"Result capture failed: {error}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Captured {args.user}: "
        f"val CER={result['metrics']['validation']['CER']:.6f}% "
        f"test CER={result['metrics']['test']['CER']:.6f}%"
    )
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
