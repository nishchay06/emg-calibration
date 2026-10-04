#!/usr/bin/env python3
"""Generate personalized checkpoint identities and greedy CER references."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


UPSTREAM_COMMIT = "3200d91eeb952cbed1f278e47d0cc56928334fd1"
USERS = tuple(f"user{index}" for index in range(8))
BENCHMARKS = {
    "randominit": {
        "checkpoint_directory": "models/personalized-randominit",
        "upstream_label": "Personalized (random-init)",
    },
    "finetuned": {
        "checkpoint_directory": "models/personalized-finetuned",
        "upstream_label": "Personalized (finetuned)",
    },
}
LFS_VERSION = "https://git-lfs.github.com/spec/v1"
OID_PATTERN = re.compile(r"^oid sha256:([0-9a-f]{64})$")
SIZE_PATTERN = re.compile(r"^size ([1-9][0-9]*)$")


class ReferenceError(ValueError):
    """Raised when pinned upstream evidence is missing or malformed."""


def upstream_head(upstream_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_lfs_pointer(text: str, source: str = "checkpoint") -> dict[str, Any]:
    lines = text.splitlines()
    if len(lines) != 3 or lines[0] != f"version {LFS_VERSION}":
        raise ReferenceError(f"Invalid Git LFS pointer: {source}")
    oid_match = OID_PATTERN.fullmatch(lines[1])
    size_match = SIZE_PATTERN.fullmatch(lines[2])
    if not oid_match or not size_match:
        raise ReferenceError(f"Invalid Git LFS pointer: {source}")
    return {
        "sha256": oid_match.group(1),
        "size_bytes": int(size_match.group(1)),
    }


def load_experimental_results(path: Path) -> list[dict[str, Any]]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ReferenceError(f"Cannot parse {path}: {error}") from error

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "EXPERIMENTAL_RESULTS"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError) as error:
            raise ReferenceError("EXPERIMENTAL_RESULTS is not literal data") from error
        if not isinstance(value, list) or not all(
            isinstance(row, dict) for row in value
        ):
            raise ReferenceError("EXPERIMENTAL_RESULTS must be a list of records")
        return value
    raise ReferenceError(f"EXPERIMENTAL_RESULTS not found in {path}")


def cer_values(
    records: list[dict[str, Any]], benchmark_label: str, metric: str
) -> tuple[float, ...]:
    matching = [
        row
        for row in records
        if row.get("Model benchmark") == benchmark_label
        and row.get("LM") == "No LM"
        and row.get("Metric") == metric
    ]
    if len(matching) != 1:
        raise ReferenceError(
            f"Expected one {benchmark_label!r} / No LM / {metric!r} row; "
            f"found {len(matching)}"
        )
    raw_values = matching[0].get("CER")
    if not isinstance(raw_values, list) or len(raw_values) != len(USERS):
        raise ReferenceError(f"Expected eight CER values for {benchmark_label} {metric}")
    values = tuple(float(value) for value in raw_values)
    if not all(math.isfinite(value) for value in values):
        raise ReferenceError(f"Non-finite CER value for {benchmark_label} {metric}")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_reference(upstream_dir: Path) -> dict[str, Any]:
    actual_commit = upstream_head(upstream_dir)
    if actual_commit != UPSTREAM_COMMIT:
        raise ReferenceError(
            f"Expected upstream {UPSTREAM_COMMIT}; found {actual_commit}"
        )

    results_path = upstream_dir / "scripts" / "experimental_results.py"
    records = load_experimental_results(results_path)
    benchmarks: dict[str, Any] = {}

    for benchmark, metadata in BENCHMARKS.items():
        validation = cer_values(records, metadata["upstream_label"], "Val CER")
        test = cer_values(records, metadata["upstream_label"], "Test CER")
        users: dict[str, Any] = {}
        total_bytes = 0
        for index, user in enumerate(USERS):
            relative_path = Path(metadata["checkpoint_directory"]) / f"{user}.ckpt"
            checkpoint_path = upstream_dir / relative_path
            try:
                checkpoint = parse_lfs_pointer(
                    checkpoint_path.read_text(encoding="utf-8"),
                    str(relative_path),
                )
            except OSError as error:
                raise ReferenceError(f"Cannot read {relative_path}: {error}") from error
            total_bytes += checkpoint["size_bytes"]
            users[user] = {
                "checkpoint": {
                    "filename": checkpoint_path.name,
                    "sha256": checkpoint["sha256"],
                    "size_bytes": checkpoint["size_bytes"],
                },
                "test_CER": test[index],
                "validation_CER": validation[index],
            }
        benchmarks[benchmark] = {
            "checkpoint_directory": metadata["checkpoint_directory"],
            "checkpoint_total_bytes": total_bytes,
            "upstream_label": metadata["upstream_label"],
            "users": users,
        }

    return {
        "benchmarks": benchmarks,
        "decoder": "ctc_greedy",
        "schema_version": 1,
        "source": {
            "checkpoint_identity": "Git LFS pointer SHA-256 and size",
            "path": "scripts/experimental_results.py",
            "repository": "facebookresearch/emg2qwerty",
            "upstream_commit": UPSTREAM_COMMIT,
        },
    }


def rendered_reference(upstream_dir: Path) -> str:
    return json.dumps(expected_reference(upstream_dir), indent=2, sort_keys=True) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "upstream_dir",
        type=Path,
        help="Path to the emg2qwerty checkout at the pinned commit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "references"
            / "personalized-greedy.json"
        ),
        help="Reference output path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the checked-in reference differs; do not write files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        expected = rendered_reference(args.upstream_dir.resolve())
    except (ReferenceError, subprocess.CalledProcessError) as error:
        print(f"Reference generation failed: {error}", file=sys.stderr)
        return 1

    if args.check:
        try:
            actual = args.output.read_text(encoding="utf-8")
        except OSError as error:
            print(f"Cannot read reference output {args.output}: {error}", file=sys.stderr)
            return 1
        if actual != expected:
            print(f"Personalized reference is stale: {args.output}", file=sys.stderr)
            return 1
        print("Validated personalized checkpoint identities and CER references")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
