#!/usr/bin/env python3
"""Verify one personalized checkpoint against the pinned upstream LFS identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = PROJECT_DIR / "references" / "personalized-greedy.json"
BENCHMARKS = ("randominit", "finetuned")
USERS = tuple(f"user{index}" for index in range(8))


class VerificationError(ValueError):
    """Raised when a checkpoint does not match its pinned identity."""


def load_checkpoint_reference(
    reference_path: Path, benchmark: str, user: str
) -> dict[str, Any]:
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        checkpoint = reference["benchmarks"][benchmark]["users"][user]["checkpoint"]
        filename = str(checkpoint["filename"])
        sha256 = str(checkpoint["sha256"])
        size_bytes = int(checkpoint["size_bytes"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise VerificationError(
            f"Cannot load {benchmark}/{user} checkpoint identity from "
            f"{reference_path}: {error}"
        ) from error
    return {
        "filename": filename,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    checkpoint_path: Path,
    expected: dict[str, Any],
) -> None:
    if not checkpoint_path.is_file():
        raise VerificationError(f"Checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.name != expected["filename"]:
        raise VerificationError(
            f"Expected checkpoint filename {expected['filename']}; "
            f"found {checkpoint_path.name}"
        )
    actual_size = checkpoint_path.stat().st_size
    if actual_size != expected["size_bytes"]:
        raise VerificationError(
            f"Expected {expected['size_bytes']} bytes; found {actual_size}: "
            f"{checkpoint_path}"
        )
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected["sha256"]:
        raise VerificationError(
            f"Expected SHA-256 {expected['sha256']}; found {actual_sha256}: "
            f"{checkpoint_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("user", choices=USERS)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Pinned personalized reference file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        expected = load_checkpoint_reference(
            args.reference_file, args.benchmark, args.user
        )
        verify_checkpoint(args.checkpoint, expected)
    except VerificationError as error:
        print(f"Checkpoint verification failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Verified {args.benchmark}/{args.user}: "
        f"{expected['size_bytes']} bytes SHA-256 {expected['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
