#!/usr/bin/env python3
"""Generate deterministic archive manifests from pinned upstream user configs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_COMMIT = "3200d91eeb952cbed1f278e47d0cc56928334fd1"
ARCHIVE_PREFIX = "emg2qwerty-data-2021-08/"
SPLITS = ("train", "val", "test")
DEFAULT_USERS = tuple(f"user{index}" for index in range(8))

SPLIT_PATTERN = re.compile(r"^  (train|val|test):\s*$")
SESSION_PATTERN = re.compile(r"^\s+-?\s*session:\s+(\S+)\s*$")


@dataclass(frozen=True)
class UserManifest:
    user: str
    splits: dict[str, tuple[str, ...]]

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(
            session
            for split in SPLITS
            for session in self.splits[split]
        )


def parse_user_config(config_path: Path, user: str) -> UserManifest:
    split_sessions: dict[str, list[str]] = {split: [] for split in SPLITS}
    active_split: str | None = None

    for line_number, line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        split_match = SPLIT_PATTERN.match(line)
        if split_match:
            active_split = split_match.group(1)
            continue

        session_match = SESSION_PATTERN.match(line)
        if session_match and active_split:
            session = session_match.group(1)
            if not session:
                raise ValueError(f"Empty session at {config_path}:{line_number}")
            split_sessions[active_split].append(session)

    frozen_splits = {
        split: tuple(split_sessions[split])
        for split in SPLITS
    }
    if any(not frozen_splits[split] for split in SPLITS):
        counts = {split: len(frozen_splits[split]) for split in SPLITS}
        raise ValueError(f"Missing split sessions in {config_path}: {counts}")

    sessions = [session for split in SPLITS for session in frozen_splits[split]]
    if len(sessions) != len(set(sessions)):
        raise ValueError(f"Duplicate sessions in {config_path}")

    return UserManifest(user=user, splits=frozen_splits)


def archive_member(session: str) -> str:
    return f"{ARCHIVE_PREFIX}{session}.hdf5"


def manifest_text(manifest: UserManifest) -> str:
    return "".join(f"{archive_member(session)}\n" for session in manifest.sessions)


def metadata_text(manifests: list[UserManifest]) -> str:
    users = {}
    all_sessions: list[str] = []
    split_totals = {split: 0 for split in SPLITS}

    for manifest in manifests:
        counts = {
            split: len(manifest.splits[split])
            for split in SPLITS
        }
        users[manifest.user] = {
            "counts": counts,
            "total": sum(counts.values()),
        }
        for split in SPLITS:
            split_totals[split] += counts[split]
        all_sessions.extend(manifest.sessions)

    if len(all_sessions) != len(set(all_sessions)):
        raise ValueError("A session is assigned to more than one test user")

    metadata = {
        "archive_prefix": ARCHIVE_PREFIX,
        "split_totals": split_totals,
        "total_unique_sessions": len(all_sessions),
        "upstream_commit": UPSTREAM_COMMIT,
        "users": users,
    }
    return json.dumps(metadata, indent=2, sort_keys=True) + "\n"


def upstream_head(upstream_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def expected_outputs(
    upstream_dir: Path, users: tuple[str, ...]
) -> dict[str, str]:
    actual_commit = upstream_head(upstream_dir)
    if actual_commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"Expected upstream {UPSTREAM_COMMIT}; found {actual_commit}"
        )

    manifests = []
    for user in users:
        config_path = upstream_dir / "config" / "user" / f"{user}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing upstream config: {config_path}")
        manifests.append(parse_user_config(config_path, user))

    outputs = {
        f"{manifest.user}-sessions.txt": manifest_text(manifest)
        for manifest in manifests
    }
    combined_members = sorted(
        archive_member(session)
        for manifest in manifests
        for session in manifest.sessions
    )
    outputs["test-users-sessions.txt"] = "".join(
        f"{member}\n" for member in combined_members
    )
    outputs["test-users.json"] = metadata_text(manifests)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "upstream_dir",
        type=Path,
        help="Path to the emg2qwerty checkout at the pinned commit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "manifests",
        help="Manifest directory (default: repository manifests directory)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if checked-in outputs differ; do not write files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = expected_outputs(args.upstream_dir.resolve(), DEFAULT_USERS)

    if args.check:
        mismatches = []
        for filename, expected in outputs.items():
            path = args.output_dir / filename
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(str(path))
        if mismatches:
            print("Manifest outputs are stale:", file=sys.stderr)
            for path in mismatches:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(f"Validated {len(outputs)} manifest outputs")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        (args.output_dir / filename).write_text(content, encoding="utf-8")
    print(f"Generated {len(outputs)} files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
