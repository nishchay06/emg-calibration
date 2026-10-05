"""Deterministic, training-only calibration allocation; no data dependencies."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping

from generate_test_user_manifests import UPSTREAM_COMMIT, UserManifest

BUDGETS = ("1", "2", "5", "10", "30", "60", "full")
SAMPLES_PER_MINUTE = 120_000
ALGORITHM = "training-contiguous-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest: UserManifest, lengths: Mapping[str, int]) -> None:
    if manifest.user not in {f"user{i}" for i in range(8)}:
        raise ValueError("Unknown benchmark user")
    if set(manifest.splits) != {"train", "val", "test"}:
        raise ValueError("Expected train/val/test splits")
    if any(not sessions for sessions in manifest.splits.values()):
        raise ValueError("Empty split")
    if len(manifest.sessions) != len(set(manifest.sessions)):
        raise ValueError("Duplicate or overlapping split sessions")
    for session in manifest.sessions:
        if not re.fullmatch(r"[A-Za-z0-9_@.+-]+", session) or session in {".", ".."}:
            raise ValueError("Unsafe session identifier")
    for session in manifest.splits["train"]:
        length = lengths.get(session)
        if type(length) is not int or length <= 0:
            raise ValueError(f"Missing or invalid sample length: {session}")


def chronological_key(session: str):
    match = re.match(r"^\d{4}-\d{2}-\d{2}-(\d+)-", session)
    if not match:
        raise ValueError(f"Missing recording timestamp in session name: {session}")
    return int(match[1]), session


def select_calibration(manifest: UserManifest, lengths: Mapping[str, int],
                       budget: str, seed: int) -> dict:
    validate_manifest(manifest, lengths)
    if budget not in BUDGETS:
        raise ValueError("Unsupported calibration budget")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    training = manifest.splits["train"]
    total = sum(lengths[session] for session in training)
    target = total if budget == "full" else int(budget) * SAMPLES_PER_MINUTE
    if target > total:
        raise ValueError(f"Insufficient training data: need {target}, have {total}")
    if budget == "full":
        spans = [{"session": session, "start": 0, "stop": lengths[session],
                  "session_samples": lengths[session]} for session in training]
    else:
        # Independent of model, method, global RNG, worker count and path layout.
        rng = random.Random(int(digest([ALGORITHM, manifest.user, budget, seed]), 16))
        order = sorted(training, key=chronological_key)
        eligible = [session for session in order if lengths[session] >= target]
        if eligible:
            session = rng.choice(eligible)
            start = rng.randrange(lengths[session] - target + 1)
            spans = [{"session": session, "start": start, "stop": start + target,
                      "session_samples": lengths[session]}]
        else:
            suffix = [sum(lengths[s] for s in order[i:]) for i in range(len(order))]
            anchor = rng.choice([i for i, capacity in enumerate(suffix) if capacity >= target])
            first = order[anchor]
            offset = rng.randrange(min(lengths[first] - 1, suffix[anchor] - target) + 1)
            spans, remaining = [], target
            for session in order[anchor:]:
                take = min(lengths[session] - offset, remaining)
                spans.append({"session": session, "start": offset, "stop": offset + take,
                              "session_samples": lengths[session]})
                remaining -= take
                offset = 0
                if remaining == 0:
                    break
    record = {
        "schema_version": 1, "algorithm": ALGORITHM, "upstream_commit": UPSTREAM_COMMIT,
        "user": manifest.user, "budget_minutes": budget, "selection_seed": seed,
        "samples_per_minute": SAMPLES_PER_MINUTE, "requested_samples": target,
        "selected_samples": sum(s["stop"] - s["start"] for s in spans),
        "available_training_samples": total,
        "split_digest": digest(manifest.splits),
        "training_lengths_digest": digest({s: lengths[s] for s in training}),
        "interval_convention": "start-inclusive, stop-exclusive",
        "session_order": "upstream" if budget == "full" else "chronological recording timestamps",
        "ranges": spans,
    }
    record["selection_digest"] = digest(record)
    return record


def verify_selection(record: dict, manifest: UserManifest, lengths: Mapping[str, int]) -> None:
    """Reject forged, stale or out-of-budget records by replaying selection."""
    if record != select_calibration(manifest, lengths, record["budget_minutes"], record["selection_seed"]):
        raise ValueError("Calibration selection differs from its deterministic allocation")


def window_accounting(record: dict, window_length: int = 8000) -> dict:
    if type(window_length) is not int or window_length <= 0:
        raise ValueError("Window length must be positive")
    counts = [(span["stop"] - span["start"]) // window_length for span in record["ranges"]]
    return {"selected_samples": record["selected_samples"], "window_length": window_length,
            "training_windows": sum(counts), "nominal_core_samples": sum(counts) * window_length,
            "nominal_remainder_samples": record["selected_samples"] - sum(counts) * window_length,
            "segments_without_complete_window": sum(count == 0 for count in counts),
            "limitation": "Nominal non-jittered window capacity; not unique samples observed during repeated training. Context and jitter stay inside allocated ranges."}
