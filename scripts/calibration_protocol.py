"""Versioned fixed-update protocols and user0/user1 tuning provenance."""
from __future__ import annotations

import json
import math

from calibration_sampler import BUDGETS, digest
from generate_test_user_manifests import UPSTREAM_COMMIT

PROFILE_KEYS = {"steps", "learning_rate", "warmup_steps", "warmup_start_lr", "minimum_lr"}


def validate_profile(profile):
    if set(profile) != PROFILE_KEYS:
        raise ValueError("Expected explicit steps, learning rate and warmup/cosine parameters")
    steps, warmup = profile["steps"], profile["warmup_steps"]
    if type(steps) is not int or steps < 1 or type(warmup) is not int or not 0 <= warmup <= max(0, steps - 2):
        raise ValueError("Invalid optimizer-step or warmup count")
    for name in ("learning_rate", "warmup_start_lr", "minimum_lr"):
        value = profile[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}")
        if value > profile["learning_rate"]:
            raise ValueError("Schedule endpoints cannot exceed the learning rate")


def learning_rate_at_update(profile, update):
    """LR used by update 0..S-1, with the final update at minimum_lr (S>1)."""
    steps, warmup, peak = profile["steps"], profile["warmup_steps"], profile["learning_rate"]
    if type(update) is not int or update < 0:
        raise ValueError("Update index must be nonnegative")
    update = min(update, steps - 1)
    if steps == 1:
        return peak
    if update < warmup:
        if warmup == 1:
            return profile["warmup_start_lr"]
        return profile["warmup_start_lr"] + (peak - profile["warmup_start_lr"]) * update / (warmup - 1)
    progress = (update - warmup) / (steps - 1 - warmup)
    return profile["minimum_lr"] + 0.5 * (peak - profile["minimum_lr"]) * (1 + math.cos(math.pi * progress))


def validate_protocol(protocol, method="full", *, require_frozen=False):
    if protocol.get("schema_version") != 1 or protocol.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Protocol schema or upstream pin mismatch")
    if protocol.get("selection") != "final" or protocol.get("schedule") != "update-warmup-cosine-v1":
        raise ValueError("Fixed adaptation requires final selection and an update-based schedule")
    if protocol.get("status") not in ("draft", "frozen"):
        raise ValueError("Protocol must be draft or frozen")
    for name, profile in protocol.get("methods", {}).items():
        if name not in ("full", "head", "norm", "lora"):
            raise ValueError("Unknown adaptation method")
        validate_profile(profile)
    if method not in protocol.get("methods", {}):
        raise ValueError(f"Protocol has no {method} profile")
    if require_frozen and protocol["status"] != "frozen":
        raise ValueError("Ordinary experiments require a frozen user0/user1-tuned protocol")
    if protocol["status"] == "frozen":
        for name, profile in protocol["methods"].items():
            matches = [r for r in protocol.get("tuning_evidence", [])
                       if r.get("method") == name and r.get("profile_digest") == digest(profile)]
            if {r.get("user") for r in matches} != {"user0", "user1"} or len(matches) < 2:
                raise ValueError("Frozen profiles require both user0/user1 tuning receipts")
            contexts = {"user0": set(), "user1": set()}
            for receipt in matches:
                if (receipt.get("selection"), receipt.get("test_evaluated"), receipt.get("optimizer_steps")) != (
                        "final", False, profile["steps"]):
                    raise ValueError("Invalid tuning receipt")
                for key in ("result_digest", "final_checkpoint_sha256"):
                    value = receipt.get(key, "")
                    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        raise ValueError("Missing tuning provenance digest")
                cer = receipt.get("validation_CER")
                if type(cer) not in (int, float) or not math.isfinite(cer) or cer < 0:
                    raise ValueError("Invalid tuning validation CER")
                budget, seed = receipt.get("budget_minutes"), receipt.get("seed")
                if budget is not None and budget not in BUDGETS:
                    raise ValueError("Invalid tuning budget")
                if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
                    raise ValueError("Invalid tuning seed")
                context = (budget, seed)
                if context in contexts[receipt["user"]]:
                    raise ValueError("Duplicate tuning user/budget/seed receipt")
                contexts[receipt["user"]].add(context)
            if contexts["user0"] != contexts["user1"]:
                raise ValueError("Tuning users require the same budget/seed contexts")
    return protocol["methods"][method]


def load_protocol(path, method="full", *, require_frozen=False):
    protocol = json.loads(path.read_text())
    profile = validate_protocol(protocol, method, require_frozen=require_frozen)
    return protocol, profile


def freeze_protocol(candidate, results):
    """Freeze a selected profile after final-checkpoint validation on tuning users."""
    frozen = json.loads(json.dumps(candidate))
    if frozen.get("status") != "draft":
        raise ValueError("Expected draft candidate")
    validate_protocol(frozen)
    receipts = []
    for result in results:
        method = result.get("method")
        profile = frozen["methods"].get(method)
        if profile is None or result.get("profile_digest") != digest(profile):
            raise ValueError("Tuning result does not match the selected profile")
        if (result.get("phase"), result.get("user"), result.get("selection"), result.get("test_evaluated"),
                result.get("completed"), result.get("optimizer_steps"), result.get("upstream_commit")) not in [
                ("tuning", user, "fixed", False, True, profile["steps"], UPSTREAM_COMMIT)
                for user in ("user0", "user1")]:
            raise ValueError("Require completed user0/user1 final-checkpoint validation without test evaluation")
        if "test" in result.get("metrics", {}):
            raise ValueError("Test metrics cannot enter tuning provenance")
        receipts.append({"method": method, "user": result["user"], "profile_digest": digest(profile),
                         "result_digest": digest(result), "selection": "final", "test_evaluated": False,
                         "optimizer_steps": result["optimizer_steps"],
                         "validation_CER": result["metrics"]["validation"]["CER"],
                         "final_checkpoint_sha256": result["trained_checkpoint"]["sha256"]})
        for key in ("budget_minutes", "seed"):
            if key in result:
                receipts[-1][key] = result[key]
    frozen.update(status="frozen", tuning_evidence=receipts)
    validate_protocol(frozen, require_frozen=True)
    return frozen
