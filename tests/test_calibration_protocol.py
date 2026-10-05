import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from calibration_protocol import freeze_protocol, learning_rate_at_update, validate_profile, validate_protocol
from calibration_sampler import digest
from generate_test_user_manifests import UPSTREAM_COMMIT
import adapt


def draft():
    return {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT, "status": "draft",
            "selection": "final", "schedule": "update-warmup-cosine-v1",
            "methods": {"full": {"steps": 5, "learning_rate": 0.001, "warmup_steps": 2,
                                  "warmup_start_lr": 1e-8, "minimum_lr": 1e-6}}}


def tuning_result(user="user0"):
    profile = draft()["methods"]["full"]
    return {"user": user, "method": "full", "phase": "tuning", "selection": "fixed",
            "test_evaluated": False, "completed": True, "optimizer_steps": profile["steps"],
            "upstream_commit": UPSTREAM_COMMIT, "profile_digest": digest(profile),
            "trained_checkpoint": {"sha256": "a" * 64}, "metrics": {"validation": {"CER": 20.0}}}


class CalibrationProtocolTest(unittest.TestCase):
    def test_explicit_update_schedule_endpoints_and_invalid_profiles(self):
        profile = draft()["methods"]["full"]
        rates = [learning_rate_at_update(profile, i) for i in range(5)]
        self.assertAlmostEqual(rates[0], 1e-8)
        self.assertAlmostEqual(rates[1], 0.001)
        self.assertAlmostEqual(rates[2], 0.001)
        self.assertAlmostEqual(rates[-1], 1e-6)
        self.assertGreater(rates[2], rates[3])
        for update in ({"steps": 0}, {"steps": True}, {"warmup_steps": 5},
                       {"learning_rate": float("nan")}, {"minimum_lr": 0.1}):
            with self.assertRaises(ValueError):
                validate_profile({**profile, **update})

    def test_freeze_requires_matching_user0_user1_final_validation(self):
        frozen = freeze_protocol(draft(), [tuning_result(), tuning_result("user1")])
        self.assertEqual(validate_protocol(frozen, require_frozen=True), draft()["methods"]["full"])
        for records in ([tuning_result()], [tuning_result(), tuning_result()],
                        [tuning_result(), tuning_result("user2")]):
            with self.assertRaises(ValueError):
                freeze_protocol(draft(), records)
        for update in ({"completed": False}, {"optimizer_steps": 4}, {"test_evaluated": True},
                       {"profile_digest": "b" * 64}, {"selection": "upstream"},
                       {"metrics": {"validation": {"CER": 1}, "test": {"CER": 1}}}):
            with self.assertRaises(ValueError):
                freeze_protocol(draft(), [{**tuning_result(), **update}, tuning_result("user1")])
        bad = copy.deepcopy(frozen)
        bad["methods"]["full"]["steps"] += 1
        with self.assertRaises(ValueError):
            validate_protocol(bad, require_frozen=True)

    def test_fixed_dry_run_stdlib_only_creates_nothing_and_rejects_draft_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "draft.json"
            path.write_text(json.dumps(draft()))
            common = ["--user", "user2", "--select", "fixed", "--budget-minutes", "1",
                      "--protocol", str(path), "--upstream-dir", str(root / "upstream"),
                      "--data-dir", str(root / "data"), "--output-dir", str(root / "output")]
            result = subprocess.run([sys.executable, "-S", str(PROJECT / "scripts/adapt.py"),
                                     *common, "--dry-run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertFalse(record["training_executed"])
            self.assertEqual(record["data_preflight"], "pending")
            self.assertEqual(list(root.iterdir()), [path])
            for extra in (["--run"], ["--dry-run", "--steps", "7"], ["--dry-run", "--tune"]):
                failed = subprocess.run([sys.executable, "-S", str(PROJECT / "scripts/adapt.py"),
                                         *common, *extra], capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse((root / "output").exists())

    def test_freeze_preserves_all_four_matched_budget_receipts(self):
        records = [{**tuning_result(user), "budget_minutes": budget, "seed": 1501}
                   for user in ("user0", "user1") for budget in ("5", "full")]
        frozen = freeze_protocol(draft(), records)
        self.assertEqual(len(frozen["tuning_evidence"]), 4)
        self.assertEqual({(r["user"], r["budget_minutes"], r["seed"])
                          for r in frozen["tuning_evidence"]},
                         {(u, b, 1501) for u in ("user0", "user1") for b in ("5", "full")})
        with self.assertRaises(ValueError):
            freeze_protocol(draft(), records[:-1])
        with self.assertRaises(ValueError):
            freeze_protocol(draft(), records + [records[0]])


if __name__ == "__main__":
    unittest.main()
