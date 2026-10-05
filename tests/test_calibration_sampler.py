import copy
import random
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from calibration_sampler import BUDGETS, select_calibration, verify_selection, window_accounting
from generate_test_user_manifests import UserManifest, parse_user_config


def manifest():
    return UserManifest("user0", {"train": ("2020-12-18-200-train", "2020-12-17-100-train", "2020-12-19-300-train"),
                                  "val": ("validation",), "test": ("test",)})


class CalibrationSamplerTest(unittest.TestCase):
    def test_all_budgets_exact_repeatable_and_training_only(self):
        m = manifest()
        lengths = dict.fromkeys(m.splits["train"], 3_000_007)
        for budget in BUDGETS:
            for seed in (0, 1501, 2**32 - 1):
                with self.subTest(budget=budget, seed=seed):
                    record = select_calibration(m, lengths, budget, seed)
                    expected = sum(lengths.values()) if budget == "full" else int(budget) * 120_000
                    self.assertEqual(record["selected_samples"], expected)
                    random.seed(27)
                    random.random()
                    self.assertEqual(record, select_calibration(m, lengths, budget, seed))
                    verify_selection(record, m, lengths)
                    spans = record["ranges"]
                    self.assertEqual(len(spans), len({s["session"] for s in spans}))
                    for span in spans:
                        self.assertIn(span["session"], m.splits["train"])
                        self.assertGreaterEqual(span["start"], 0)
                        self.assertGreater(span["stop"], span["start"])
                        self.assertLessEqual(span["stop"], lengths[span["session"]])

    def test_single_session_and_consecutive_multisession(self):
        m = manifest()
        lengths = dict.fromkeys(m.splits["train"], 3_000_007)
        self.assertEqual(len(select_calibration(m, lengths, "10", 1501)["ranges"]), 1)
        spans = select_calibration(m, lengths, "60", 1501)["ranges"]
        self.assertEqual([s["session"] for s in spans], sorted(m.splits["train"]))
        for previous, following in zip(spans, spans[1:]):
            self.assertEqual(previous["stop"], lengths[previous["session"]])
            self.assertEqual(following["start"], 0)

    def test_full_preserves_upstream_order_and_all_samples(self):
        m = manifest()
        lengths = dict(zip(m.splits["train"], (17, 19, 23)))
        spans = select_calibration(m, lengths, "full", 1501)["ranges"]
        self.assertEqual([s["session"] for s in spans], list(m.splits["train"]))
        self.assertEqual([(s["start"], s["stop"]) for s in spans], [(0, 17), (0, 19), (0, 23)])

    def test_exact_capacity_and_reject_insufficient_missing_and_overlap(self):
        m = manifest()
        lengths = dict.fromkeys(m.splits["train"], 40_000)
        record = select_calibration(m, lengths, "1", 1501)
        self.assertTrue(all(s["start"] == 0 for s in record["ranges"]))
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_calibration(m, lengths, "2", 1501)
        for bad in ({}, {**lengths, m.splits["train"][0]: 0}, {**lengths, m.splits["train"][0]: True}):
            with self.assertRaises(ValueError):
                select_calibration(m, bad, "full", 1501)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            select_calibration(UserManifest("user0", {**m.splits, "val": m.splits["train"][:1]}), lengths, "full", 0)

    def test_forged_or_stale_records_rejected(self):
        m = manifest()
        lengths = dict.fromkeys(m.splits["train"], 200_000)
        good = select_calibration(m, lengths, "1", 1501)
        for change in ("range", "digest", "count"):
            bad = copy.deepcopy(good)
            if change == "range":
                bad["ranges"][0]["session"] = "validation"
            elif change == "digest":
                bad["selection_digest"] = "a" * 64
            else:
                bad["selected_samples"] += 1
            with self.assertRaises(ValueError):
                verify_selection(bad, m, lengths)
        with self.assertRaises(ValueError):
            verify_selection(good, m, {**lengths, m.splits["train"][0]: 200_001})

    def test_window_accounting_does_not_claim_unique_consumption(self):
        m = manifest()
        lengths = dict.fromkeys(m.splits["train"], 8001)
        record = select_calibration(m, lengths, "full", 0)
        accounting = window_accounting(record)
        self.assertEqual(accounting["training_windows"], 3)
        self.assertEqual(accounting["nominal_remainder_samples"], 3)

    def test_all_pinned_users_and_budgets(self):
        upstream = SCRIPTS.parent / "upstream/emg2qwerty"
        if not upstream.is_dir():
            self.skipTest("requires pinned upstream configurations")
        for i in range(8):
            m = parse_user_config(upstream / "config/user" / f"user{i}.yaml", f"user{i}")
            # Synthetic lengths validate the real split identities, not real coverage.
            lengths = dict.fromkeys(m.splits["train"], 2_000_000)
            for budget in BUDGETS:
                record = select_calibration(m, lengths, budget, 1501)
                verify_selection(record, m, lengths)


if __name__ == "__main__":
    unittest.main()
