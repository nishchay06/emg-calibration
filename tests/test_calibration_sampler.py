import copy
import random
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from calibration_sampler import (BUDGETS, SPREAD_ALGORITHM, select_calibration, verify_selection,
                                 window_accounting)
from generate_test_user_manifests import UserManifest, parse_user_config


def manifest():
    return UserManifest("user0", {"train": ("2020-12-18-200-train", "2020-12-17-100-train", "2020-12-19-300-train"),
                                  "val": ("validation",), "test": ("test",)})


def eight_session_manifest():
    train = tuple(f"2020-12-{10 + i:02d}-{1000 + i}-train" for i in range(8))
    m = UserManifest("user3", {"train": train, "val": ("validation",), "test": ("test",)})
    return m, {s: 2_000_000 + 37_003 * i for i, s in enumerate(train)}


# Digests produced by training-contiguous-v1 before training-spread-v1 existed.
CONTIGUOUS_DIGESTS = {
    ("1", 1501): "532b54caac20eafac947498464efdf5f6a31af93484fad2302ab600dfa30c235",
    ("1", 1502): "a6ae468c7e1acc1d52a7e5166f72e3ff552e240ccf99c7e2c11ccd070e4282be",
    ("5", 1501): "5459e4392478f8bee5cda715f3aa7902ab4fe6f4233fde2c4ed52fc93ce76ebd",
    ("5", 1502): "c2d9e7baea1bc183f9449a61d5fbe080d880b8660928a7e68c842cf5d4eadb79",
    ("10", 1501): "9b92bcd00bed513b8fa088d86fb65c73015e6274f3940272fa9cf3d12555a628",
    ("10", 1502): "4d22cffed695d4facad98c976627b2f1e97834d953468ca1730c211423a5d28b",
    ("60", 1501): "617df1ec8925239283f4900a7b21a4422aa6c3392b4b1f2d0352952703a32940",
    ("60", 1502): "d7a682f117caadb2a2ea3c05858da5f5e58012b218484c57d018a92ec89b24ab",
    ("full", 1501): "e676b199179a47886965acff84b1df1594eaeafba7ee1a114628137b60239f8f",
    ("full", 1502): "4ea3c8a4799753a46aa258a2a5a63566a84689568508e46a66e51efd28d336a8",
}


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

    def test_contiguous_allocation_unchanged_by_spread(self):
        m, lengths = eight_session_manifest()
        for (budget, seed), expected in CONTIGUOUS_DIGESTS.items():
            with self.subTest(budget=budget, seed=seed):
                record = select_calibration(m, lengths, budget, seed)
                self.assertEqual(record["selection_digest"], expected)
                self.assertEqual(record, select_calibration(m, lengths, budget, seed, "contiguous"))
                self.assertNotIn("spread_sessions", record)

    def test_spread_exact_distinct_training_only_and_repeatable(self):
        m, lengths = eight_session_manifest()
        for budget in ("1", "2", "5", "10", "30", "60"):
            for seed in (0, 1501, 1502, 2**32 - 1):
                with self.subTest(budget=budget, seed=seed):
                    record = select_calibration(m, lengths, budget, seed, "spread")
                    target = int(budget) * 120_000
                    self.assertEqual(record["algorithm"], SPREAD_ALGORITHM)
                    self.assertEqual(record["spread_sessions"], 5)
                    self.assertEqual(record["requested_samples"], target)
                    self.assertEqual(record["selected_samples"], target)
                    spans = record["ranges"]
                    self.assertEqual(len({s["session"] for s in spans}), 5)
                    self.assertEqual([s["session"] for s in spans], sorted(s["session"] for s in spans))
                    for span in spans:
                        self.assertIn(span["session"], m.splits["train"])
                        self.assertEqual(span["stop"] - span["start"], target // 5)
                        self.assertGreaterEqual(span["start"], 0)
                        self.assertLessEqual(span["stop"], lengths[span["session"]])
                        self.assertEqual(span["session_samples"], lengths[span["session"]])
                    random.seed(27)
                    random.random()
                    self.assertEqual(record, select_calibration(m, lengths, budget, seed, "spread"))
                    verify_selection(record, m, lengths)

    def test_spread_seeds_and_budgets_draw_independently(self):
        m, lengths = eight_session_manifest()
        for budget in ("5", "10"):
            with self.subTest(budget=budget):
                first = select_calibration(m, lengths, budget, 1501, "spread")["ranges"]
                second = select_calibration(m, lengths, budget, 1502, "spread")["ranges"]
                self.assertNotEqual(first, second)
        five = select_calibration(m, lengths, "5", 1501, "spread")
        ten = select_calibration(m, lengths, "10", 1501, "spread")
        self.assertNotEqual([(s["session"], s["start"]) for s in five["ranges"]],
                            [(s["session"], s["start"]) for s in ten["ranges"]])

    def test_spread_rejects_full_unknown_and_insufficient_sessions(self):
        m, lengths = eight_session_manifest()
        with self.assertRaisesRegex(ValueError, "minute budget"):
            select_calibration(m, lengths, "full", 1501, "spread")
        with self.assertRaisesRegex(ValueError, "allocation"):
            select_calibration(m, lengths, "5", 1501, "nested")
        with self.assertRaisesRegex(ValueError, "5 training sessions"):
            select_calibration(manifest(), dict.fromkeys(manifest().splits["train"], 3_000_007), "5", 1501, "spread")
        short = {s: (100_000 if i < 4 else 2_000_000) for i, s in enumerate(m.splits["train"])}
        with self.assertRaisesRegex(ValueError, "5 training sessions"):
            select_calibration(m, short, "5", 1501, "spread")

    def test_forged_spread_records_rejected(self):
        m, lengths = eight_session_manifest()
        good = select_calibration(m, lengths, "5", 1501, "spread")
        for change in ("algorithm", "unknown", "range", "count"):
            bad = copy.deepcopy(good)
            if change == "algorithm":
                bad["algorithm"] = "training-contiguous-v1"
            elif change == "unknown":
                bad["algorithm"] = "training-nested-v0"
            elif change == "range":
                bad["ranges"][0]["start"] += 1
                bad["ranges"][0]["stop"] += 1
            else:
                bad["spread_sessions"] = 4
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_selection(bad, m, lengths)

    def test_spread_window_accounting_has_complete_windows_per_session(self):
        m, lengths = eight_session_manifest()
        accounting = window_accounting(select_calibration(m, lengths, "5", 1501, "spread"))
        self.assertEqual(accounting["training_windows"], 5 * (120_000 // 8000))
        self.assertEqual(accounting["segments_without_complete_window"], 0)

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
            for budget in ("5", "10"):
                for seed in (1501, 1502):
                    verify_selection(select_calibration(m, lengths, budget, seed, "spread"), m, lengths)


if __name__ == "__main__":
    unittest.main()
