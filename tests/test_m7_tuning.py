"""Prevent incomplete or test-informed tuning from producing a frozen profile."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibration_sampler import digest
from generate_test_user_manifests import UPSTREAM_COMMIT
from run_m7_tuning import choose_candidate, projected_seconds
from calibration_protocol import freeze_protocol


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.matrix = json.loads((ROOT / "configs/m7-proposed-matrix.json").read_text())
        self.protocols = {c["path"]: json.loads((ROOT / c["path"]).read_text())
                          for c in self.matrix["candidates"]}
        self.results = {}
        for run in self.matrix["tuning_runs"]:
            profile = self.protocols[run["candidate"]]["methods"]["full"]
            self.results[run["id"]] = {
                "phase": "tuning", "user": run["user"], "budget_minutes": run["budget_minutes"],
                "upstream_commit": UPSTREAM_COMMIT,
                "initial_checkpoint": {"sha256": "338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81"},
                "method": "full", "seed": 1501, "selection": "fixed", "completed": True,
                "test_evaluated": False, "profile_digest": digest(profile),
                "optimizer_steps": profile["steps"], "validation_batches_during_fit": 0,
                "finite_losses": True, "finite_gradients": True, "data_preflight": "passed",
                "trained_checkpoint": {"sha256": "a" * 64}, "evaluation_checkpoint_sha256": "a" * 64,
                "metrics": {"validation": {"CER": 25.0}},
                "calibration_selection": {"user": run["user"], "budget": run["budget_minutes"]}}

    def choose(self):
        return choose_candidate(self.matrix, self.protocols, self.results)

    def test_tie_prefers_fewer_steps_then_lower_lr(self):
        winner, scores = self.choose()
        profile = self.protocols[winner]["methods"]["full"]
        self.assertEqual((profile["steps"], profile["learning_rate"]), (1000, 0.0003))
        self.assertEqual(len(scores), 4)

    def test_unweighted_four_pair_mean(self):
        best = self.matrix["candidates"][-1]["path"]
        for run in self.matrix["tuning_runs"]:
            if run["candidate"] == best:
                self.results[run["id"]]["metrics"]["validation"]["CER"] = 24
        self.assertEqual(self.choose()[0], best)

    def test_incomplete_grid_cannot_choose(self):
        self.results.pop(next(iter(self.results)))
        with self.assertRaises(ValueError): self.choose()

    def test_test_metrics_are_rejected_even_if_test_flag_is_false(self):
        next(iter(self.results.values()))["metrics"]["test"] = {"CER": 0}
        with self.assertRaises(ValueError): self.choose()

    def test_cross_candidate_allocation_mismatch_is_rejected(self):
        record = self.results[self.matrix["tuning_runs"][4]["id"]]
        record["calibration_selection"]["budget"] = "different"
        with self.assertRaises(ValueError): self.choose()

    def test_invalid_training_and_reload_proof_are_rejected(self):
        original = copy.deepcopy(self.results)
        for key, bad_value in [("finite_gradients", False), ("optimizer_steps", 5),
                               ("validation_batches_during_fit", 1),
                               ("evaluation_checkpoint_sha256", "b" * 64)]:
            self.results = copy.deepcopy(original)
            next(iter(self.results.values()))[key] = bad_value
            with self.subTest(key=key), self.assertRaises(ValueError): self.choose()

    def test_projection_includes_all_remaining_runs_and_overheads(self):
        self.assertEqual(projected_seconds(self.matrix["tuning_runs"], 0.25, 30), 8480)

    def test_complete_matrix_selection_freezes_all_winning_receipts(self):
        winner, scores = self.choose()
        records = [self.results[r["id"]] for r in self.matrix["tuning_runs"] if r["candidate"] == winner]
        frozen = freeze_protocol(self.protocols[winner], records)
        self.assertEqual(frozen["status"], "frozen")
        self.assertEqual(len(frozen["tuning_evidence"]), 4)
        self.assertEqual({r["budget_minutes"] for r in frozen["tuning_evidence"]}, {"5", "full"})


if __name__ == "__main__":
    unittest.main()
