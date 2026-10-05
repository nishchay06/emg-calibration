"""Audit completeness, provenance, allocation replay and honest scientific failures."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibration_report import collect_report, load_manifests, spearman, write_report
from calibration_report_demo import demo_inputs
from calibration_sampler import digest
from generate_test_user_manifests import UPSTREAM_COMMIT, UserManifest


def fixture_manifests():
    manifests = {}
    for i in range(8):
        user = f"user{i}"
        names = [f"2021-01-01-{i * 100 + j}-fictional" for j in range(13 if i < 4 else 12)]
        manifests[user] = UserManifest(user, {"train": tuple(names[:-4]), "val": tuple(names[-4:-2]),
                                             "test": tuple(names[-2:])})
    return manifests


class CalibrationReportTests(unittest.TestCase):
    def setUp(self):
        self.manifests = fixture_manifests()
        self.protocol, self.index, self.runs, self.baselines, self.refs = demo_inputs(self.manifests)

    def collect(self, **kwargs):
        return collect_report(self.protocol, self.index, self.manifests, self.runs,
                              self.baselines, self.refs, synthetic_demo=kwargs.get("demo", True))

    def test_complete_report_uses_actual_full_duration_and_marks_cohorts(self):
        report = self.collect()
        self.assertEqual(len(report["curve"]), 21)
        self.assertTrue(report["scientific_gate"]["passed"])
        self.assertEqual(report["sentinel"]["user"], "user5")
        user2 = [r for r in report["curve"] if r["user"] == "user2"]
        self.assertEqual(user2[-1]["selected_minutes"], 180)
        self.assertEqual({r["cohort"] for r in user2}, {"untouched"})

    def test_missing_duplicate_and_extra_runs_are_refused(self):
        original = copy.deepcopy(self.runs)
        for changed in (original[:-1], original + [original[0]], original[:-2]):
            self.runs = changed
            with self.subTest(count=len(changed)), self.assertRaises(ValueError): self.collect()

    def test_mixed_protocol_tuning_and_invalid_training_proof_are_refused(self):
        original = copy.deepcopy(self.runs[0])
        for update in ({"phase": "tuning"}, {"completed": False}, {"test_evaluated": False},
                       {"protocol_digest": "a" * 64}, {"session_index_digest": "a" * 64},
                       {"evaluation_checkpoint_sha256": "a" * 64}, {"optimizer_steps": 2},
                       {"finite_gradients": False}, {"validation_batches_during_fit": 1},
                       {"runtime": {"accelerator": "cpu"}},
                       {"fit_wall_clock_seconds": 170}, {"metrics": {"test": {"CER": float("nan")}}},
                       {"metrics": {"test": {"CER": 20}, "validation": {"CER": 10}}}):
            self.runs[0] = {**original, **update}
            with self.subTest(update=update), self.assertRaises(ValueError): self.collect()

    def test_tampered_allocation_is_rejected_even_with_recomputed_digest(self):
        selection = self.runs[0]["calibration_selection"]
        selection["ranges"][0]["start"] += 1
        selection["ranges"][0]["stop"] += 1
        selection["selection_digest"] = digest({k: v for k, v in selection.items() if k != "selection_digest"})
        with self.assertRaises(ValueError): self.collect()

    def test_index_split_and_generic_provenance_are_checked(self):
        original = copy.deepcopy(self.index)
        next(iter(self.index["sessions"].values()))["split"] = "test"
        self.index["index_digest"] = digest({k: v for k, v in self.index.items() if k != "index_digest"})
        with self.assertRaises(ValueError): self.collect()
        self.index = original
        self.baselines[0]["checkpoint"]["sha256"] = "a" * 64
        with self.assertRaises(ValueError): self.collect()

        self.setUp()
        for receipt in self.protocol["tuning_evidence"]:
            if receipt["budget_minutes"] == "full":
                receipt["profile_digest"] = "a" * 64
        with self.assertRaises(ValueError): self.collect()

    def test_synthetic_inputs_never_enter_ordinary_mode_or_mix_with_measured(self):
        with self.assertRaises(ValueError): self.collect(demo=False)
        self.runs[0].pop("synthetic_fixture")
        with self.assertRaises(ValueError): self.collect()

    def test_failed_gates_and_inversions_are_retained_not_filtered(self):
        self.runs[1]["metrics"]["test"]["CER"] = 80
        self.runs[6]["metrics"]["test"]["CER"] = 40
        report = self.collect()
        self.assertFalse(report["scientific_gate"]["passed"])
        self.assertFalse(report["scientific_gate"]["full_compatibility"]["user0"]["passed"])
        self.assertTrue(report["scientific_gate"]["adjacent_inversions"])
        self.assertEqual(len(report["curve"]), 21)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            write_report(report, output)
            self.assertIn("**FAIL**", (output / "table.md").read_text())
            with (output / "curve.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 21)
            self.assertEqual({r["provenance_kind"] for r in rows}, {"synthetic_pipeline_demo"})
            self.assertEqual(json.loads((output / "summary.json").read_text())["kind"], "synthetic_pipeline_demo")
            with self.assertRaises(FileExistsError): write_report(report, output)

    def test_cer_above_100_is_valid_and_tied_rank_correlation_is_defined(self):
        self.runs[0]["metrics"]["test"]["CER"] = 140
        self.assertEqual(self.collect()["curve"][0]["test_CER_percent"], 140)
        self.assertEqual(spearman([3, 2, 1]), -1)
        self.assertAlmostEqual(spearman([3, 2, 2]), -0.8660254037844387)
        self.assertIsNone(spearman([2, 2, 2]))

    def test_dirty_official_config_cannot_be_treated_as_pinned_splits(self):
        with patch("calibration_report.upstream_head", return_value=UPSTREAM_COMMIT), \
                patch("calibration_report.subprocess.run", return_value=Mock(returncode=1)):
            with self.assertRaisesRegex(ValueError, "configurations were modified"):
                load_manifests(Path("unused-checkout"))


if __name__ == "__main__":
    unittest.main()
