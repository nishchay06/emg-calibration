from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import adapt


class AdaptGuardsTest(unittest.TestCase):
    def args(self, root, *extra):
        return ["--user", "user0", "--upstream-dir", str(root / "upstream"),
                "--data-dir", str(root / "data"), "--output-dir", str(root / "output"), *extra]

    def test_dry_run_needs_no_data_dependencies_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, "-S", str(SCRIPTS / "adapt.py"),
                                     *self.args(root, "--dry-run")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertFalse(record["training_executed"])
            self.assertIn("hydra.job.chdir=True", record["command"])
            self.assertEqual(list(root.iterdir()), [])

    def test_rejects_leakage_step_override_and_future_methods(self):
        cases = [("--budget-minutes", "1"), ("--method", "head"),
                 ("--method", "norm"), ("--method", "lora"),
                 ("--select", "fixed"), ("--steps", "100"), ("--seed", "-1"),
                 ("--user", "user8"), ("--user", "user5", "--run"),
                 ("--user", "user2", "--run")]
        with tempfile.TemporaryDirectory() as directory:
            for options in cases:
                with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                    mode = [] if "--run" in options else ["--dry-run"]
                    with self.assertRaises(SystemExit):
                        adapt.arguments(self.args(Path(directory), *mode, *options))

    def test_refuses_existing_output_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                adapt.arguments(self.args(root, "--dry-run"))

    def test_only_approved_upstream_patch_is_permitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "emg2qwerty/decoder.py"
            source.parent.mkdir()
            original = subprocess.check_output(["git", "-C", str(adapt.PROJECT / "upstream/emg2qwerty"),
                                                "show", f"{adapt.UPSTREAM_COMMIT}:emg2qwerty/decoder.py"], text=True) if (
                adapt.PROJECT / "upstream/emg2qwerty"
            ).is_dir() else None
            if original is None:
                self.skipTest("requires upstream checkout for the compatibility fixture")
            source.write_text(original)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "emg2qwerty/decoder.py"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                            "commit", "--quiet", "-m", "Synthetic upstream fixture"], check=True)
            head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            with mock.patch.object(adapt, "UPSTREAM_COMMIT", head):
                adapt.verify_upstream(root)
                subprocess.run(["git", "-C", str(root), "apply",
                                str(adapt.PROJECT / "patches/greedy-decoder-optional-kenlm.patch")], check=True)
                adapt.verify_upstream(root)
                source.write_text(source.read_text() + "\n# additional unapproved edit\n")
                with self.assertRaisesRegex(ValueError, "approved"):
                    adapt.verify_upstream(root)

    def test_gate_boundary_and_invalid_metrics(self):
        for value in (19.57, 20.57, 21.57):
            self.assertTrue(adapt.test_acceptance(value, 20.57)["passed"])
        for value in (19.569, 21.571):
            self.assertFalse(adapt.test_acceptance(value, 20.57)["passed"])
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                adapt.test_acceptance(value, 20.57)

    def test_m5a_evidence_cannot_unlock_user5_training(self):
        path = adapt.PROJECT / "results/m5-user0-personalized-finetuned-greedy.json"
        with self.assertRaisesRegex(ValueError, "M5b"):
            adapt.verify_user0_gate(path)

    def test_capture_requires_best_identity_complete_training_and_no_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            best = root / "checkpoints/epoch=42-step=123.ckpt"
            best.write_bytes(b"synthetic test checkpoint")
            metrics = "\n".join(f"'{split}/{metric}': {20.5 if metric == 'CER' else 0.5}"
                                for split in ("val", "test") for metric in ("loss", "CER", "IER", "DER", "SER"))
            text = metrics + f"\n'best_checkpoint': '{best}'\n"
            log = root / "console.log"
            log.write_text(text)
            def loader(path):
                return {"epoch": 149 if path.name == "last.ckpt" else 42, "global_step": 123}
            record = {"acceptance": {"test_CER_reference": 20.57}}
            adapt.capture_training_result(record, root, loader)
            self.assertTrue(record["acceptance"]["passed"])
            self.assertEqual(record["trained_checkpoint"]["sha256"], adapt.sha256(best))
            for invalid in (metrics, text + "Resuming training from checkpoint anything",
                            text.replace(str(best), str(root / "foreign.ckpt")),
                            text.replace("'test/CER': 20.5", "")):
                with self.subTest(invalid=invalid[:40]), self.assertRaises(ValueError):
                    log.write_text(invalid)
                    adapt.capture_training_result({}, root, loader)
            log.write_text(text)
            with self.assertRaisesRegex(ValueError, "150 epochs"):
                adapt.capture_training_result({}, root, lambda path: {"epoch": 148, "global_step": 100})

    def test_failed_or_incomplete_user0_cannot_unlock_user5(self):
        evidence = {"milestone": "M5b", "user": "user0", "budget_minutes": "full",
                    "method": "full", "selection": "upstream", "upstream_commit": adapt.UPSTREAM_COMMIT,
                    "training_executed": True, "process_exit_code": 0,
                    "acceptance": {"passed": True}, "last_checkpoint": {"epoch": 149},
                    "trained_checkpoint": {"sha256": "a" * 64}, "metrics": {"test": {"CER": 20.5}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(evidence))
            adapt.verify_user0_gate(path)
            for update in ({"training_executed": False}, {"process_exit_code": 1},
                           {"last_checkpoint": {"epoch": 148}}, {"acceptance": {"passed": False}},
                           {"metrics": {"test": {"CER": 25}}}):
                with self.subTest(update=update):
                    path.write_text(json.dumps({**evidence, **update}))
                    with self.assertRaises(ValueError):
                        adapt.verify_user0_gate(path)


if __name__ == "__main__":
    unittest.main()
