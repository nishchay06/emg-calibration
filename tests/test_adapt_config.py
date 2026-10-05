"""Real Hydra composition tests; require m5b-config.txt and pinned checkout."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
import adapt

UPSTREAM = PROJECT / "upstream/emg2qwerty"
READY = UPSTREAM.is_dir() and importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(READY, "requires pinned upstream and requirements/m5b-config.txt")
class AdaptCompositionTest(unittest.TestCase):
    def test_all_users_and_budgets_compose_fixed_without_outputs(self):
        from calibration_sampler import BUDGETS, digest
        from fixed_adaptation import compose_fixed, plan_fixed

        protocol = {"schema_version": 1, "upstream_commit": adapt.UPSTREAM_COMMIT,
                    "status": "draft", "selection": "final", "schedule": "update-warmup-cosine-v1",
                    "methods": {"full": {"steps": 5, "learning_rate": 0.001, "warmup_steps": 2,
                                          "warmup_start_lr": 1e-8, "minimum_lr": 1e-6}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "protocol.json"
            path.write_text(json.dumps(protocol))
            for index in range(8):
                for budget in BUDGETS:
                    with self.subTest(user=index, budget=budget):
                        args = adapt.arguments(["--user", f"user{index}", "--select", "fixed",
                                                "--budget-minutes", budget, "--protocol", str(path),
                                                "--check-config", "--upstream-dir", str(UPSTREAM),
                                                "--data-dir", str(root / "data"),
                                                "--output-dir", str(root / f"user{index}-{budget}")])
                        record = plan_fixed(args)
                        config = compose_fixed(args, record)
                        self.assertEqual(record["profile_digest"], digest(protocol["methods"]["full"]))
                        self.assertEqual(config["trainer"]["max_steps"], 5)
                        self.assertEqual(config["trainer"]["limit_val_batches"], 0)
                        self.assertEqual(config["trainer"]["num_sanity_val_steps"], 0)
                        self.assertEqual(config["lr_scheduler"]["interval"], "step")
                        self.assertEqual(config["callbacks"], [])
            self.assertEqual(list(root.iterdir()), [path])

    def test_all_users_compose_exact_splits_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(8):
                with self.subTest(user=index):
                    args = adapt.arguments(["--user", f"user{index}", "--check-config",
                                            "--upstream-dir", str(UPSTREAM), "--data-dir", str(root / "data"),
                                            "--output-dir", str(root / f"user{index}"), "--accelerator", "gpu"])
                    config = adapt.compose_config(args)
                    self.assertEqual(config["seed"], 1501)
                    self.assertEqual(config["trainer"]["max_epochs"], 150)
                    self.assertEqual(config["trainer"]["accelerator"], "gpu")
                    self.assertEqual(config["transforms"]["test"], config["transforms"]["val"])
            self.assertEqual(list(root.iterdir()), [])

    def test_paths_with_spaces_and_hydra_punctuation_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data space,=[x]@"quote'
            args = adapt.arguments(["--user", "user5", "--check-config", "--upstream-dir", str(UPSTREAM),
                                    "--data-dir", str(data), "--output-dir", str(root / "output space")])
            config = adapt.compose_config(args)
            self.assertEqual(config["dataset"]["root"], str(data))
            self.assertEqual(config["callbacks"][1]["dirpath"], str(root / "output space/checkpoints"))
            self.assertEqual(list(root.iterdir()), [])

    def test_upstream_mtime_resume_is_confirmed_and_run_plan_is_isolated(self):
        train = (UPSTREAM / "emg2qwerty/train.py").read_text()
        utils = (UPSTREAM / "emg2qwerty/utils.py").read_text()
        self.assertIn('Path.cwd().joinpath("checkpoints")', train)
        self.assertIn("p.stat().st_mtime", utils)


if __name__ == "__main__":
    if not READY:
        raise SystemExit("Install m5b-config.txt and check out pinned upstream before this explicit check")
    unittest.main()
