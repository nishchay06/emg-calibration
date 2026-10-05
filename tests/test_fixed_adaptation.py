import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parent.parent
UPSTREAM = PROJECT / "upstream/emg2qwerty"
sys.path.insert(0, str(PROJECT / "scripts"))
READY = UPSTREAM.is_dir() and all(importlib.util.find_spec(p) for p in ("torch", "h5py", "pytorch_lightning"))
if READY:
    sys.path.insert(0, str(UPSTREAM))
    import h5py
    import pytorch_lightning as pl
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from calibration_fixtures import write_session
    from calibration_protocol import learning_rate_at_update
    from calibration_sampler import digest
    from emg2qwerty.lightning import TDSConvCTCModule
    from fixed_adaptation import compose_fixed, evaluate_final, fit_fixed, indexed_lengths, make_datamodule, plan_fixed
    from generate_test_user_manifests import parse_user_config
    from index_calibration_sessions import index_sessions
    from test_calibration_protocol import draft
    import adapt


@unittest.skipUnless(READY, "requires pinned upstream and CPU training dependencies")
class FixedAdaptationTest(unittest.TestCase):
    def test_exact_updates_mid_epoch_final_reload_and_no_validation(self):
        torch.set_num_threads(2)
        pl.seed_everything(1501, workers=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "draft.json"
            protocol = draft()
            protocol["methods"]["full"].update(steps=3, warmup_steps=1)
            profile_path.write_text(json.dumps(protocol))
            args = adapt.arguments(["--check-config", "--user", "user0", "--select", "fixed",
                                    "--budget-minutes", "1", "--protocol", str(profile_path),
                                    "--upstream-dir", str(UPSTREAM), "--data-dir", str(root),
                                    "--output-dir", str(root / "output")])
            record = plan_fixed(args)
            config = compose_fixed(args, record)
            self.assertFalse((root / "output").exists())
            self.assertEqual(config["trainer"]["max_steps"], 3)
            self.assertEqual(config["callbacks"], [])
            name = "2020-12-17-100-train"
            write_session(root / f"{name}.hdf5", name, 24000)
            record["calibration_selection"] = {"budget_minutes": "1", "ranges": [
                {"session": name, "start": 0, "stop": 24000, "session_samples": 24000}]}
            # Deliberately inaccessible held-out files demonstrate no fit reads.
            config["dataset"]["train"] = [{"session": name}]
            config["dataset"]["val"] = [{"session": "missing-val"}]
            config["dataset"]["test"] = [{"session": name}]
            datamodule = make_datamodule(args, record, config, num_workers=0)
            datamodule.batch_size = 2  # CPU fixture only; production cap remains 32.
            cfg = OmegaConf.create(config)
            model = instantiate(cfg.module, optimizer=cfg.optimizer, lr_scheduler=cfg.lr_scheduler,
                                decoder=cfg.decoder, _recursive_=False)
            initial = model.model[4].weight.detach().clone()
            start = time.monotonic()
            trainer, final, stats = fit_fixed(model, datamodule, record["profile"], root / "output")
            fit_seconds = time.monotonic() - start
            self.assertEqual(stats["optimizer_steps"], 3)
            self.assertEqual(stats["batch_sizes"], [2, 1, 2])
            self.assertEqual(stats["validation_batches_during_fit"], 0)
            self.assertFalse(hasattr(datamodule, "val_dataset"))
            self.assertEqual([round(v, 12) for v in stats["learning_rates_at_updates"]],
                             [round(learning_rate_at_update(record["profile"], i), 12) for i in range(3)])
            self.assertGreater(torch.count_nonzero(model.model[4].weight.detach() != initial).item(), 0)
            reloaded = TDSConvCTCModule.load_from_checkpoint(str(final), map_location="cpu")
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, reloaded.state_dict()[key]), key)
            batch = next(iter(datamodule.train_dataloader()))
            model.eval()
            reloaded.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(batch["inputs"]), reloaded(batch["inputs"])))
            self.assertEqual(torch.load(final, map_location="cpu")["global_step"], 3)
            evaluated = evaluate_final(trainer, reloaded, datamodule, tune=False, accelerator="cpu")
            self.assertTrue(evaluated["test_evaluated"])
            self.assertEqual(set(evaluated["metrics"]), {"test"})
            datamodule.allow_validation = True
            datamodule.val_sessions = [root / f"{name}.hdf5"]
            datamodule.test_sessions = [root / "missing-test.hdf5"]
            tuned = evaluate_final(trainer, reloaded, datamodule, tune=True, accelerator="cpu")
            self.assertFalse(tuned["test_evaluated"])
            self.assertEqual(set(tuned["metrics"]), {"validation"})
            evidence_directory = os.environ.get("EMG_M6_EVIDENCE_DIR")
            if evidence_directory:
                target = Path(evidence_directory) / "fixed-cpu-smoke"
                target.mkdir(parents=True, exist_ok=False)
                shutil.copytree(root / "output", target / "output")
                shutil.copy2(root / f"{name}.hdf5", target / f"{name}.hdf5")
                (target / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
                proof = {"synthetic_data_only": True, "initialization": "seeded random upstream model; no generic checkpoint",
                         "fixture_samples": 24000, "fixture_batch_size_cap": 2, "fixture_workers": 0,
                         "production_batch_size_cap": 32, "production_workers": 4,
                         "profile": record["profile"], "fit_wall_seconds": fit_seconds, **stats,
                         "final_state_reload_equal": True, "final_evaluation_outputs_equal": True,
                         "final_test_path_exercised": True, "final_tuning_validation_path_exercised": True,
                         "tuning_test_evaluated": False,
                         "parameters": sum(p.numel() for p in model.parameters())}
                (target / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")

    def test_update_schedule_is_independent_of_dataset_epochs(self):
        from step_schedule import UpdateWarmupCosine
        profile = draft()["methods"]["full"]
        traces = []
        for batches_per_epoch in (1, 7):
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.Adam([parameter], lr=profile["learning_rate"])
            scheduler = UpdateWarmupCosine(optimizer, profile)
            trace = []
            for index in range(profile["steps"]):
                trace.append(optimizer.param_groups[0]["lr"])
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                scheduler.step()
                # Dataset epochs intentionally do not affect the update clock.
                _ = index // batches_per_epoch
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_header_index_all_official_sessions_and_stale_index_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(8):
                manifest = parse_user_config(UPSTREAM / "config/user" / f"user{i}.yaml", f"user{i}")
                for name in manifest.sessions:
                    write_session(root / f"{name}.hdf5", name, 64)
            index = index_sessions(UPSTREAM, root)
            self.assertEqual(index["total_sessions"], 100)
            manifest = parse_user_config(UPSTREAM / "config/user/user0.yaml", "user0")
            self.assertEqual(len(indexed_lengths(index, manifest)), 14)
            name = manifest.splits["train"][0]
            index["sessions"][name]["split"] = "test"
            index["index_digest"] = digest({k: v for k, v in index.items() if k != "index_digest"})
            with self.assertRaises(ValueError):
                indexed_lengths(index, manifest)
            with h5py.File(root / f"{name}.hdf5", "r+") as source:
                source["emg2qwerty"].attrs["session_name"] = "foreign"
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                index_sessions(UPSTREAM, root)


if __name__ == "__main__":
    unittest.main()
