import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
UPSTREAM = PROJECT / "upstream/emg2qwerty"
sys.path.insert(0, str(PROJECT / "scripts"))
READY = UPSTREAM.is_dir() and all(importlib.util.find_spec(p) for p in ("torch", "h5py", "pytorch_lightning"))
if READY:
    sys.path.insert(0, str(UPSTREAM))
    import numpy as np
    import torch
    from calibration_data import BoundedWindowedEMGDataset, CalibrationDataModule, training_dataset
    from calibration_fixtures import write_session
    from emg2qwerty.data import LabelData, WindowedEMGDataset


@unittest.skipUnless(READY, "requires pinned upstream and CPU training dependencies")
class CalibrationDataTest(unittest.TestCase):
    def test_padding_jitter_and_labels_stay_inside_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            name = "2020-12-17-100-train"
            path = Path(directory) / f"{name}.hdf5"
            write_session(path, name, 40000, indexed_signal=True,
                          key_positions=[(9999, "x"), (10000, "a"), (18000, "b"),
                                         (25999, "c"), (26000, "z")])
            span = {"session": name, "start": 10000, "stop": 26000, "session_samples": 40000}
            transform = lambda raw: torch.tensor(raw["emg_left"][:, :1].copy())
            dataset = BoundedWindowedEMGDataset(path, span, transform=transform)
            excluded = set(LabelData.from_str("xz").labels)
            for seed in range(20):
                np.random.seed(seed)
                for index in range(len(dataset)):
                    inputs, labels = dataset[index]
                    self.assertGreaterEqual(float(inputs.min()), 10000)
                    self.assertLess(float(inputs.max()), 26000)
                    self.assertFalse(excluded.intersection(labels.tolist()))
            dataset.jitter = False
            self.assertEqual(dataset[0][1].tolist(), LabelData.from_str("a").labels.tolist())
            self.assertEqual(dataset[1][1].tolist(), LabelData.from_str("bc").labels.tolist())
            with self.assertRaises(IndexError):
                dataset[2]
            with self.assertRaises(ValueError):
                dataset.session[-1:10]

    def test_full_delegates_exactly_to_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root, name = Path(directory), "2020-12-17-100-train"
            path = root / f"{name}.hdf5"
            write_session(path, name, 17000)
            transform = lambda raw: torch.tensor(raw["emg_left"].copy())
            selection = {"budget_minutes": "full", "ranges": [
                {"session": name, "start": 0, "stop": 17000, "session_samples": 17000}]}
            actual = training_dataset(root, selection, transform)
            expected = WindowedEMGDataset(path, window_length=8000, padding=(1800, 200),
                                           jitter=True, transform=transform)
            self.assertIs(type(actual.datasets[0]), WindowedEMGDataset)
            self.assertEqual(len(actual), len(expected))
            for index in range(len(expected)):
                np.random.seed(1501)
                left = actual[index]
                np.random.seed(1501)
                right = expected[index]
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))

    def test_small_batches_no_heldout_reads_and_no_complete_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root, name = Path(directory), "2020-12-17-100-train"
            write_session(root / f"{name}.hdf5", name, 16000)
            selection = {"budget_minutes": "1", "ranges": [
                {"session": name, "start": 0, "stop": 16000, "session_samples": 16000}]}
            transform = lambda raw: torch.tensor(raw["emg_left"].copy())
            module = CalibrationDataModule(data_dir=root, selection=selection, window_length=8000,
                                           padding=(1800, 200), batch_size=32, num_workers=0,
                                           train_sessions=[root / f"{name}.hdf5"],
                                           val_sessions=[root / "missing-val.hdf5"],
                                           test_sessions=[root / "missing-test.hdf5"],
                                           train_transform=transform, val_transform=transform, test_transform=transform)
            module.setup("fit")
            self.assertEqual(len(next(iter(module.train_dataloader()))["input_lengths"]), 2)
            self.assertEqual(module.val_dataloader(), [])
            with self.assertRaisesRegex(ValueError, "Validation"):
                module.setup("validate")
            selection["ranges"][0]["stop"] = 7999
            with self.assertRaisesRegex(ValueError, "no complete"):
                training_dataset(root, selection, transform)

    def test_stale_lengths_invalid_bounds_and_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, name = Path(directory), "2020-12-17-100-train"
            path = root / f"{name}.hdf5"
            write_session(path, name, 16000)
            good = {"session": name, "start": 0, "stop": 8000, "session_samples": 16000}
            for update in ({"start": -1}, {"stop": 16001}, {"session_samples": 16001}, {"session": "foreign"}):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    BoundedWindowedEMGDataset(path, {**good, **update}, transform=lambda x: x)


if __name__ == "__main__":
    unittest.main()
