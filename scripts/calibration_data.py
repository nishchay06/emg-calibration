"""Budget-bounded views of pinned upstream HDF5 datasets."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from emg2qwerty.data import EMGSessionData, WindowedEMGDataset
from emg2qwerty.lightning import WindowedEMGDataModule


class BoundedSession:
    """Translate all sample reads into one allocated half-open interval."""
    def __init__(self, path, start, stop):
        self.source = EMGSessionData(path)
        self.start, self.stop = start, stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise ValueError("Bounded sessions support contiguous slices only")
        start = 0 if key.start is None else key.start
        stop = len(self) if key.stop is None else min(key.stop, len(self))
        if not 0 <= start <= stop <= len(self):
            raise ValueError("Read outside calibration range")
        return self.source[self.start + start:self.start + stop]

    def ground_truth(self, start_t, end_t):
        # Upstream's unpadded-window timestamps came from bounded signal reads.
        first = float(self.source[self.start:self.start + 1]["time"][0])
        last = float(self.source[self.stop - 1:self.stop]["time"][0])
        if not first <= start_t <= end_t <= last:
            raise ValueError("Labels outside calibration range")
        return self.source.ground_truth(start_t, end_t)


class BoundedWindowedEMGDataset(WindowedEMGDataset):
    def __init__(self, path, span, *, transform, window_length=8000,
                 padding=(1800, 200), jitter=True):
        start, stop, length = span["start"], span["stop"], span["session_samples"]
        if any(type(n) is not int for n in (start, stop, length)) or not 0 <= start < stop <= length:
            raise ValueError("Invalid calibration bounds")
        with EMGSessionData(path) as session:
            if len(session) != length or session.condition != "on_keyboard":
                raise ValueError("Session changed since calibration allocation")
            if session.session_name != span["session"]:
                raise ValueError("Session identity mismatch")
        self.hdf5_path = Path(path)
        self.start, self.stop = start, stop
        self.session_length = stop - start
        self.window_length = window_length
        self.stride = window_length
        self.left_padding, self.right_padding = padding
        self.jitter = jitter
        self.transform = transform
        if window_length <= 0 or any(p < 0 for p in padding):
            raise ValueError("Invalid window or context length")

    def __len__(self):
        # Unlike upstream's minimum length of one, short fragments yield no
        # complete windows. Never read beyond a fragment to make it trainable.
        return self.session_length // self.window_length

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Training window out of range")
        if not hasattr(self, "session"):
            self.session = BoundedSession(self.hdf5_path, self.start, self.stop)
        return super().__getitem__(index)


def training_dataset(data_dir, selection, transform, window_length=8000, padding=(1800, 200)):
    datasets = []
    for span in selection["ranges"]:
        path = Path(data_dir) / f"{span['session']}.hdf5"
        if selection["budget_minutes"] == "full":
            with EMGSessionData(path) as session:
                if (span["start"], span["stop"], span["session_samples"]) != (0, len(session), len(session)):
                    raise ValueError("Full selection must match the complete source session")
            dataset = WindowedEMGDataset(path, window_length=window_length, padding=padding,
                                         jitter=True, transform=transform)
        else:
            dataset = BoundedWindowedEMGDataset(path, span, transform=transform,
                                                window_length=window_length, padding=padding)
        if len(dataset):
            datasets.append(dataset)
    if not datasets:
        raise ValueError("Calibration allocation has no complete training windows")
    return ConcatDataset(datasets)


class CalibrationDataModule(WindowedEMGDataModule):
    def __init__(self, *, data_dir, selection, allow_validation=False, **kwargs):
        super().__init__(**kwargs)
        self.data_dir, self.selection = Path(data_dir), selection
        self.allow_validation = allow_validation

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = training_dataset(self.data_dir, self.selection,
                                                 self.train_transform, self.window_length, self.padding)
        if stage == "validate":
            if not self.allow_validation:
                raise ValueError("Validation is disabled for ordinary calibration adaptation")
            self.val_dataset = ConcatDataset([
                WindowedEMGDataset(path, transform=self.val_transform,
                                   window_length=self.window_length, padding=self.padding, jitter=False)
                for path in self.val_sessions])
        if stage == "test":
            self.test_dataset = ConcatDataset([
                WindowedEMGDataset(path, transform=self.test_transform,
                                   window_length=None, padding=(0, 0), jitter=False)
                for path in self.test_sessions])

    def _loader(self, dataset, *, shuffle=False, batch_size=None):
        return DataLoader(dataset, batch_size=batch_size or self.batch_size,
                          shuffle=shuffle, num_workers=self.num_workers,
                          collate_fn=WindowedEMGDataset.collate, drop_last=False,
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        # Even tuning evaluates validation only after fixed-step fitting.
        if not hasattr(self, "val_dataset"):
            return []
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset, batch_size=1)
