"""Small synthetic HDF5 fixtures; contains no participant recordings."""
import json

import h5py
import numpy as np


def write_session(path, name, samples=16000, *, key_positions=None, indexed_signal=False):
    dtype = [("time", np.float64), ("emg_left", np.float32, (16,)), ("emg_right", np.float32, (16,))]
    raw = np.zeros(samples, dtype=dtype)
    raw["time"] = np.arange(samples) / 2000
    if indexed_signal:
        raw["emg_left"] = np.arange(samples)[:, None]
        raw["emg_right"] = np.arange(samples)[:, None]
    else:
        rng = np.random.default_rng(1501)
        raw["emg_left"] = rng.standard_normal((samples, 16))
        raw["emg_right"] = rng.standard_normal((samples, 16))
    if key_positions is None:
        key_positions = [(i, "a") for i in range(1000, samples, 4000)]
    keys = [{"key": key, "start": position / 2000, "end": (position + 1) / 2000}
            for position, key in key_positions]
    with h5py.File(path, "w") as source:
        group = source.create_group("emg2qwerty")
        group.create_dataset("timeseries", data=raw, compression="gzip")
        group.attrs.update(session_name=name, user="synthetic", condition="on_keyboard",
                           duration_mins=samples / 120000,
                           keystrokes=json.dumps(keys), prompts="[]")
