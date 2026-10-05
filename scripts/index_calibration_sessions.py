#!/usr/bin/env python3
"""Read HDF5 headers to index official sessions; never copy participant signals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calibration_sampler import digest, select_calibration
from generate_test_user_manifests import DEFAULT_USERS, UPSTREAM_COMMIT, parse_user_config, upstream_head


def index_sessions(upstream: Path, data: Path) -> dict:
    import h5py

    if upstream_head(upstream) != UPSTREAM_COMMIT:
        raise ValueError("Upstream pin mismatch")
    lengths, sessions = {}, {}
    for user in DEFAULT_USERS:
        manifest = parse_user_config(upstream / "config/user" / f"{user}.yaml", user)
        for split, names in manifest.splits.items():
            for name in names:
                if name in sessions:
                    raise ValueError("Session assigned to multiple users")
                path = data / f"{name}.hdf5"
                with h5py.File(path, "r") as source:
                    group = source["emg2qwerty"]
                    samples = group["timeseries"]
                    if samples.ndim != 1 or samples.shape[0] <= 0:
                        raise ValueError(f"Invalid timeseries: {name}")
                    if not {"time", "emg_left", "emg_right"} <= set(samples.dtype.names or ()):
                        raise ValueError(f"Missing signal fields: {name}")
                    for field in ("emg_left", "emg_right"):
                        if samples.dtype[field].shape != (16,):
                            raise ValueError(f"Invalid electrode shape: {name}")
                    if group.attrs.get("session_name") != name or group.attrs.get("condition") != "on_keyboard":
                        raise ValueError(f"Session metadata mismatch: {name}")
                    for field in ("keystrokes", "prompts"):
                        if not isinstance(json.loads(group.attrs[field]), list):
                            raise ValueError(f"Invalid label metadata: {name}")
                    lengths[name] = int(samples.shape[0])
                sessions[name] = {"user": user, "split": split, "samples": lengths[name],
                                  "file_bytes": path.stat().st_size}
        select_calibration(manifest, lengths, "full", 1501)
    result = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
              "source": "HDF5 headers from supplied data directory",
              "sessions": sessions, "total_sessions": len(sessions),
              "limitation": "Header/schema index, not a full signal-integrity or file-digest check."}
    result["index_digest"] = digest(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(index_sessions(args.upstream_dir, args.data_dir), indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"Session indexing failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
