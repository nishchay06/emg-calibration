from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "generate_test_user_manifests.py"
)
SPEC = importlib.util.spec_from_file_location("generate_test_user_manifests", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ParseUserConfigTest(unittest.TestCase):
    def test_preserves_split_and_session_order(self) -> None:
        config = """# @package _global_
user: user0
dataset:
  train:
  - user: 123
    session: train-a
  - user: 123
    session: train-b
  val:
  - user: 123
    session: val-a
  test:
  - user: 123
    session: test-a
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user0.yaml"
            path.write_text(config, encoding="utf-8")
            manifest = MODULE.parse_user_config(path, "user0")

        self.assertEqual(manifest.splits["train"], ("train-a", "train-b"))
        self.assertEqual(manifest.splits["val"], ("val-a",))
        self.assertEqual(manifest.splits["test"], ("test-a",))
        self.assertEqual(
            MODULE.manifest_text(manifest).splitlines(),
            [
                "emg2qwerty-data-2021-08/train-a.hdf5",
                "emg2qwerty-data-2021-08/train-b.hdf5",
                "emg2qwerty-data-2021-08/val-a.hdf5",
                "emg2qwerty-data-2021-08/test-a.hdf5",
            ],
        )

    def test_rejects_missing_split(self) -> None:
        config = """dataset:
  train:
  - user: 123
    session: train-a
  val:
  - user: 123
    session: val-a
  test:
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user0.yaml"
            path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing split sessions"):
                MODULE.parse_user_config(path, "user0")

    def test_rejects_duplicate_session(self) -> None:
        config = """dataset:
  train:
  - user: 123
    session: repeated
  val:
  - user: 123
    session: repeated
  test:
  - user: 123
    session: test-a
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user0.yaml"
            path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate sessions"):
                MODULE.parse_user_config(path, "user0")


if __name__ == "__main__":
    unittest.main()
