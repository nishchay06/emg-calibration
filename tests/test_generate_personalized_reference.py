from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_DIR / "scripts" / "generate_personalized_reference.py"
SPEC = importlib.util.spec_from_file_location(
    "generate_personalized_reference", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GeneratePersonalizedReferenceTest(unittest.TestCase):
    def test_parses_lfs_pointer(self) -> None:
        digest = "a" * 64
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{digest}\n"
            "size 12345\n"
        )
        self.assertEqual(
            MODULE.parse_lfs_pointer(pointer),
            {"sha256": digest, "size_bytes": 12345},
        )

    def test_rejects_non_pointer_checkpoint(self) -> None:
        with self.assertRaisesRegex(MODULE.ReferenceError, "Invalid Git LFS pointer"):
            MODULE.parse_lfs_pointer("checkpoint bytes")

    def test_loads_literal_experimental_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.py"
            path.write_text(
                "EXPERIMENTAL_RESULTS = "
                "[{'Model benchmark': 'Example', 'LM': 'No LM', "
                "'Metric': 'Val CER', 'CER': [1, 2, 3, 4, 5, 6, 7, 8]}]\n",
                encoding="utf-8",
            )
            records = MODULE.load_experimental_results(path)
        self.assertEqual(
            MODULE.cer_values(records, "Example", "Val CER"),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        )

    def test_rejects_wrong_cer_count(self) -> None:
        records = [
            {
                "Model benchmark": "Example",
                "LM": "No LM",
                "Metric": "Test CER",
                "CER": [1, 2],
            }
        ]
        with self.assertRaisesRegex(MODULE.ReferenceError, "Expected eight CER"):
            MODULE.cer_values(records, "Example", "Test CER")


if __name__ == "__main__":
    unittest.main()
