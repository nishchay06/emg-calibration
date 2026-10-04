from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_DIR / "scripts" / "verify_personalized_checkpoint.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_personalized_checkpoint", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VerifyPersonalizedCheckpointTest(unittest.TestCase):
    def test_verifies_filename_size_and_sha256(self) -> None:
        payload = b"checkpoint fixture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user0.ckpt"
            path.write_bytes(payload)
            MODULE.verify_checkpoint(
                path,
                {
                    "filename": "user0.ckpt",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            )

    def test_rejects_wrong_size_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user0.ckpt"
            path.write_bytes(b"short")
            with self.assertRaisesRegex(MODULE.VerificationError, "Expected 10 bytes"):
                MODULE.verify_checkpoint(
                    path,
                    {
                        "filename": "user0.ckpt",
                        "size_bytes": 10,
                        "sha256": "0" * 64,
                    },
                )

    def test_loads_pinned_finetuned_identity(self) -> None:
        expected = MODULE.load_checkpoint_reference(
            MODULE.DEFAULT_REFERENCE_PATH,
            "finetuned",
            "user0",
        )
        self.assertEqual(expected["size_bytes"], 63610538)
        self.assertEqual(
            expected["sha256"],
            "b2335e1da5b3eeaf8693e4431b806e8d4dda553c37cf844414bf53da249a1474",
        )


if __name__ == "__main__":
    unittest.main()
