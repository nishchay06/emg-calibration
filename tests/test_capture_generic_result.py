from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_DIR / "scripts" / "capture_generic_result.py"
FIXTURE_PATH = PROJECT_DIR / "tests" / "fixtures" / "user0-completed-console.txt"
SPEC = importlib.util.spec_from_file_location("capture_generic_result", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CaptureGenericResultTest(unittest.TestCase):
    def test_parses_completed_lightning_output(self) -> None:
        metrics = MODULE.parse_metrics(FIXTURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(metrics["val"]["CER"], 60.08256530761719)
        self.assertEqual(metrics["test"]["CER"], 61.50963592529297)
        self.assertEqual(metrics["test"]["loss"], 2.8309335708618164)

    def test_rejects_incomplete_output(self) -> None:
        with self.assertRaisesRegex(MODULE.CaptureError, "Missing completed metrics"):
            MODULE.parse_metrics("'val/CER': 60.0")

    def test_cer_acceptance_is_inclusive_at_pinned_threshold(self) -> None:
        self.assertTrue(MODULE.cer_delta_is_accepted(0.10))
        self.assertTrue(MODULE.cer_delta_is_accepted(-0.10))
        self.assertFalse(MODULE.cer_delta_is_accepted(0.100001))

    def test_cli_writes_structured_result_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "user0",
                str(FIXTURE_PATH),
                str(output_path),
            ]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["user"], "user0")
            self.assertEqual(result["reference_CER"]["validation"], 60.07)
            self.assertAlmostEqual(
                result["CER_delta_percentage_points"]["test"],
                0.02963592529297,
            )
            self.assertEqual(
                result["acceptance"],
                {
                    "maximum_absolute_CER_delta_percentage_points": 0.1,
                    "passed": True,
                    "test": True,
                    "validation": True,
                },
            )

            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 1)
            self.assertIn("Refusing to overwrite", second.stderr)

    def test_cli_writes_failed_result_then_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            console_path = directory_path / "failed-console.txt"
            output_path = directory_path / "result.json"
            console_path.write_text(
                FIXTURE_PATH.read_text(encoding="utf-8").replace(
                    "60.08256530761719",
                    "60.5",
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "user0",
                    str(console_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("Acceptance failed for user0", completed.stderr)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(result["acceptance"]["validation"])
            self.assertTrue(result["acceptance"]["test"])
            self.assertFalse(result["acceptance"]["passed"])


if __name__ == "__main__":
    unittest.main()
