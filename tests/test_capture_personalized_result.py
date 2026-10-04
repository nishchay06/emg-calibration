from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "capture_personalized_result.py"
FIXTURE_PATH = PROJECT_DIR / "tests" / "fixtures" / "user0-completed-console.txt"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("capture_personalized_result", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def personalized_console(validation_cer: str, test_cer: str) -> str:
    return (
        FIXTURE_PATH.read_text(encoding="utf-8")
        .replace("60.08256530761719", validation_cer)
        .replace("61.50963592529297", test_cer)
    )


class CapturePersonalizedResultTest(unittest.TestCase):
    def test_builds_accepted_finetuned_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            console = Path(directory) / "console.log"
            console.write_text(
                personalized_console("17.96", "20.57"),
                encoding="utf-8",
            )
            result = MODULE.build_result("finetuned", "user0", console)

        self.assertTrue(result["acceptance"]["passed"])
        self.assertEqual(result["benchmark"], "finetuned")
        self.assertEqual(result["upstream_benchmark"], "Personalized (finetuned)")
        self.assertEqual(result["reference_CER"]["validation"], 17.96)
        self.assertEqual(
            result["checkpoint"]["sha256"],
            "b2335e1da5b3eeaf8693e4431b806e8d4dda553c37cf844414bf53da249a1474",
        )

    def test_loads_randominit_reference(self) -> None:
        reference = MODULE.load_reference(
            MODULE.DEFAULT_REFERENCE_PATH,
            "randominit",
            "user5",
        )
        self.assertEqual(reference["validation_CER"], 9.363)
        self.assertEqual(reference["test_CER"], 7.648)

    def test_rejects_result_outside_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            console = Path(directory) / "console.log"
            console.write_text(
                personalized_console("18.2", "20.57"),
                encoding="utf-8",
            )
            result = MODULE.build_result("finetuned", "user0", console)
        self.assertFalse(result["acceptance"]["validation"])
        self.assertTrue(result["acceptance"]["test"])
        self.assertFalse(result["acceptance"]["passed"])


if __name__ == "__main__":
    unittest.main()
