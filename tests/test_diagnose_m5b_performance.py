import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_m5b_performance as probe


class DiagnosticGuardsTest(unittest.TestCase):
    def test_throttling_deltas_and_missing_telemetry(self):
        def snap(value):
            return {"cgroups": [{"resolved_path": "/sys/fs/cgroup", "files": {"cpu.stat": value}}]}
        self.assertEqual(probe.throttling_delta(snap("nr_throttled 7\nthrottled_usec 50"),
            snap("nr_throttled 9\nthrottled_usec 150")),
            {"/sys/fs/cgroup": {"nr_throttled": 2, "throttled_usec": 100}})
        self.assertEqual(probe.throttling_delta(snap({"unavailable": "missing"}), snap("nr_throttled 9")), {})

    def test_dry_run_creates_nothing_without_training_packages(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "fresh"
            result = subprocess.run([sys.executable, str(Path(probe.__file__)), "--dry-run",
                "--upstream-dir", root, "--data-dir", root, "--checkpoint", root,
                "--output-dir", str(output)], capture_output=True, text=True, check=True)
            self.assertFalse(output.exists())
            self.assertFalse(json.loads(result.stdout)["full_training"])

    def test_refuses_existing_and_repository_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                probe.validate_output(Path(root))
        with self.assertRaises(ValueError):
            probe.validate_output(probe.adapt.PROJECT / "results" / "unsafe-new-probe")

    def test_timeout_kills_and_reaps_worker(self):
        with tempfile.TemporaryDirectory() as root:
            start = time.monotonic()
            result = probe.supervise([sys.executable, "-c", "import time; time.sleep(30)"],
                Path(root) / "console.log", time.time() + 0.15)
            self.assertTrue(result["timed_out"])
            self.assertTrue(result["worker_reaped"])
            self.assertLess(time.monotonic() - start, 3)
            self.assertFalse(result["pod_billing_stopped"])

    def test_deadline_and_success(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / "console.log"
            for deadline in (time.time() - 1, time.time() + 700):
                with self.assertRaises(ValueError):
                    probe.supervise([sys.executable, "-c", "pass"], log, deadline)
            result = probe.supervise([sys.executable, "-c", "print('done')"], log, time.time() + 5)
            self.assertEqual(result["worker_exit_code"], 0)
            self.assertEqual(log.read_text().strip(), "done")


if __name__ == "__main__":
    unittest.main()
