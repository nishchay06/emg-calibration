"""Run separately in the analysis environment; export format and semantic plot checks."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from calibration_report import collect_report
from calibration_report_demo import demo_inputs
from plot_calibration_report import build_figure, save_figure
from test_calibration_report import fixture_manifests


class PlotChecks(unittest.TestCase):
    def test_log_axes_full_coordinates_generic_lines_and_synthetic_labels(self):
        manifests = fixture_manifests()
        protocol, index, runs, baselines, refs = demo_inputs(manifests)
        report = collect_report(protocol, index, manifests, runs, baselines, refs, synthetic_demo=True)
        figure = build_figure(report)
        import matplotlib.pyplot as plt
        try:
            self.assertEqual(len(figure.axes), 3)
            for ax in figure.axes:
                self.assertEqual(ax.get_xscale(), "log")
                self.assertEqual(ax.lines[0].get_xdata()[-1], 180)
                self.assertEqual(ax.lines[1].get_linestyle(), "--")
            self.assertTrue(any("NOT RESEARCH RESULTS" in t.get_text() for t in figure.texts))
        finally:
            plt.close(figure)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figures"
            save_figure(report, output)
            for extension, signature in (("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")):
                data = (output / f"cer-vs-minutes.{extension}").read_bytes()
                self.assertTrue(data.startswith(signature))
                self.assertGreater(len(data), 1000)
            self.assertIn("NOT RESEARCH RESULTS", (output / "cer-vs-minutes.svg").read_text())
        report["curve"][0]["test_CER_percent"] += 1
        with self.assertRaises(ValueError): build_figure(report)


if __name__ == "__main__":
    unittest.main()
