"""
Smoke tests for the unified-homotopy-framework demos.

These tests verify that:
  1. The demo scripts run end-to-end without errors.
  2. The reported numerical results stay within published tolerances.
  3. Figure files are produced in the expected location.

The tests do not re-implement the algorithms; they invoke the demos
as black boxes and inspect their output and side effects. Run with:

    python -m unittest discover -s tests

or, equivalently:

    python tests/test_demos.py
"""

import os
import re
import subprocess
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FIG_DIR = os.path.join(REPO_ROOT, "figures")


def _run_demo(script_name, timeout=120):
    """Run a demo script and capture stdout."""
    script = os.path.join(REPO_ROOT, script_name)
    proc = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc


class RBFDemoTest(unittest.TestCase):
    """Smoke test for demo_21paper.py (RBF, 5 Gaussian centers)."""

    @classmethod
    def setUpClass(cls):
        cls.proc = _run_demo("demo_21paper.py")

    def test_exit_code(self):
        self.assertEqual(
            self.proc.returncode, 0,
            f"RBF demo exited with code {self.proc.returncode}.\n"
            f"stderr:\n{self.proc.stderr}"
        )

    def test_completes(self):
        self.assertIn("DEMO COMPLETE", self.proc.stdout)

    def test_residual_reduction(self):
        # Expect "total reduction: <ratio>x" in stdout.
        match = re.search(
            r"total reduction:\s*([0-9.]+)\s*x",
            self.proc.stdout,
        )
        self.assertIsNotNone(
            match, f"Could not find residual reduction line in:\n{self.proc.stdout}"
        )
        ratio = float(match.group(1))
        self.assertGreater(
            ratio, 1.5,
            f"Residual reduction ratio {ratio:.2f} below threshold 1.5",
        )

    def test_simulation_accuracy(self):
        # Expect "Improvement factor: <factor>x" in stdout.
        match = re.search(
            r"Improvement factor:\s*([0-9.]+)\s*x",
            self.proc.stdout,
        )
        self.assertIsNotNone(match)
        factor = float(match.group(1))
        self.assertGreater(
            factor, 3.0,
            f"Improvement-over-linear factor {factor:.2f} below threshold 3.0",
        )

    def test_figures_exist(self):
        for fname in (
            "fig_identification.png",
            "fig_simulation.png",
            "fig_residuals.png",
            "fig_rbf_components.png",
        ):
            with self.subTest(fname=fname):
                self.assertTrue(
                    os.path.isfile(os.path.join(FIG_DIR, fname)),
                    f"Expected figure not produced: {fname}",
                )


class MLPDemoTest(unittest.TestCase):
    """Smoke test for demo_21paper_mlp.py (MLP, 8 sigmoid units)."""

    @classmethod
    def setUpClass(cls):
        cls.proc = _run_demo("demo_21paper_mlp.py")

    def test_exit_code(self):
        self.assertEqual(
            self.proc.returncode, 0,
            f"MLP demo exited with code {self.proc.returncode}.\n"
            f"stderr:\n{self.proc.stderr}"
        )

    def test_completes(self):
        self.assertIn("DEMO COMPLETE", self.proc.stdout)

    def test_simulation_accuracy(self):
        match = re.search(
            r"MLP max error:\s*[0-9.]+\s*\(([0-9.]+)%\)",
            self.proc.stdout,
        )
        self.assertIsNotNone(match)
        rel_err = float(match.group(1))
        self.assertLess(
            rel_err, 5.0,
            f"MLP max error {rel_err:.2f}% above threshold 5%",
        )

    def test_figures_exist(self):
        for fname in (
            "fig_mlp_identification.png",
            "fig_mlp_simulation.png",
            "fig_mlp_residuals.png",
            "fig_mlp_sigmoids.png",
            "fig_mlp_vs_rbf.png",
        ):
            with self.subTest(fname=fname):
                self.assertTrue(
                    os.path.isfile(os.path.join(FIG_DIR, fname)),
                    f"Expected figure not produced: {fname}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
