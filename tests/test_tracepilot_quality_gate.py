from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tracepilot.quality_gate import render_markdown, run_quality_gate


ROOT = Path(__file__).resolve().parents[1]


class TracePilotQualityGateTest(unittest.TestCase):
    def test_gate_checks_core_runtime_without_recursing_into_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_quality_gate(
                ROOT,
                include_tests=False,
                artifact_dir=tmp,
            )

            self.assertTrue(report.passed)
            self.assertTrue((Path(tmp) / "general-benchmark.json").exists())
            self.assertTrue((Path(tmp) / "workbench.html").exists())
            self.assertIn("tool_contracts_and_policy", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
