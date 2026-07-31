from __future__ import annotations

import unittest
from pathlib import Path

from tracepilot.benchmark import render_markdown, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


class TracePilotBenchmarkTest(unittest.TestCase):
    def test_all_generic_cases_pass_full_tracepilot(self) -> None:
        report = run_benchmark(
            ROOT / "benchmarks" / "general_cases.jsonl",
            ROOT / "benchmarks" / "general_sources.jsonl",
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.total_cases, 4)
        self.assertEqual(report.passed_cases, 4)
        for case in report.cases:
            methods = {item.method: item for item in case.methods}
            self.assertFalse(methods["one_shot"].passed)
            self.assertTrue(methods["tracepilot"].passed)
            self.assertEqual(methods["tracepilot"].required_terms_missing, ())
            self.assertEqual(methods["tracepilot"].citation_coverage, 1.0)
        self.assertIn("4/4", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
