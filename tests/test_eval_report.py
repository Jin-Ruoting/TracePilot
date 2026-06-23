from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioml_agent.eval_report import (
    eval_report_to_dict,
    format_eval_coverage_report,
    run_eval_coverage_report,
    write_eval_coverage_report,
)


class EvalReportTest(unittest.TestCase):
    def test_eval_coverage_report_summarizes_all_default_corpora(self) -> None:
        report = run_eval_coverage_report()

        self.assertTrue(report.passed)
        self.assertEqual(report.total_cases, 60)
        self.assertEqual(report.passed_cases, 60)
        self.assertEqual(len(report.corpora), 6)
        self.assertIsNotNone(report.comparison)
        self.assertEqual(report.comparison.agent_passed_count, 4)
        self.assertEqual(report.comparison.baseline_passed_count, 0)

    def test_eval_coverage_report_tracks_guardrails_and_families(self) -> None:
        report = run_eval_coverage_report()
        guardrails = {item.name: item for item in report.guardrails}
        families = {item.family: item for item in report.families}

        self.assertTrue(guardrails["source poisoning"].covered)
        self.assertIn("poisoned_retrieval_source", guardrails["source poisoning"].cases)
        self.assertTrue(guardrails["claim citation binding"].covered)
        self.assertIn("abstention", families)
        self.assertIn("real_literature", families)
        self.assertIn("source_poisoning", families)
        self.assertGreaterEqual(families["real_literature"].total, 13)
        self.assertGreaterEqual(families["retrieval"].total, 10)
        self.assertGreaterEqual(families["graph"].total, 10)
        self.assertGreaterEqual(families["moe"].total, 10)
        self.assertGreaterEqual(families["contrastive"].total, 10)

    def test_eval_coverage_report_formats_markdown(self) -> None:
        report = run_eval_coverage_report()
        text = format_eval_coverage_report(report)

        self.assertIn("TracePilot Eval Coverage Report", text)
        self.assertIn("Corpus Coverage", text)
        self.assertIn("Guardrail Coverage", text)
        self.assertIn("Baseline Delta", text)
        self.assertIn("source_poisoning_guarded", text)

    def test_eval_coverage_report_serializes_to_jsonable_dict(self) -> None:
        report = run_eval_coverage_report()
        data = eval_report_to_dict(report)

        self.assertTrue(data["passed"])
        self.assertEqual(data["total_cases"], 60)
        self.assertIn("guardrails", data)
        self.assertEqual(data["comparison"]["agent_passed_count"], 4)

    def test_write_eval_coverage_report_creates_markdown_file(self) -> None:
        report = run_eval_coverage_report()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "eval-coverage.md"
            write_eval_coverage_report(report, output_path)
            written = output_path.read_text(encoding="utf-8")

        self.assertIn("Eval Coverage Report", written)
        self.assertIn("Check Coverage", written)

    def test_eval_report_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "eval-coverage.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.eval_report",
                    "--json",
                    "--out",
                    str(output_path),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertTrue(data["passed"])
        self.assertEqual(data["passed_cases"], 60)


if __name__ == "__main__":
    unittest.main()
