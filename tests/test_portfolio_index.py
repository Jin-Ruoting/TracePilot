from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioml_agent.portfolio_index import format_portfolio_index, write_portfolio_index


class PortfolioIndexTest(unittest.TestCase):
    def test_portfolio_index_summarizes_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "traces"
            _write_fixture_artifacts(root)

            text = format_portfolio_index(root)

        self.assertIn("TracePilot Dashboard", text)
        self.assertIn("Reference Catalog Report", text)
        self.assertIn("reference-catalog-report.md", text)
        self.assertIn("Service Interface Report", text)
        self.assertIn("service-interface-report.md", text)
        self.assertIn("Service Runtime Demo", text)
        self.assertIn("service-runtime-demo.md", text)
        self.assertIn("Service API Readiness", text)
        self.assertIn("service-api-readiness.md", text)
        self.assertIn("Service MCP Readiness", text)
        self.assertIn("service-mcp-readiness.md", text)
        self.assertIn("Review Workbench", text)
        self.assertIn("review-workbench.html", text)
        self.assertIn("20/20", text)
        self.assertIn("60/60 cases", text)
        self.assertIn("agent 4/4 vs baseline 0/4", text)
        self.assertIn("promotion-decision-review.html", text)

    def test_portfolio_index_marks_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            text = format_portfolio_index(tmp_dir)

        self.assertIn("not generated", text)
        self.assertIn("Missing", text)

    def test_write_portfolio_index_creates_html_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "traces"
            _write_fixture_artifacts(root)
            output = root / "index.html"
            write_portfolio_index(trace_dir=root, out=output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("<!doctype html>", text)
        self.assertIn("Project Artifacts", text)

    def test_portfolio_index_cli_builds_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "traces"
            _write_fixture_artifacts(root)
            output = root / "index.html"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.portfolio_index",
                    "build",
                    "--trace-dir",
                    str(root),
                    "--out",
                    str(output),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())


def _write_fixture_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "quality-gate-report.json").write_text(
        json.dumps({"passed": True, "passed_count": 20, "total": 20}),
        encoding="utf-8",
    )
    (root / "eval-coverage-report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "passed_cases": 60,
                "total_cases": 60,
                "comparison": {
                    "agent_passed_count": 4,
                    "baseline_passed_count": 0,
                    "total": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "portfolio-demo.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    for name in (
        "quality-gate-report.md",
        "reference-catalog-report.md",
        "service-interface-report.md",
        "service-runtime-demo.md",
        "service-api-readiness.md",
        "service-mcp-readiness.md",
        "eval-coverage-report.md",
        "promotion-decision-review.html",
        "review-workbench.html",
        "promotion-decision-packet.md",
        "promotion-run.json",
        "portfolio-demo.json",
        "reference-catalog-report.json",
        "review-workbench.json",
        "service-interface-report.json",
        "service-runtime-demo.json",
        "service-api-readiness.json",
        "service-mcp-readiness.json",
    ):
        (root / name).write_text("artifact", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
