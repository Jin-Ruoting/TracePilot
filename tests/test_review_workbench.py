from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from bioml_agent.agent_loop import BioMLResearchAgent
from bioml_agent.review_workbench import (
    build_workbench_summary,
    format_workbench_html,
    load_run,
    workbench_summary_to_dict,
    write_workbench,
)
from bioml_agent.serialization import write_run_json


ROOT = Path(__file__).resolve().parents[1]
PROMOTION_QUESTION = "Should we try retrieval-aware conditioning for promoter expression prediction?"


class ReviewWorkbenchTest(unittest.TestCase):
    def test_workbench_summary_covers_review_readiness(self) -> None:
        run = BioMLResearchAgent().run(PROMOTION_QUESTION)

        summary = build_workbench_summary(run)
        data = workbench_summary_to_dict(summary)

        self.assertTrue(summary.passed)
        self.assertEqual(data["recommendation"], "promote to small controlled screen")
        self.assertEqual(data["review_status"], "requires_approval")
        self.assertEqual(data["evidence_count"], 2)
        self.assertEqual(data["cited_claim_count"], 2)
        self.assertEqual(data["required_action_count"], 3)
        self.assertEqual(data["trace_step_count"], 5)
        self.assertEqual(data["high_fit_evidence_count"], 1)

    def test_workbench_html_renders_product_review_sections(self) -> None:
        run = BioMLResearchAgent().run(PROMOTION_QUESTION)

        text = format_workbench_html(run)

        self.assertIn("<!doctype html>", text)
        self.assertIn("TracePilot Review Workbench", text)
        self.assertIn("Approval Queue", text)
        self.assertIn("Evidence Coverage", text)
        self.assertIn("Claim Grounding", text)
        self.assertIn("Risk Register", text)
        self.assertIn("Tool Trace Sequence", text)
        self.assertIn("requires_approval", text)

    def test_workbench_html_escapes_untrusted_trace_text(self) -> None:
        run = BioMLResearchAgent().run(PROMOTION_QUESTION)
        run.evidence[0] = replace(run.evidence[0], supports="<script>alert('x')</script>")

        text = format_workbench_html(run)

        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", text)
        self.assertNotIn("<script>alert", text)

    def test_write_workbench_outputs_html_and_json(self) -> None:
        run = BioMLResearchAgent().run(PROMOTION_QUESTION)
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / "review-workbench.html"
            json_path = Path(tmp_dir) / "review-workbench.json"
            write_workbench(run, html_path)
            write_workbench(run, json_path, json_output=True)

            self.assertIn("Approval Queue", html_path.read_text(encoding="utf-8"))
            data = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertTrue(data["passed"])
        self.assertEqual(data["trace_step_count"], 5)

    def test_workbench_loads_trace_json(self) -> None:
        run = BioMLResearchAgent().run(PROMOTION_QUESTION)
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            write_run_json(run, trace_path)
            loaded = load_run(trace_path)

        self.assertEqual(loaded.question.text, run.question.text)
        self.assertEqual(len(loaded.trace), len(run.trace))

    def test_workbench_cli_writes_json(self) -> None:
        run = BioMLResearchAgent().run(PROMOTION_QUESTION)
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            output = Path(tmp_dir) / "review-workbench.json"
            write_run_json(run, trace_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.review_workbench",
                    str(trace_path),
                    "--json",
                    "--out",
                    str(output),
                ],
                check=False,
                cwd=ROOT,
                env={"PYTHONPATH": "src"},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(data["passed"])
        self.assertEqual(data["required_action_count"], 3)


if __name__ == "__main__":
    unittest.main()
