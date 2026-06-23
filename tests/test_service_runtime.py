from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioml_agent.quality_gate import PROMOTION_QUESTION
from bioml_agent.service_runtime import (
    ServiceSessionState,
    build_service_runtime,
    format_service_runtime_demo,
    run_service_runtime_demo,
    service_runtime_demo_to_dict,
    write_service_runtime_demo,
)


class ServiceRuntimeTest(unittest.TestCase):
    def test_runtime_executes_contract_derived_routes_with_session_state(self) -> None:
        runtime = build_service_runtime()
        state = ServiceSessionState()

        search = runtime.invoke(
            "POST",
            "/tools/search_literature",
            {"query": PROMOTION_QUESTION, "top_k": 4},
            state,
        )
        context = runtime.invoke(
            "POST",
            "/tools/read_experiment_context",
            {"project_name": "EpiCETRA"},
            state,
        )
        score = runtime.invoke("POST", "/tools/score_transfer_fit", {}, state)
        draft = runtime.invoke("POST", "/tools/draft_experiment_plan", {}, state)
        review = runtime.invoke("POST", "/tools/create_review_checkpoint", {}, state)

        self.assertTrue(search.ok)
        self.assertTrue(context.ok)
        self.assertTrue(score.ok)
        self.assertTrue(draft.ok)
        self.assertTrue(review.ok)
        self.assertTrue(review.approval_required)
        self.assertEqual(len(state.trace), 5)
        self.assertIn("review", review.state_after)

    def test_runtime_rejects_invalid_request_before_tool_call(self) -> None:
        runtime = build_service_runtime()
        state = ServiceSessionState()

        result = runtime.invoke("POST", "/tools/search_literature", {"query": PROMOTION_QUESTION}, state)

        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 422)
        self.assertIn("missing required fields", result.error)
        self.assertEqual(len(state.trace), 0)

    def test_runtime_rejects_state_dependent_tool_without_session_state(self) -> None:
        runtime = build_service_runtime()
        state = ServiceSessionState()

        result = runtime.invoke("POST", "/tools/draft_experiment_plan", {}, state)

        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 409)
        self.assertIn("missing session state", result.error)
        self.assertEqual(len(state.trace), 0)

    def test_service_runtime_demo_serializes_markdown_and_json(self) -> None:
        report = run_service_runtime_demo()
        data = service_runtime_demo_to_dict(report)
        text = format_service_runtime_demo(report)

        self.assertTrue(report.passed)
        self.assertEqual(report.route_count, 5)
        self.assertEqual(report.invocation_count, 5)
        self.assertEqual(report.trace_event_count, 5)
        self.assertTrue(report.approval_required)
        self.assertIn("Service Runtime Demo", text)
        self.assertEqual(data["final_state_keys"][-1], "trace")

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "service-runtime-demo.md"
            json_path = Path(tmp_dir) / "service-runtime-demo.json"
            write_service_runtime_demo(md_path, report=report)
            write_service_runtime_demo(json_path, report=report, json_output=True)
            written_md = md_path.read_text(encoding="utf-8")
            written_json = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("Invocation Trace", written_md)
        self.assertTrue(written_json["passed"])

    def test_cli_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "service-runtime-demo.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.service_runtime",
                    "--json",
                    "--out",
                    str(output),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                capture_output=True,
                text=True,
            )
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(data["passed"])
        self.assertEqual(data["invocation_count"], 5)


if __name__ == "__main__":
    unittest.main()
