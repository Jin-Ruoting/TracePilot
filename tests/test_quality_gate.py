from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bioml_agent.quality_gate import (
    CheckSpec,
    CommandResult,
    default_check_specs,
    format_quality_gate_result,
    quality_gate_to_dict,
    run_quality_gate,
    write_quality_gate_result,
)


class QualityGateTest(unittest.TestCase):
    def test_default_check_specs_cover_core_portfolio_gates(self) -> None:
        specs = default_check_specs(python="python")
        names = [spec.name for spec in specs]

        self.assertEqual(
            names,
            [
                "compile",
                "unit_tests",
                "tool_contracts",
                "eval_coverage_markdown",
                "eval_coverage_json",
                "promotion_artifacts",
                "review_workbench_html",
                "review_workbench_json",
                "portfolio_demo",
                "reference_catalog_markdown",
                "reference_catalog_json",
                "service_interface_markdown",
                "service_interface_json",
                "service_runtime_markdown",
                "service_runtime_json",
                "service_api_markdown",
                "service_api_json",
                "service_mcp_markdown",
                "service_mcp_json",
                "portfolio_index",
            ],
        )
        promotion = next(spec for spec in specs if spec.name == "promotion_artifacts")
        self.assertIn("traces/promotion-decision-review.html", promotion.artifacts)
        self.assertIn("--report-html-out", promotion.command)
        workbench = next(spec for spec in specs if spec.name == "review_workbench_html")
        self.assertIn("bioml_agent.review_workbench", workbench.command)
        self.assertEqual(workbench.artifacts, ("traces/review-workbench.html",))
        index = next(spec for spec in specs if spec.name == "portfolio_index")
        self.assertIn("bioml_agent.portfolio_index", index.command)
        self.assertEqual(
            index.artifacts,
            (
                "traces/index.html",
                "traces/reference-catalog-report.md",
                "traces/review-workbench.html",
                "traces/service-interface-report.md",
                "traces/service-runtime-demo.md",
                "traces/service-api-readiness.md",
                "traces/service-mcp-readiness.md",
            ),
        )
        reference = next(spec for spec in specs if spec.name == "reference_catalog_markdown")
        self.assertIn("bioml_agent.reference_catalog", reference.command)
        self.assertEqual(reference.artifacts, ("traces/reference-catalog-report.md",))
        service = next(spec for spec in specs if spec.name == "service_interface_markdown")
        self.assertIn("bioml_agent.service_interface", service.command)
        self.assertEqual(service.artifacts, ("traces/service-interface-report.md",))
        runtime = next(spec for spec in specs if spec.name == "service_runtime_markdown")
        self.assertIn("bioml_agent.service_runtime", runtime.command)
        self.assertEqual(runtime.artifacts, ("traces/service-runtime-demo.md",))
        api = next(spec for spec in specs if spec.name == "service_api_markdown")
        self.assertIn("bioml_agent.service_api", api.command)
        self.assertEqual(api.artifacts, ("traces/service-api-readiness.md",))
        mcp = next(spec for spec in specs if spec.name == "service_mcp_markdown")
        self.assertIn("bioml_agent.service_mcp", mcp.command)
        self.assertEqual(mcp.artifacts, ("traces/service-mcp-readiness.md",))

    def test_quality_gate_passes_with_successful_runner_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "artifact.txt").write_text("ok", encoding="utf-8")
            specs = (CheckSpec(name="fake", command=("python", "--version"), artifacts=("artifact.txt",)),)

            result = run_quality_gate(specs=specs, cwd=root, runner=_successful_runner)

        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 1)
        self.assertEqual(result.checks[0].stdout_tail, "ok")

    def test_quality_gate_fails_when_command_fails(self) -> None:
        specs = (CheckSpec(name="fake", command=("python", "--bad")),)

        result = run_quality_gate(specs=specs, runner=_failing_runner)

        self.assertFalse(result.passed)
        self.assertEqual(result.checks[0].returncode, 2)
        self.assertIn("bad flag", result.checks[0].stderr_tail)

    def test_quality_gate_fails_when_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            specs = (CheckSpec(name="artifact", command=("python", "--version"), artifacts=("missing.txt",)),)

            result = run_quality_gate(specs=specs, cwd=tmp_dir, runner=_successful_runner)

        self.assertFalse(result.passed)
        self.assertEqual(result.checks[0].missing_artifacts, ("missing.txt",))

    def test_quality_gate_can_fail_fast(self) -> None:
        specs = (
            CheckSpec(name="first", command=("python", "--bad")),
            CheckSpec(name="second", command=("python", "--version")),
        )

        result = run_quality_gate(specs=specs, runner=_failing_runner, stop_on_failure=True)

        self.assertEqual(result.total, 1)
        self.assertFalse(result.passed)

    def test_quality_gate_formats_and_serializes(self) -> None:
        specs = (CheckSpec(name="fake", command=("python", "--version")),)
        result = run_quality_gate(specs=specs, runner=_successful_runner)
        text = format_quality_gate_result(result)
        data = quality_gate_to_dict(result)

        self.assertIn("TracePilot Quality Gate", text)
        self.assertTrue(data["passed"])
        self.assertEqual(data["checks"][0]["name"], "fake")

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "quality.md"
            json_path = Path(tmp_dir) / "quality.json"
            write_quality_gate_result(result, md_path)
            write_quality_gate_result(result, json_path, json_output=True)
            written_md = md_path.read_text(encoding="utf-8")
            written_json = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("Quality Gate", written_md)
        self.assertTrue(written_json["passed"])


def _successful_runner(command: tuple[str, ...], cwd: Path) -> CommandResult:
    return CommandResult(returncode=0, stdout="ok\n", stderr="")


def _failing_runner(command: tuple[str, ...], cwd: Path) -> CommandResult:
    return CommandResult(returncode=2, stdout="", stderr="bad flag\n")


if __name__ == "__main__":
    unittest.main()
