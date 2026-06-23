from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioml_agent.service_mcp import (
    build_service_mcp_readiness,
    format_service_mcp_readiness,
    service_mcp_readiness_to_dict,
    write_service_mcp_readiness,
)


class ServiceMcpTest(unittest.TestCase):
    def test_mcp_readiness_derives_tool_bindings_from_service_interface(self) -> None:
        report = build_service_mcp_readiness()
        data = service_mcp_readiness_to_dict(report)
        bindings = {binding["name"]: binding for binding in data["tool_bindings"]}

        self.assertTrue(report.passed)
        self.assertEqual(report.tool_binding_count, 5)
        self.assertEqual(report.resource_count, 2)
        self.assertEqual(report.prompt_count, 1)
        self.assertEqual(report.transport_count, 2)
        self.assertIn("search_literature", bindings)
        self.assertIn("create_review_checkpoint", bindings)
        self.assertTrue(bindings["search_literature"]["annotations"]["readOnlyHint"])
        self.assertFalse(bindings["search_literature"]["annotations"]["destructiveHint"])
        self.assertEqual(
            bindings["create_review_checkpoint"]["state_dependencies"],
            ["plan", "evidence", "context"],
        )

    def test_mcp_readiness_includes_resources_prompts_and_transports(self) -> None:
        report = build_service_mcp_readiness()
        data = service_mcp_readiness_to_dict(report)
        resources = {resource["uri"]: resource for resource in data["resources"]}
        prompts = {prompt["name"]: prompt for prompt in data["prompts"]}
        transports = {transport["name"]: transport for transport in data["transports"]}
        text = format_service_mcp_readiness(report)

        self.assertIn("bioml://service/interface", resources)
        self.assertIn("bioml://service/runtime-demo", resources)
        self.assertTrue(resources["bioml://service/interface"]["read_only"])
        self.assertIn("bioml_experiment_review", prompts)
        self.assertEqual(prompts["bioml_experiment_review"]["arguments"], ["question", "focus"])
        self.assertIn("stdio", transports)
        self.assertIn("streamable-http", transports)
        self.assertIn("python -m pip install -e '.[mcp]'", text)

    def test_optional_dependency_status_is_explicit(self) -> None:
        report = build_service_mcp_readiness()
        data = service_mcp_readiness_to_dict(report)

        self.assertEqual([item["name"] for item in data["optional_dependencies"]], ["mcp"])
        self.assertIn("runnable_now", data)
        self.assertFalse(data["runnable_now"])

    def test_write_service_mcp_readiness_outputs_markdown_and_json(self) -> None:
        report = build_service_mcp_readiness()
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "service-mcp-readiness.md"
            json_path = Path(tmp_dir) / "service-mcp-readiness.json"
            write_service_mcp_readiness(md_path, report=report)
            write_service_mcp_readiness(json_path, report=report, json_output=True)
            written_md = md_path.read_text(encoding="utf-8")
            data = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("MCP Readiness", written_md)
        self.assertTrue(data["passed"])
        self.assertEqual(data["tool_binding_count"], 5)

    def test_cli_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "service-mcp-readiness.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.service_mcp",
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
        self.assertEqual(data["resource_count"], 2)


if __name__ == "__main__":
    unittest.main()
