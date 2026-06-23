from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioml_agent.service_interface import (
    build_service_interface,
    format_service_interface,
    service_interface_to_dict,
    validate_service_interface,
    write_service_interface,
)


class ServiceInterfaceTest(unittest.TestCase):
    def test_builds_mcp_style_descriptors_from_tool_contracts(self) -> None:
        report = build_service_interface()
        data = service_interface_to_dict(report)
        descriptors = {tool["name"]: tool for tool in data["mcp_tools"]}

        self.assertTrue(report.passed)
        self.assertEqual(report.tool_count, 5)
        self.assertEqual(len(data["mcp_tools"]), 5)
        self.assertIn("search_literature", descriptors)
        self.assertIn("create_review_checkpoint", descriptors)
        for descriptor in data["mcp_tools"]:
            self.assertIn("inputSchema", descriptor)
            self.assertTrue(descriptor["annotations"]["readOnlyHint"])
            self.assertFalse(descriptor["annotations"]["destructiveHint"])
            self.assertTrue(descriptor["trace_required"])
            self.assertTrue(descriptor["approval_policy"])
            self.assertTrue(descriptor["failure_modes"])

    def test_builds_http_route_specs_from_tool_contracts(self) -> None:
        report = build_service_interface()
        routes = {route.path: route for route in report.http_routes}

        self.assertEqual(len(routes), 5)
        self.assertIn("/tools/search_literature", routes)
        self.assertIn("/tools/create_review_checkpoint", routes)
        self.assertEqual(routes["/tools/search_literature"].method, "POST")
        self.assertEqual(routes["/tools/search_literature"].request_schema["required"], ["query", "top_k"])
        self.assertEqual(
            routes["/tools/create_review_checkpoint"].state_dependencies,
            ("plan", "evidence", "context"),
        )

    def test_validation_checks_derivation_and_approval_policy(self) -> None:
        report = build_service_interface()
        validation = validate_service_interface(report)
        text = format_service_interface(report)

        self.assertTrue(validation.passed)
        self.assertFalse(validation.errors)
        self.assertTrue(report.warnings)
        self.assertIn("requires_approval", text)
        self.assertIn("session-injected state", text)

    def test_write_service_interface_outputs_markdown_and_json(self) -> None:
        report = build_service_interface()
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "service.md"
            json_path = Path(tmp_dir) / "service.json"
            write_service_interface(md_path, report=report)
            write_service_interface(json_path, report=report, json_output=True)
            md_text = md_path.read_text(encoding="utf-8")
            data = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("Service Interface Report", md_text)
        self.assertTrue(data["passed"])
        self.assertEqual(data["tool_count"], 5)

    def test_cli_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "service-interface-report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.service_interface",
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
        self.assertEqual(data["tool_count"], 5)


if __name__ == "__main__":
    unittest.main()
