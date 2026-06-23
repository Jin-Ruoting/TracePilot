from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioml_agent.service_api import (
    build_service_api_readiness,
    format_service_api_readiness,
    service_api_readiness_to_dict,
    write_service_api_readiness,
)


class ServiceApiTest(unittest.TestCase):
    def test_api_readiness_derives_endpoints_from_service_interface(self) -> None:
        report = build_service_api_readiness()
        data = service_api_readiness_to_dict(report)
        endpoints = {(endpoint["method"], endpoint["path"]): endpoint for endpoint in data["endpoints"]}

        self.assertTrue(report.passed)
        self.assertEqual(report.endpoint_count, 7)
        self.assertEqual(report.tool_endpoint_count, 5)
        self.assertIn(("GET", "/health"), endpoints)
        self.assertIn(("GET", "/service-interface"), endpoints)
        self.assertIn(("POST", "/tools/search_literature"), endpoints)
        self.assertIn(("POST", "/tools/create_review_checkpoint"), endpoints)
        self.assertEqual(
            endpoints[("POST", "/tools/create_review_checkpoint")]["state_dependencies"],
            ["plan", "evidence", "context"],
        )

    def test_openapi_preview_contains_tool_request_schemas(self) -> None:
        report = build_service_api_readiness()
        preview = report.openapi_preview
        search_schema = preview["paths"]["/tools/search_literature"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]

        self.assertEqual(preview["openapi"], "3.1.0")
        self.assertEqual(preview["info"]["title"], "TracePilot Service")
        self.assertEqual(search_schema["required"], ["query", "top_k"])
        self.assertFalse(search_schema["additionalProperties"])

    def test_optional_dependency_status_is_explicit(self) -> None:
        report = build_service_api_readiness()
        data = service_api_readiness_to_dict(report)
        dependency_names = {item["name"] for item in data["optional_dependencies"]}
        text = format_service_api_readiness(report)

        self.assertEqual(dependency_names, {"fastapi", "uvicorn"})
        self.assertIn("python -m pip install -e '.[api]'", text)
        self.assertIn("Runnable now:", text)

    def test_write_service_api_readiness_outputs_markdown_and_json(self) -> None:
        report = build_service_api_readiness()
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "service-api-readiness.md"
            json_path = Path(tmp_dir) / "service-api-readiness.json"
            write_service_api_readiness(md_path, report=report)
            write_service_api_readiness(json_path, report=report, json_output=True)
            written_md = md_path.read_text(encoding="utf-8")
            data = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("Service API Readiness", written_md)
        self.assertTrue(data["passed"])
        self.assertEqual(data["endpoint_count"], 7)

    def test_cli_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "service-api-readiness.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.service_api",
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
        self.assertEqual(data["tool_endpoint_count"], 5)


if __name__ == "__main__":
    unittest.main()
