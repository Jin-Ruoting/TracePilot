from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectConfigTest(unittest.TestCase):
    def test_makefile_exposes_quality_gate_entrypoints(self) -> None:
        text = (ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("check:", text)
        self.assertIn("tracepilot.quality_gate", text)
        self.assertIn("--public-output", text)
        self.assertIn("legacy-check:", text)
        self.assertIn("bioml_agent.quality_gate", text)
        self.assertIn("$(TRACE_DIR)/quality-gate-report.md", text)
        self.assertIn("promotion-decision-review.html", text)
        self.assertIn("review-workbench:", text)
        self.assertIn("review-workbench.html", text)
        self.assertIn("reference-catalog:", text)
        self.assertIn("reference-catalog-report.md", text)
        self.assertIn("service-interface:", text)
        self.assertIn("service-interface-report.md", text)
        self.assertIn("service-runtime:", text)
        self.assertIn("service-runtime-demo.md", text)
        self.assertIn("service-api:", text)
        self.assertIn("service-api-readiness.md", text)
        self.assertIn("service-mcp:", text)
        self.assertIn("service-mcp-readiness.md", text)
        self.assertIn("portfolio-index:", text)
        self.assertIn("serve-portfolio:", text)
        self.assertIn("bioml_agent.portfolio_index", text)
        self.assertIn("bioml_agent.review_workbench", text)
        self.assertIn("bioml_agent.reference_catalog", text)
        self.assertIn("bioml_agent.service_interface", text)
        self.assertIn("bioml_agent.service_runtime", text)
        self.assertIn("bioml_agent.service_api", text)
        self.assertIn("bioml_agent.service_mcp", text)
        self.assertIn("--root .", text)

    def test_github_actions_runs_same_quality_gate(self) -> None:
        workflow = ROOT / ".github" / "workflows" / "quality-gate.yml"
        text = workflow.read_text(encoding="utf-8")

        self.assertIn("make check", text)
        self.assertIn("make check-json", text)
        self.assertIn("make demo", text)
        self.assertIn("actions/checkout@v4", text)
        self.assertIn("actions/setup-python@v5", text)
        self.assertIn("actions/upload-artifact@v4", text)
        self.assertIn("traces/quality-gate-report.md", text)
        self.assertIn("traces/quality-gate-report.json", text)
        self.assertIn("traces/general-benchmark.md", text)
        self.assertIn("traces/general-benchmark.json", text)
        self.assertIn("traces/technical-decision-run.json", text)
        self.assertIn("traces/workbench.html", text)

    def test_pyproject_exposes_console_entrypoints(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('bioml-service-interface = "bioml_agent.service_interface:main"', text)
        self.assertIn('bioml-service-runtime = "bioml_agent.service_runtime:main"', text)
        self.assertIn('bioml-service-api = "bioml_agent.service_api:main"', text)
        self.assertIn('bioml-service-mcp = "bioml_agent.service_mcp:main"', text)
        self.assertIn('bioml-review-workbench = "bioml_agent.review_workbench:main"', text)
        self.assertIn('api = ["fastapi>=0.111", "uvicorn>=0.30"]', text)
        self.assertIn('mcp = ["mcp[cli]>=1.27,<2"]', text)
        self.assertIn('bioml-check = "bioml_agent.quality_gate:main"', text)
        self.assertIn('bioml-portfolio = "bioml_agent.portfolio_index:main"', text)
        self.assertIn('tracepilot = "tracepilot.cli:main"', text)
        self.assertIn('tracepilot-check = "tracepilot.quality_gate:main"', text)
        self.assertIn('tracepilot-serve = "tracepilot.api:main"', text)
        self.assertIn('tracepilot-portfolio = "tracepilot.workbench:main"', text)
        self.assertIn('bioml-agent = "bioml_agent.cli:main"', text)
        self.assertNotIn("bioml-application-packet", text)

    def test_readme_links_public_project_path(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("docs/ARCHITECTURE.md", readme)
        self.assertIn("[English](README.md)", readme)
        self.assertIn("[简体中文](docs/README.zh-CN.md)", readme)
        self.assertIn("examples/technical_decisions", readme)
        self.assertIn("benchmarks/general_sources.jsonl", readme)
        self.assertIn("http://127.0.0.1:8787", readme)
        self.assertIn("tracepilot.taskpacks.bioml", readme)

    def test_chinese_readme_is_nested_and_links_back_to_english(self) -> None:
        chinese_path = ROOT / "docs" / "README.zh-CN.md"
        readme = chinese_path.read_text(encoding="utf-8")

        self.assertFalse((ROOT / "README.zh-CN.md").exists())
        self.assertIn("[English](../README.md)", readme)
        self.assertIn("[简体中文](README.zh-CN.md)", readme)
        self.assertIn("[`ARCHITECTURE.md`](ARCHITECTURE.md)", readme)

    def test_public_release_excludes_internal_material_and_outputs(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in (
            "traces/",
            "docs/agent.md",
            "docs/*INTERVIEW*",
            "docs/*KNOWLEDGE_BASE*",
        ):
            self.assertIn(pattern, ignore)

        public_docs = {
            path.relative_to(ROOT / "docs").as_posix()
            for path in (ROOT / "docs").rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            public_docs,
            {"ARCHITECTURE.md", "README.zh-CN.md"},
        )


if __name__ == "__main__":
    unittest.main()
