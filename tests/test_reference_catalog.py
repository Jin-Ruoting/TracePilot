from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioml_agent.reference_catalog import (
    build_reference_catalog_report,
    format_reference_catalog_report,
    reference_catalog_to_dict,
    write_reference_catalog_report,
)


class ReferenceCatalogTest(unittest.TestCase):
    def test_reference_catalog_covers_required_agent_sources(self) -> None:
        report = build_reference_catalog_report()
        source_ids = {entry.source_id for entry in report.entries}

        self.assertGreaterEqual(report.total, 9)
        self.assertGreaterEqual(report.github_count, 6)
        self.assertIn("openai_agents_sdk", source_ids)
        self.assertIn("langgraph", source_ids)
        self.assertIn("mcp", source_ids)
        self.assertIn("ragas", source_ids)
        self.assertIn("anthropic_effective_agents", source_ids)
        self.assertIn("autogen", source_ids)

    def test_reference_entries_are_actionable_for_design_notes(self) -> None:
        report = build_reference_catalog_report()

        for entry in report.entries:
            self.assertTrue(entry.url.startswith("https://"), entry.source_id)
            self.assertTrue(entry.current_signal, entry.source_id)
            self.assertTrue(entry.borrowed_pattern, entry.source_id)
            self.assertTrue(entry.project_evidence, entry.source_id)
            self.assertTrue(entry.design_note, entry.source_id)

    def test_reference_catalog_formats_and_serializes(self) -> None:
        report = build_reference_catalog_report()
        text = format_reference_catalog_report(report)
        data = reference_catalog_to_dict(report)

        self.assertIn("Agent Reference Catalog", text)
        self.assertIn("openai/openai-agents-python", text)
        self.assertIn("microsoft/autogen", text)
        self.assertEqual(data["total"], report.total)
        self.assertEqual(data["github_count"], report.github_count)

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "reference-catalog.md"
            json_path = Path(tmp_dir) / "reference-catalog.json"
            write_reference_catalog_report(report, md_path)
            write_reference_catalog_report(report, json_path, json_output=True)
            written_md = md_path.read_text(encoding="utf-8")
            written_json = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("Source Matrix", written_md)
        self.assertEqual(written_json["total"], report.total)

    def test_reference_catalog_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "reference-catalog.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.reference_catalog",
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

        self.assertGreaterEqual(data["total"], 9)


if __name__ == "__main__":
    unittest.main()
