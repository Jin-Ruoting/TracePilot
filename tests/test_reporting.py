from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from bioml_agent.agent_loop import BioMLResearchAgent
from bioml_agent.audit import audit_run
from bioml_agent.evals import evaluate_run
from bioml_agent.literature import FrozenLiteratureProvider
from bioml_agent.reporting import (
    _run_from_dict,
    format_decision_packet,
    format_decision_packet_html,
    write_decision_packet,
    write_decision_packet_html,
)
from bioml_agent.serialization import run_to_dict, write_run_json
from bioml_agent.source_validation import validate_run_sources
from bioml_agent.tools import build_default_registry


class ReportingTest(unittest.TestCase):
    def test_decision_packet_summarizes_promotion_run(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        text = format_decision_packet(
            run,
            audit_report=audit_run(run),
            eval_result=evaluate_run(run),
        )

        self.assertIn("TracePilot Decision Packet", text)
        self.assertIn("ready_for_human_review", text)
        self.assertIn("Evidence Table", text)
        self.assertIn("claim_citations_bound", text)
        self.assertIn("approval:promotion-gate", text)
        self.assertIn("search_literature", text)

    def test_decision_packet_includes_source_resolution_when_provided(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        source_resolution = validate_run_sources(run)

        text = format_decision_packet(run, source_resolution=source_resolution)

        self.assertIn("Source Resolution", text)
        self.assertIn("fixture source is offline-only", text)

    def test_decision_packet_exposes_source_poisoning_guardrail(self) -> None:
        registry = build_default_registry(
            literature_provider=FrozenLiteratureProvider("benchmarks/source_poisoning_literature_snapshot.jsonl")
        )
        run = BioMLResearchAgent(registry=registry).run(
            "Should we try poisoned retrieval conditioning for promoter expression prediction?"
        )

        text = format_decision_packet(run)

        self.assertIn("do not run", text)
        self.assertIn("Source-poisoning risk", text)
        self.assertIn("source_poisoning_guarded", text)
        self.assertIn("blocked from promotion", text)

    def test_write_decision_packet_creates_markdown_file(self) -> None:
        run = BioMLResearchAgent().run("Should we try graph transfer for promoter expression prediction?")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "packet.md"
            write_decision_packet(run, output_path)
            written = output_path.read_text(encoding="utf-8")

        self.assertIn("Decision Summary", written)
        self.assertIn("run diagnostic-only screen", written)

    def test_trace_json_can_be_rendered_back_to_packet(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            write_run_json(run, trace_path)
            loaded = _run_from_dict(json.loads(trace_path.read_text(encoding="utf-8")))
            text = format_decision_packet(loaded)

        self.assertEqual(run_to_dict(loaded)["question"], run_to_dict(run)["question"])
        self.assertIn("Tool Trace", text)
        self.assertIn("draft_experiment_plan", text)

    def test_html_decision_packet_summarizes_review_artifact(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")

        text = format_decision_packet_html(run)

        self.assertIn("<!doctype html>", text)
        self.assertIn("TracePilot Decision Review", text)
        self.assertIn("Evidence", text)
        self.assertIn("Decision Audit", text)
        self.assertIn("Tool Trace", text)
        self.assertIn("requires_approval", text)

    def test_html_decision_packet_escapes_untrusted_text(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        run.evidence[0] = replace(run.evidence[0], supports="<script>alert('x')</script>")

        text = format_decision_packet_html(run)

        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", text)
        self.assertNotIn("<script>alert", text)

    def test_write_html_decision_packet_creates_file(self) -> None:
        run = BioMLResearchAgent().run("Should we try graph transfer for promoter expression prediction?")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "packet.html"
            write_decision_packet_html(run, output_path)
            written = output_path.read_text(encoding="utf-8")

        self.assertIn("<html", written)
        self.assertIn("run diagnostic-only screen", written)

    def test_reporting_cli_renders_html_from_trace_json(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trace_path = tmp_path / "trace.json"
            html_path = tmp_path / "packet.html"
            write_run_json(run, trace_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bioml_agent.reporting",
                    str(trace_path),
                    "--format",
                    "html",
                    "--out",
                    str(html_path),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            written = html_path.read_text(encoding="utf-8")

        self.assertIn("Decision Review", written)
        self.assertIn("create_review_checkpoint", written)


if __name__ == "__main__":
    unittest.main()
