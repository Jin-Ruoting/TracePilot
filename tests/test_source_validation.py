from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bioml_agent.agent_loop import BioMLResearchAgent
from bioml_agent.literature import FrozenLiteratureProvider
from bioml_agent.source_validation import (
    format_source_resolution_report,
    validate_run_sources,
    write_source_resolution_report,
)
from bioml_agent.tools import build_default_registry


class SourceValidationTest(unittest.TestCase):
    def test_fixture_sources_are_skipped(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")

        report = validate_run_sources(run, fetcher=lambda url: (200, "reachable"))

        self.assertTrue(report.passed)
        self.assertTrue(report.results)
        self.assertTrue(all(result.status == "skipped" for result in report.results))

    def test_real_snapshot_sources_are_normalized_and_resolved(self) -> None:
        registry = build_default_registry(
            literature_provider=FrozenLiteratureProvider("benchmarks/live_literature_snapshot.jsonl")
        )
        run = BioMLResearchAgent(registry=registry).run(
            "Should we try retrieval-aware conditioning for promoter expression prediction?"
        )
        seen_urls: list[str] = []

        def fake_fetcher(url: str):
            seen_urls.append(url)
            return 200, "reachable"

        report = validate_run_sources(run, fetcher=fake_fetcher)

        self.assertTrue(report.passed)
        self.assertIn("https://arxiv.org/abs/2307.14338", seen_urls)
        self.assertIn("https://doi.org/10.48550/arXiv.2307.14338", seen_urls)
        self.assertTrue(all(result.status == "ok" for result in report.results))

    def test_failed_remote_source_fails_report(self) -> None:
        registry = build_default_registry(
            literature_provider=FrozenLiteratureProvider("benchmarks/live_literature_snapshot.jsonl")
        )
        run = BioMLResearchAgent(registry=registry).run(
            "Should we try graph transfer for promoter expression prediction?"
        )

        report = validate_run_sources(run, fetcher=lambda url: (404, "not found"))

        self.assertFalse(report.passed)
        self.assertTrue(all(result.status == "failed" for result in report.results))
        self.assertIn("FAILED", format_source_resolution_report(report))

    def test_source_resolution_report_can_be_written(self) -> None:
        registry = build_default_registry(
            literature_provider=FrozenLiteratureProvider("benchmarks/live_literature_snapshot.jsonl")
        )
        run = BioMLResearchAgent(registry=registry).run(
            "Should we try graph transfer for promoter expression prediction?"
        )
        report = validate_run_sources(run, fetcher=lambda url: (200, "reachable"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sources.json"
            write_source_resolution_report(report, output_path)
            text = output_path.read_text(encoding="utf-8")

        self.assertIn('"passed": true', text)
        self.assertIn("pubmed.ncbi.nlm.nih.gov", text)


if __name__ == "__main__":
    unittest.main()
