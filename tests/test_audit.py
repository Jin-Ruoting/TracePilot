from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from bioml_agent.agent_loop import BioMLResearchAgent
from bioml_agent.audit import audit_run, audit_to_dict, format_audit_report, write_audit_report
from bioml_agent.literature import FrozenLiteratureProvider
from bioml_agent.tools import build_default_registry


class AuditTest(unittest.TestCase):
    def test_promotion_run_is_ready_for_human_review(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")

        report = audit_run(run)

        self.assertTrue(report.passed)
        self.assertEqual(report.decision, "ready_for_human_review")
        self.assertEqual(report.score, 1.0)
        self.assertIn("requires human approval", report.summary)
        self.assertTrue(any(finding.name == "promotion_evidence" for finding in report.findings))

    def test_diagnostic_run_is_recorded_without_promotion(self) -> None:
        run = BioMLResearchAgent().run("Should we try graph transfer for promoter expression prediction?")

        report = audit_run(run)

        self.assertTrue(report.passed)
        self.assertEqual(report.decision, "record_diagnostic")

    def test_abstention_run_is_recorded_as_negative_screening(self) -> None:
        registry = build_default_registry(
            literature_provider=FrozenLiteratureProvider("benchmarks/abstention_literature_snapshot.jsonl")
        )
        run = BioMLResearchAgent(registry=registry).run(
            "Should we try unsupported quantum graph retrieval for promoter expression prediction?"
        )

        report = audit_run(run)

        self.assertTrue(report.passed)
        self.assertEqual(report.decision, "record_abstention")

    def test_source_poisoning_run_is_recorded_as_abstention(self) -> None:
        registry = build_default_registry(
            literature_provider=FrozenLiteratureProvider("benchmarks/source_poisoning_literature_snapshot.jsonl")
        )
        run = BioMLResearchAgent(registry=registry).run(
            "Should we try poisoned retrieval conditioning for promoter expression prediction?"
        )

        report = audit_run(run)
        poisoning = next(finding for finding in report.findings if finding.name == "source_poisoning")

        self.assertTrue(report.passed)
        self.assertEqual(report.decision, "record_abstention")
        self.assertEqual(poisoning.status, "pass")
        self.assertIn("blocked from promotion", poisoning.detail)

    def test_audit_fails_when_promotion_lacks_approval(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        run.review = replace(run.review, status="not_required", required_actions=())

        report = audit_run(run)

        self.assertFalse(report.passed)
        self.assertEqual(report.decision, "needs_revision")
        self.assertTrue(any(finding.name == "review_checkpoint" for finding in report.findings if finding.status == "fail"))
        self.assertTrue(report.required_actions)

    def test_audit_fails_when_split_gate_is_incomplete(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        run.plan = replace(run.plan, promotion_gate=("Pearson improves",))

        report = audit_run(run)

        self.assertFalse(report.passed)
        self.assertIn("split/leakage", " ".join(report.required_actions))

    def test_audit_report_can_be_serialized_and_formatted(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        report = audit_run(run)
        data = audit_to_dict(report)
        text = format_audit_report(report)

        self.assertEqual(data["decision"], "ready_for_human_review")
        self.assertIn("Decision Audit", text)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "audit.json"
            write_audit_report(report, output_path)
            written = output_path.read_text(encoding="utf-8")
        self.assertIn("ready_for_human_review", written)


if __name__ == "__main__":
    unittest.main()
