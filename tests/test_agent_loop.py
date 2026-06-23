from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bioml_agent.agent_loop import BioMLResearchAgent, TracePilotAgent
from bioml_agent.literature import FixtureLiteratureProvider
from bioml_agent.openai_controller import build_openai_tool_specs
from bioml_agent.serialization import review_packet_to_dict, write_review_packet, write_run_json
from bioml_agent.tools import build_default_registry, search_literature


class BioMLResearchAgentTest(unittest.TestCase):
    def test_search_prioritizes_retrieval_query(self) -> None:
        candidates = search_literature("retrieval-aware conditioning for promoter expression")
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0].paper_id, "tabr-2023")

    def test_agent_run_contains_plan_gate_and_trace(self) -> None:
        run = BioMLResearchAgent().run(
            "Should we try retrieval-aware conditioning for promoter expression prediction?"
        )
        self.assertEqual(run.plan.recommendation, "promote to small controlled screen")
        self.assertTrue(any("chromosome-aware split" in gate for gate in run.plan.promotion_gate))
        self.assertEqual(run.evidence[0].evidence_id, "E1")
        self.assertEqual(run.plan.cited_claims[0].evidence_ids, ("E1",))
        self.assertEqual(run.plan.cited_claims[0].source_refs, ("fixture://papers/tabr-2023",))
        self.assertEqual(run.review.status, "requires_approval")
        self.assertIn("chromosome-aware split", " ".join(run.review.required_actions))
        self.assertEqual(
            [event.tool for event in run.trace],
            [
                "search_literature",
                "read_experiment_context",
                "score_transfer_fit",
                "draft_experiment_plan",
                "create_review_checkpoint",
            ],
        )

    def test_tracepilot_alias_uses_same_workflow(self) -> None:
        run = TracePilotAgent().run("Should we try retrieval-aware conditioning?")

        self.assertEqual(run.trace[0].tool, "search_literature")
        self.assertIn(run.plan.recommendation, {"promote to small controlled screen", "run diagnostic-only screen"})

    def test_graph_query_uses_graph_specific_matrix(self) -> None:
        run = BioMLResearchAgent().run("Should we try graph transfer for promoter expression prediction?")
        self.assertEqual(run.candidates[0].paper_id, "graphreg-2020")
        self.assertEqual(run.plan.recommendation, "run diagnostic-only screen")
        self.assertEqual(run.review.status, "not_required")
        self.assertTrue(any("regulatory graph prior" in item for item in run.plan.matrix))

    def test_context_rejects_unknown_project(self) -> None:
        with self.assertRaises(ValueError):
            BioMLResearchAgent().run("test", project_name="unknown")

    def test_registry_accepts_custom_literature_provider(self) -> None:
        registry = build_default_registry(literature_provider=FixtureLiteratureProvider())
        run = BioMLResearchAgent(registry=registry).run("Should we try graph transfer?")
        self.assertEqual(run.trace[0].tool, "search_literature")

    def test_trace_can_be_written_as_json(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "trace.json"
            write_run_json(run, output_path)
            text = output_path.read_text(encoding="utf-8")
        self.assertIn("search_literature", text)
        self.assertIn("promotion_gate", text)

    def test_review_packet_can_be_written_as_json(self) -> None:
        run = BioMLResearchAgent().run("Should we try retrieval-aware conditioning?")
        packet = review_packet_to_dict(run)
        self.assertEqual(packet["review"]["status"], "requires_approval")
        self.assertIn("cited_claims", packet)
        self.assertNotIn("trace", packet)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "review.json"
            write_review_packet(run, output_path)
            text = output_path.read_text(encoding="utf-8")
        self.assertIn("approval:promotion-gate", text)
        self.assertIn("confirm main evaluation uses chromosome-aware split", text)

    def test_openai_tool_specs_are_strict(self) -> None:
        specs = build_openai_tool_specs()
        self.assertEqual(
            [spec["name"] for spec in specs],
            [
                "search_literature",
                "read_experiment_context",
                "score_transfer_fit",
                "draft_experiment_plan",
                "create_review_checkpoint",
            ],
        )
        for spec in specs:
            self.assertTrue(spec["strict"])
            self.assertFalse(spec["parameters"]["additionalProperties"])
            self.assertIn("required", spec["parameters"])


if __name__ == "__main__":
    unittest.main()
