from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bioml_agent.state_graph import (
    DONE_NODE,
    initial_state,
    load_graph_checkpoint,
    run_state_graph,
    state_to_run,
    write_graph_checkpoint,
)
from bioml_agent.tools import ToolRegistry, build_default_registry


class StateGraphTest(unittest.TestCase):
    def test_checkpoint_can_stop_and_resume(self) -> None:
        registry = build_default_registry()
        state = initial_state("Should we try retrieval-aware conditioning?")

        partial = run_state_graph(registry, state, stop_after="score_transfer_fit")

        self.assertEqual(partial.next_node, "draft_experiment_plan")
        self.assertIsNone(partial.plan)
        self.assertEqual(
            [event.tool for event in partial.trace],
            ["search_literature", "read_experiment_context", "score_transfer_fit"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.json"
            write_graph_checkpoint(partial, checkpoint_path)
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            loaded = load_graph_checkpoint(checkpoint_path)

        self.assertEqual(data["next_node"], "draft_experiment_plan")
        self.assertEqual(loaded.next_node, "draft_experiment_plan")
        self.assertEqual(loaded.evidence[0].evidence_id, "E1")

        resumed = run_state_graph(registry, loaded)
        run = state_to_run(resumed)

        self.assertEqual(resumed.next_node, DONE_NODE)
        self.assertEqual(run.plan.recommendation, "promote to small controlled screen")
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

    def test_completed_state_is_idempotent(self) -> None:
        registry = build_default_registry()
        completed = run_state_graph(registry, initial_state("Should we try graph transfer?"))
        trace_len = len(completed.trace)

        resumed = run_state_graph(registry, completed)

        self.assertIs(resumed, completed)
        self.assertEqual(resumed.next_node, DONE_NODE)
        self.assertEqual(len(resumed.trace), trace_len)

    def test_incomplete_state_cannot_be_converted_to_run(self) -> None:
        registry = build_default_registry()
        partial = run_state_graph(
            registry,
            initial_state("Should we try retrieval-aware conditioning?"),
            stop_after="read_experiment_context",
        )

        with self.assertRaises(RuntimeError):
            state_to_run(partial)

    def test_failed_node_can_be_checkpointed_and_retried(self) -> None:
        failing_registry = ToolRegistry()
        failing_registry.register("search_literature", lambda query, top_k=4: [])

        def fail_context(project_name: str) -> object:
            raise ValueError(f"context temporarily unavailable: {project_name}")

        failing_registry.register("read_experiment_context", fail_context)
        failing_registry.register("score_transfer_fit", lambda candidates, context: [])
        failing_registry.register("draft_experiment_plan", lambda evidence, context: None)
        failing_registry.register("create_review_checkpoint", lambda plan, evidence, context: None)

        failed = run_state_graph(
            failing_registry,
            initial_state("Should we try retrieval-aware conditioning?"),
            capture_errors=True,
        )

        self.assertEqual(failed.next_node, "read_experiment_context")
        self.assertIsNotNone(failed.failure)
        self.assertEqual(failed.failure.node, "read_experiment_context")
        self.assertEqual(failed.failure.error_type, "ValueError")

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "failed.json"
            write_graph_checkpoint(failed, checkpoint_path)
            loaded = load_graph_checkpoint(checkpoint_path)

        self.assertIsNotNone(loaded.failure)
        resumed = run_state_graph(build_default_registry(), loaded)
        run = state_to_run(resumed)

        self.assertIsNone(resumed.failure)
        self.assertEqual(resumed.next_node, DONE_NODE)
        self.assertEqual(run.plan.recommendation, "do not run")
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


if __name__ == "__main__":
    unittest.main()
