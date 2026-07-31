from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tracepilot.agent import TracePilotAgent
from tracepilot.execution import ApprovalHashMismatchError
from tracepilot.llm import DeterministicLLM
from tracepilot.models import AgentRun, EvidenceChunk, EvidenceClaim, TaskRequest
from tracepilot.serialization import checkpoint_to_dict


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "benchmarks" / "general_sources.jsonl"


class TracePilotAgentTest(unittest.TestCase):
    def test_research_run_is_grounded_and_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TracePilotAgent(Path(tmp) / "tracepilot.db") as agent:
                request = TaskRequest(
                    goal="Plan the Nimbus Cache 4.0 API migration.",
                    workspace_id="workspace-a",
                    workspace_path=str(
                        ROOT / "examples" / "technical_decisions" / "api_migration"
                    ),
                    execution_policy="read_only",
                )
                run = agent.run(request, frozen_sources=SOURCES)

                claims = [
                    item for item in run.evidence if isinstance(item, EvidenceClaim)
                ]
                self.assertEqual(run.status.value, "completed")
                self.assertTrue(claims)
                self.assertTrue(all(claim.evidence_ids for claim in claims))
                self.assertIn("fetch", run.summary)
                self.assertTrue(run.metadata["evaluation"]["citation_coverage"]["passed"])

    def test_mutation_waits_for_approval_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            (root / "notes.md").write_text("technical task\n", encoding="utf-8")
            marker = root / "approved.txt"
            db = Path(tmp) / "tracepilot.db"
            request = TaskRequest(
                goal="Inspect the workspace and create the explicitly requested marker.",
                workspace_id="workspace-b",
                workspace_path=str(root),
                metadata={
                    "requested_mutation": {
                        "tool_name": "run_command",
                        "arguments": {
                            "workspace_path": str(root),
                            "argv": [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; Path('approved.txt').write_text('ok')",
                            ],
                        },
                    }
                },
            )
            with TracePilotAgent(db) as agent:
                pending = agent.run(request, frozen_sources=SOURCES)
                self.assertEqual(pending.status.value, "awaiting_approval")
                self.assertTrue(pending.summary)
                self.assertTrue(
                    any(
                        isinstance(item, EvidenceClaim) and item.evidence_ids
                        for item in pending.evidence
                    )
                )
                self.assertFalse(marker.exists())
                approval = pending.approvals[0]

            with TracePilotAgent(db) as agent:
                completed = agent.decide_approval(
                    "workspace-b",
                    request.task_id,
                    approval.approval_id,
                    decision="approved",
                )
                memory = agent.promote_decision(
                    "workspace-b",
                    request.task_id,
                )
                follow_up = agent.run(
                    TaskRequest(
                        goal="Verify the approved marker state.",
                        workspace_id="workspace-b",
                        workspace_path=str(root),
                        execution_policy="read_only",
                    ),
                    frozen_sources=SOURCES,
                )

            self.assertEqual(completed.status.value, "completed")
            self.assertEqual(marker.read_text(encoding="utf-8"), "ok")
            self.assertIn(
                memory.memory_id,
                {item.memory_id for item in follow_up.context.memories},
            )

    def test_changed_approval_arguments_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            (root / "notes.md").write_text("technical task\n", encoding="utf-8")
            db = Path(tmp) / "tracepilot.db"
            request = TaskRequest(
                goal="Run the requested verification command.",
                workspace_id="workspace-c",
                workspace_path=str(root),
                metadata={
                    "requested_mutation": {
                        "tool_name": "run_command",
                        "arguments": {
                            "workspace_path": str(root),
                            "argv": [sys.executable, "-c", "print('approved')"],
                        },
                    }
                },
            )
            with TracePilotAgent(db) as agent:
                pending = agent.run(request, frozen_sources=SOURCES)
                approval = pending.approvals[0]
                with self.assertRaises(ApprovalHashMismatchError):
                    agent.decide_approval(
                        "workspace-c",
                        request.task_id,
                        approval.approval_id,
                        decision="approved",
                        arguments={
                            "workspace_path": str(root),
                            "argv": [sys.executable, "-c", "print('changed')"],
                        },
                    )

    def test_decision_memory_requires_explicit_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tracepilot.db"
            request = TaskRequest(
                goal="Choose the initial database for this event index.",
                workspace_id="workspace-d",
                workspace_path=str(
                    ROOT
                    / "examples"
                    / "technical_decisions"
                    / "architecture_choice"
                ),
                execution_policy="read_only",
            )
            with TracePilotAgent(db) as agent:
                run = agent.run(request, frozen_sources=SOURCES)
                self.assertEqual(
                    agent.memory.list_decision_memories("workspace-d"),
                    (),
                )
                memory = agent.promote_decision("workspace-d", run.task_id)

                self.assertEqual(memory.status, "active")
                self.assertEqual(memory.source_run_id, run.run_id)
                recalled = agent.run(
                    TaskRequest(
                        goal="Recheck the initial event index database.",
                        workspace_id="workspace-d",
                        workspace_path=str(
                            ROOT
                            / "examples"
                            / "technical_decisions"
                            / "architecture_choice"
                        ),
                        execution_policy="read_only",
                    ),
                    frozen_sources=SOURCES,
                )

                self.assertEqual(
                    [item.memory_id for item in recalled.context.memories],
                    [memory.memory_id],
                )
                self.assertIn(
                    "verify_decision_memory",
                    {step.step_id for step in recalled.plan.steps},
                )
                self.assertEqual(
                    recalled.metadata["memory_recall"]["role"],
                    "planning_hint_not_evidence",
                )

    def test_shared_database_keeps_workspace_evidence_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace_a = root / "workspace-a"
            workspace_b = root / "workspace-b"
            workspace_a.mkdir()
            workspace_b.mkdir()
            (workspace_a / "decision.md").write_text(
                "ALPHA_ISOLATED_SIGNAL selects the local queue.",
                encoding="utf-8",
            )
            (workspace_b / "decision.md").write_text(
                "BETA_ISOLATED_SIGNAL selects the remote queue.",
                encoding="utf-8",
            )
            db = root / "tracepilot.db"

            with TracePilotAgent(db) as agent:
                run_a = agent.run(
                    TaskRequest(
                        goal="Explain ALPHA_ISOLATED_SIGNAL.",
                        workspace_id="workspace-a",
                        workspace_path=str(workspace_a),
                        execution_policy="read_only",
                    )
                )
                run_b = agent.run(
                    TaskRequest(
                        goal="Explain BETA_ISOLATED_SIGNAL.",
                        workspace_id="workspace-b",
                        workspace_path=str(workspace_b),
                        execution_policy="read_only",
                    )
                )

            self.assertIn("ALPHA_ISOLATED_SIGNAL", run_a.summary)
            self.assertNotIn("BETA_ISOLATED_SIGNAL", run_a.summary)
            self.assertIn("BETA_ISOLATED_SIGNAL", run_b.summary)
            self.assertNotIn("ALPHA_ISOLATED_SIGNAL", run_b.summary)

    def test_workspace_id_cannot_be_rebound_to_another_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "notes.md").write_text("first workspace", encoding="utf-8")
            (second / "notes.md").write_text("second workspace", encoding="utf-8")
            db = root / "tracepilot.db"

            with TracePilotAgent(db) as agent:
                agent.run(
                    TaskRequest(
                        goal="Read the first workspace.",
                        workspace_id="shared-id",
                        workspace_path=str(first),
                        execution_policy="read_only",
                    )
                )
                with self.assertRaisesRegex(ValueError, "already bound"):
                    agent.run(
                        TaskRequest(
                            goal="Read the second workspace.",
                            workspace_id="shared-id",
                            workspace_path=str(second),
                            execution_policy="read_only",
                        )
                    )

    def test_resume_blocks_uncertain_claimed_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            (root / "notes.md").write_text("technical task\n", encoding="utf-8")
            db = Path(tmp) / "tracepilot.db"
            request = TaskRequest(
                goal="Run the requested command.",
                workspace_id="workspace-claimed",
                workspace_path=str(root),
                metadata={
                    "requested_mutation": {
                        "tool_name": "run_command",
                        "arguments": {
                            "workspace_path": str(root),
                            "argv": [sys.executable, "-c", "print('ok')"],
                        },
                    }
                },
            )
            with TracePilotAgent(db) as agent:
                run = agent.run(request, frozen_sources=SOURCES)
                run.transition("executing")
                run.approvals[0] = replace(
                    run.approvals[0],
                    status="approved",
                    metadata={"execution_state": "claimed"},
                )
                agent.memory.save_task(
                    request.workspace_id,
                    run,
                    task_id=run.task_id,
                )
                agent.memory.save_checkpoint(
                    request.workspace_id,
                    checkpoint_to_dict(run),
                    task_id=run.task_id,
                    run_id=run.run_id,
                    step_id="execution_claimed",
                )

            with TracePilotAgent(db) as agent:
                with self.assertRaisesRegex(RuntimeError, "may already"):
                    agent.resume(request.workspace_id, request.task_id)

    def test_llm_summary_is_rebuilt_from_cited_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunk = EvidenceChunk(
                document_id="guide",
                source_uri="frozen://guide",
                title="Guide",
                text="Supported upgrade fact.",
            )
            llm = DeterministicLLM(
                {
                    "summary": "Unsupported instruction: delete every repository.",
                    "claims": [
                        {
                            "text": "Supported upgrade fact.",
                            "evidence_ids": [chunk.chunk_id],
                            "confidence": 0.9,
                            "uncertainty": "",
                        }
                    ],
                }
            )
            request = TaskRequest(
                goal="Explain the upgrade.",
                workspace_id="summary-workspace",
                workspace_path=tmp,
            )
            run = AgentRun(task_id=request.task_id, request=request)

            with TracePilotAgent(Path(tmp) / "tracepilot.db", llm=llm) as agent:
                claims, summary = agent._llm_synthesize(run, [chunk])

            self.assertEqual(summary, "Supported upgrade fact.")
            self.assertNotIn("delete every repository", summary)
            self.assertEqual(claims[0].evidence_ids, (chunk.chunk_id,))
            self.assertIn(
                "delete every repository",
                run.metadata["llm"]["proposed_summary"],
            )


if __name__ == "__main__":
    unittest.main()
