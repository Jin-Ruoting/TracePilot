from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from tracepilot.memory import (
    SCHEMA_VERSION,
    IdempotencyConflictError,
    ImmutableRecordConflictError,
    MemoryStateConflictError,
    MemoryStore,
    WorkspaceScopeError,
    content_fingerprint,
    source_fingerprint,
    workspace_fingerprint,
)
from tracepilot.models import (
    AgentRun,
    ApprovalRequest,
    MemoryRecord,
    TaskRequest,
    TraceEvent,
)
from tracepilot.serialization import checkpoint_to_dict


@dataclass(frozen=True)
class FakeMemoryRecord:
    memory_id: str
    workspace_id: str
    kind: str
    content: dict[str, object]
    source_run_id: str = ""
    content_hash: str = ""
    workspace_fingerprint: str = ""
    status: str = "active"
    supersedes_id: str = ""
    created_at: str = ""
    updated_at: str = ""


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "memory.sqlite3"
        self.store = MemoryStore(self.database_path)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary_directory.cleanup()

    def test_schema_version_and_wal_mode_are_enabled(self) -> None:
        self.assertEqual(self.store.journal_mode, "wal")
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            version = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
        self.assertEqual(version, str(SCHEMA_VERSION))

    def test_fingerprints_are_canonical_and_domain_separated(self) -> None:
        left = {"b": [2, 1], "a": {"x": True}}
        right = {"a": {"x": True}, "b": [2, 1]}
        self.assertEqual(content_fingerprint(left), content_fingerprint(right))
        self.assertEqual(len(content_fingerprint(left)), 64)
        self.assertNotEqual(content_fingerprint("source"), source_fingerprint("source"))
        self.assertEqual(
            workspace_fingerprint("workspace-a"),
            workspace_fingerprint(" workspace-a "),
        )

    def test_working_memory_is_workspace_and_task_scoped(self) -> None:
        first = self.store.put_working_memory(
            "workspace-a",
            "planner",
            {"next_node": "search"},
            task_id="task-1",
            idempotency_key="put-1",
        )
        replay = self.store.put_working_memory(
            "workspace-a",
            "planner",
            {"next_node": "search"},
            task_id="task-1",
            idempotency_key="put-1",
        )
        self.assertEqual(first, replay)
        self.store.put_working_memory(
            "workspace-b",
            "planner",
            {"next_node": "review"},
            task_id="task-1",
        )
        self.assertEqual(
            self.store.get_working_memory(
                "workspace-a", "planner", task_id="task-1"
            ),
            {"next_node": "search"},
        )
        self.assertEqual(
            self.store.get_working_memory(
                "workspace-b", "planner", task_id="task-1"
            ),
            {"next_node": "review"},
        )
        self.assertIsNone(
            self.store.get_working_memory(
                "workspace-a", "planner", task_id="different-task"
            )
        )
        with self.assertRaises(IdempotencyConflictError):
            self.store.put_working_memory(
                "workspace-a",
                "planner",
                {"next_node": "different"},
                task_id="task-1",
                idempotency_key="put-1",
            )
        self.assertTrue(
            self.store.delete_working_memory(
                "workspace-a",
                "planner",
                task_id="task-1",
                idempotency_key="delete-1",
            )
        )
        self.assertTrue(
            self.store.delete_working_memory(
                "workspace-a",
                "planner",
                task_id="task-1",
                idempotency_key="delete-1",
            )
        )
        with self.assertRaises(IdempotencyConflictError):
            self.store.delete_working_memory(
                "workspace-a",
                "another-key",
                task_id="task-1",
                idempotency_key="delete-1",
            )

    def test_checkpoint_recovery_and_immutable_run_history(self) -> None:
        checkpoint_one = self.store.save_checkpoint(
            "workspace-a",
            {"task_id": "task-1", "run_id": "run-1", "next_node": "search"},
            checkpoint_id="checkpoint-1",
        )
        checkpoint_two = self.store.save_checkpoint(
            "workspace-a",
            {"task_id": "task-1", "run_id": "run-1", "next_node": "review"},
            checkpoint_id="checkpoint-2",
        )
        self.assertEqual(
            self.store.latest_checkpoint("workspace-a", "task-1"),
            checkpoint_two,
        )
        self.assertEqual(
            self.store.load_checkpoint("workspace-a", checkpoint_one.checkpoint_id)[
                "next_node"
            ],
            "search",
        )

        run = {"run_id": "run-1", "task_id": "task-1", "status": "completed", "answer": 42}
        stored = self.store.append_run("workspace-a", run)
        self.assertEqual(self.store.append_run("workspace-a", run), stored)
        with self.assertRaises(ImmutableRecordConflictError):
            self.store.append_run(
                "workspace-a",
                {
                    "run_id": "run-1",
                    "task_id": "task-1",
                    "status": "completed",
                    "answer": 43,
                },
            )
        self.assertIsNone(self.store.get_run("workspace-b", "run-1"))

    def test_decision_memory_lifecycle_and_model_hydration(self) -> None:
        record = FakeMemoryRecord(
            memory_id="memory-1",
            workspace_id="workspace-a",
            kind="decision",
            content={"recommendation": "diagnostic"},
            source_run_id="run-1",
        )
        promoted = self.store.promote_decision_memory(
            "workspace-a", record, idempotency_key="promote-1"
        )
        replay = self.store.promote_decision_memory(
            "workspace-a", record, idempotency_key="promote-1"
        )
        self.assertEqual(promoted, replay)
        self.assertEqual(promoted.status, "active")
        hydrated = promoted.to_model(FakeMemoryRecord)
        self.assertEqual(hydrated.memory_id, "memory-1")
        self.assertEqual(hydrated.content_hash, promoted.content_fingerprint)
        self.assertEqual(
            [item.memory_id for item in self.store.list_decision_memories("workspace-a")],
            ["memory-1"],
        )
        self.assertIsNone(self.store.get_decision_memory("workspace-b", "memory-1"))

        disabled = self.store.disable_decision_memory("workspace-a", "memory-1")
        self.assertEqual(disabled.status, "disabled")
        self.assertEqual(self.store.list_decision_memories("workspace-a"), ())

        replacement = self.store.supersede_decision_memory(
            "workspace-a",
            "memory-1",
            {
                "memory_id": "memory-2",
                "workspace_id": "workspace-a",
                "kind": "decision",
                "content": {"recommendation": "promote"},
                "source_run_id": "run-2",
            },
        )
        self.assertEqual(replacement.supersedes_id, "memory-1")
        self.assertEqual(
            replacement.workspace_fingerprint,
            promoted.workspace_fingerprint,
        )
        self.assertEqual(
            replacement.source_fingerprint,
            promoted.source_fingerprint,
        )
        old = self.store.get_decision_memory("workspace-a", "memory-1")
        self.assertIsNotNone(old)
        assert old is not None
        self.assertEqual(old.status, "superseded")
        self.assertEqual(old.superseded_by_id, "memory-2")

        stale = self.store.mark_decision_memory_stale(
            "workspace-a", "memory-2", reason="source snapshot changed"
        )
        self.assertEqual(stale.status, "stale")
        self.assertTrue(self.store.is_decision_memory_stale("workspace-a", "memory-2"))
        deleted = self.store.delete_decision_memory("workspace-a", "memory-2")
        self.assertEqual(deleted.status, "deleted")
        self.assertIsNone(self.store.get_decision_memory("workspace-a", "memory-2"))
        self.assertEqual(
            self.store.get_decision_memory(
                "workspace-a", "memory-2", include_deleted=True
            ).status,
            "deleted",
        )

    def test_public_models_round_trip_through_generic_storage(self) -> None:
        memory = MemoryRecord(
            memory_id="memory-public",
            workspace_id="workspace-a",
            kind="decision",
            content={"gate": "human approval"},
            source_task_id="task-public",
            workspace_revision="revision-1",
            source_versions={"paper": "sha256:abc"},
            confirmed=True,
        )
        memory_entry = self.store.promote_decision_memory("workspace-a", memory)
        restored_memory = memory_entry.to_model(MemoryRecord)
        self.assertEqual(restored_memory, memory)

        request = TaskRequest(
            task_id="task-public",
            goal="test persistence",
            workspace_id="workspace-a",
            workspace_path=self.temporary_directory.name,
        )
        run = AgentRun(run_id="run-public", task_id=request.task_id, request=request)
        self.store.append_run("workspace-a", run)
        restored_run = self.store.load_run(
            "workspace-a", "run-public", model_type=AgentRun
        )
        self.assertIsInstance(restored_run, AgentRun)
        self.assertEqual(restored_run.run_id, run.run_id)
        self.assertEqual(restored_run.request.goal, run.request.goal)

        checkpoint = checkpoint_to_dict(run)
        checkpoint_entry = self.store.save_checkpoint(
            "workspace-a", checkpoint, checkpoint_id="checkpoint-public"
        )
        restored_checkpoint = checkpoint_entry.to_model(AgentRun)
        self.assertIsInstance(restored_checkpoint, AgentRun)
        self.assertEqual(restored_checkpoint.task_id, request.task_id)

        approval = ApprovalRequest(
            approval_id="approval-public",
            task_id=request.task_id,
            step_id="step-public",
            tool_name="write_file",
            arguments={"path": "result.txt"},
            arguments_hash="frozen-hash",
            status="pending",
        )
        approval_entry = self.store.save_approval("workspace-a", approval)
        self.assertEqual(
            approval_entry.to_model(ApprovalRequest).arguments_hash,
            "frozen-hash",
        )

        event = TraceEvent(
            event_id="event-public",
            task_id=request.task_id,
            step_id="step-public",
            event_type="tool_completed",
            status="completed",
        )
        event_entry = self.store.append_event("workspace-a", event)
        self.assertEqual(
            event_entry.to_model(TraceEvent).event_type,
            "tool_completed",
        )

    def test_fingerprint_mismatch_marks_memory_stale_without_mutating_it(self) -> None:
        entry = self.store.promote_decision_memory(
            "workspace-a",
            {
                "memory_id": "memory-1",
                "workspace_id": "workspace-a",
                "kind": "constraint",
                "content": "use chromosome-aware split",
            },
            source={"snapshot": "v1"},
            workspace_state={"commit": "abc"},
        )
        self.assertFalse(
            self.store.is_decision_memory_stale(
                "workspace-a",
                "memory-1",
                current_source_fingerprint=entry.source_fingerprint,
                current_workspace_fingerprint=entry.workspace_fingerprint,
            )
        )
        self.assertTrue(
            self.store.is_decision_memory_stale(
                "workspace-a",
                "memory-1",
                current_source_fingerprint=source_fingerprint({"snapshot": "v2"}),
            )
        )
        self.assertEqual(
            self.store.get_decision_memory("workspace-a", "memory-1").status,
            "active",
        )

    def test_stale_memory_requires_current_state_to_supersede(self) -> None:
        old = self.store.promote_decision_memory(
            "workspace-a",
            {
                "memory_id": "old",
                "workspace_id": "workspace-a",
                "content": {"summary": "old decision"},
            },
            source={"guide": "v1"},
            workspace_state={"commit": "one"},
        )
        self.store.mark_decision_memory_stale("workspace-a", "old")

        with self.assertRaisesRegex(
            MemoryStateConflictError,
            "current source and workspace state",
        ):
            self.store.supersede_decision_memory(
                "workspace-a",
                "old",
                {
                    "workspace_id": "workspace-a",
                    "content": {"summary": "new decision"},
                },
            )

        replacement = self.store.supersede_decision_memory(
            "workspace-a",
            "old",
            {
                "workspace_id": "workspace-a",
                "content": {"summary": "new decision"},
            },
            source={"guide": "v2"},
            workspace_state={"commit": "two"},
        )

        self.assertEqual(replacement.status, "active")
        self.assertNotEqual(
            replacement.workspace_fingerprint,
            old.workspace_fingerprint,
        )
        self.assertNotEqual(
            replacement.source_fingerprint,
            old.source_fingerprint,
        )

    def test_payload_workspace_mismatch_is_rejected(self) -> None:
        with self.assertRaises(WorkspaceScopeError):
            self.store.promote_decision_memory(
                "workspace-a",
                {
                    "memory_id": "memory-1",
                    "workspace_id": "workspace-b",
                    "kind": "decision",
                    "content": "wrong scope",
                },
            )
        with self.assertRaises(WorkspaceScopeError):
            self.store.append_run(
                "workspace-a",
                {
                    "run_id": "run-wrong-scope",
                    "task_id": "task-1",
                    "request": {"workspace_id": "workspace-b"},
                },
            )
        with self.assertRaises(WorkspaceScopeError):
            self.store.put_working_memory(
                "workspace-a",
                "scoped-record",
                {"workspace_id": "workspace-b", "value": 1},
            )

    def test_nested_source_and_workspace_state_are_fingerprinted(self) -> None:
        first = self.store.append_run(
            "workspace-a",
            {
                "run_id": "run-source-v1",
                "task_id": "task-1",
                "request": {
                    "workspace_id": "workspace-a",
                    "workspace_path": "/workspace/project",
                    "source_urls": ["https://example.test/paper"],
                },
                "context": {
                    "workspace_id": "workspace-a",
                    "workspace_revision": "revision-1",
                    "source_versions": {"paper": "v1"},
                },
            },
        )
        second = self.store.append_run(
            "workspace-a",
            {
                "run_id": "run-source-v2",
                "task_id": "task-1",
                "request": {
                    "workspace_id": "workspace-a",
                    "workspace_path": "/workspace/project",
                    "source_urls": ["https://example.test/paper"],
                },
                "context": {
                    "workspace_id": "workspace-a",
                    "workspace_revision": "revision-2",
                    "source_versions": {"paper": "v2"},
                },
            },
        )
        self.assertTrue(first.source_fingerprint)
        self.assertTrue(first.workspace_fingerprint)
        self.assertNotEqual(first.source_fingerprint, second.source_fingerprint)
        self.assertNotEqual(first.workspace_fingerprint, second.workspace_fingerprint)

    def test_task_step_approval_event_and_recovery_bundle(self) -> None:
        task = self.store.save_task(
            "workspace-a",
            {"task_id": "task-1", "status": "running", "kind": "research"},
        )
        step = self.store.save_step(
            "workspace-a",
            "task-1",
            {
                "step_id": "step-1",
                "task_id": "task-1",
                "status": "completed",
                "kind": "search",
                "ordinal": 1,
            },
        )
        approval = self.store.save_approval(
            "workspace-a",
            {
                "approval_id": "approval-1",
                "task_id": "task-1",
                "step_id": "step-1",
                "tool_name": "execute_experiment",
                "arguments": {"run_id": "run-1"},
                "arguments_hash": "hash",
                "status": "pending",
                "reason": "promotion gate",
            },
        )
        event = self.store.append_event(
            "workspace-a",
            {
                "event_id": "event-1",
                "task_id": "task-1",
                "step_id": "step-1",
                "kind": "tool_completed",
                "timestamp": "2026-07-31T00:00:00Z",
                "data": {"count": 2},
            },
        )
        checkpoint = self.store.save_checkpoint(
            "workspace-a",
            {"task_id": "task-1", "next_node": "review"},
            checkpoint_id="checkpoint-1",
        )
        run = self.store.append_run(
            "workspace-a",
            {"run_id": "run-1", "task_id": "task-1", "status": "completed"},
        )

        recovery = self.store.recover_task("workspace-a", "task-1")
        self.assertEqual(recovery.task, task)
        self.assertEqual(recovery.checkpoint, checkpoint)
        self.assertEqual(recovery.steps, (step,))
        self.assertEqual(recovery.approvals, (approval,))
        self.assertEqual(recovery.events, (event,))
        self.assertEqual(recovery.runs, (run,))
        self.assertTrue(recovery.resumable)

    def test_database_can_be_reopened(self) -> None:
        self.store.put_working_memory("workspace-a", "key", {"value": 1})
        self.store.close()
        self.store = MemoryStore(self.database_path)
        self.assertEqual(
            self.store.get_working_memory("workspace-a", "key"),
            {"value": 1},
        )


if __name__ == "__main__":
    unittest.main()
