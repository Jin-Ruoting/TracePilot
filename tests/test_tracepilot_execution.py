from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tracepilot.execution import (
    ApprovalHashMismatchError,
    ApprovalRequiredError,
    ExecutionRuntime,
    IdempotencyConflictError,
    WorkspaceBoundaryError,
)
from tracepilot.models import ApprovalStatus
from tracepilot.tooling import ToolContract, ToolRegistry


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class TracePilotExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.registry = ToolRegistry()
        self.registry.register(
            ToolContract(
                name="read_file",
                input_schema=_schema({"path": {"type": "string"}}, ["path"]),
                effect_class="read",
                permissions=("workspace_read",),
            ),
            lambda path: f"read:{path}",
        )
        self.mutations: list[tuple[str, str]] = []
        self.registry.register(
            ToolContract(
                name="apply_patch",
                input_schema=_schema(
                    {
                        "path": {"type": "string"},
                        "patch": {"type": "string"},
                    },
                    ["path", "patch"],
                ),
                effect_class="mutation",
                idempotent=False,
                permissions=("workspace_write",),
            ),
            self._mutate,
        )
        self.runtime = ExecutionRuntime(self.registry, self.workspace)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _mutate(self, path: str, patch: str) -> dict:
        self.mutations.append((path, patch))
        return {"path": path, "applied": True}

    def test_reads_execute_without_approval(self) -> None:
        result = self.runtime.execute("read_file", {"path": "README.md"})

        self.assertTrue(result.success)
        self.assertEqual(result.output, "read:README.md")

    def test_mutation_requires_matching_approval_and_is_at_most_once(self) -> None:
        arguments = {"path": "src/example.py", "patch": "new content"}
        pending = self.runtime.prepare(
            "apply_patch",
            arguments,
            task_id="task-1",
            step_id="mutate",
        )
        self.assertEqual(pending.status, ApprovalStatus.PENDING)
        with self.assertRaises(ApprovalRequiredError):
            self.runtime.execute("apply_patch", arguments)

        approved = self.runtime.approve(pending, approved_by="reviewer")
        first = self.runtime.execute(
            "apply_patch",
            arguments,
            task_id="task-1",
            step_id="mutate",
            approval=approved,
        )
        second = self.runtime.execute(
            "apply_patch",
            arguments,
            task_id="task-1",
            step_id="mutate",
            approval=approved,
        )

        self.assertTrue(first.success)
        self.assertTrue(second.cached)
        self.assertEqual(len(self.mutations), 1)

    def test_parameter_hash_cannot_change_after_approval(self) -> None:
        pending = self.runtime.prepare(
            "apply_patch",
            {"path": "a.py", "patch": "one"},
        )
        approved = self.runtime.approve(pending)

        with self.assertRaises(ApprovalHashMismatchError):
            self.runtime.execute(
                "apply_patch",
                {"path": "a.py", "patch": "two"},
                approval=approved,
            )

    def test_persisted_approval_can_be_restored_and_executed(self) -> None:
        arguments = {"path": "restored.py", "patch": "content"}
        approved = self.runtime.approve(
            self.runtime.prepare("apply_patch", arguments)
        )
        resumed = ExecutionRuntime(self.registry, self.workspace)

        restored = resumed.restore_approval(approved)
        result = resumed.execute("apply_patch", arguments, approval=restored)

        self.assertEqual(restored, approved)
        self.assertTrue(result.success)

    def test_workspace_escape_is_blocked(self) -> None:
        with self.assertRaises(WorkspaceBoundaryError):
            self.runtime.execute("read_file", {"path": "../outside.txt"})

    def test_nested_workspace_escape_is_blocked(self) -> None:
        self.registry.register(
            ToolContract(
                name="batch_patch",
                input_schema=_schema(
                    {"changes": {"type": "array"}},
                    ["changes"],
                ),
                effect_class="mutation",
            ),
            lambda changes: changes,
        )
        with self.assertRaises(WorkspaceBoundaryError):
            self.runtime.prepare(
                "batch_patch",
                {"changes": [{"path": "../outside.txt", "content": "x"}]},
            )

    def test_idempotency_key_cannot_be_reused_for_different_calls(self) -> None:
        self.runtime.execute(
            "read_file",
            {"path": "one.txt"},
            idempotency_key="same-key",
        )
        with self.assertRaises(IdempotencyConflictError):
            self.runtime.execute(
                "read_file",
                {"path": "two.txt"},
                idempotency_key="same-key",
            )


if __name__ == "__main__":
    unittest.main()
