from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tracepilot.models import (
    AgentRun,
    EvidenceChunk,
    EvidenceClaim,
    SourceDocument,
    TaskRequest,
    TaskStatus,
    content_digest,
)


class TracePilotModelsTest(unittest.TestCase):
    def test_request_normalizes_recoverable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            request = TaskRequest(
                goal="Upgrade the API",
                workspace_id="workspace-a",
                workspace_path=tmp_dir,
                source_urls=["https://example.test/docs"],
                constraints=["keep compatibility"],
            )

        self.assertEqual(request.workspace_path, str(Path(tmp_dir).resolve()))
        self.assertEqual(request.source_urls, ("https://example.test/docs",))
        self.assertEqual(request.constraints, ("keep compatibility",))
        self.assertTrue(request.task_id.startswith("task_"))

    def test_documents_and_chunks_get_stable_content_hashes(self) -> None:
        document = SourceDocument(content="same content")
        chunk = EvidenceChunk(document_id=document.document_id, text="same content")

        self.assertEqual(document.content_hash, content_digest("same content"))
        self.assertEqual(chunk.content_hash, document.content_hash)
        self.assertEqual(chunk.content, chunk.text)

    def test_claim_confidence_and_alias_are_safe(self) -> None:
        claim = EvidenceClaim(
            text="The API changed.",
            evidence_ids=["chunk-1"],
            confidence=1.5,
        )

        self.assertEqual(claim.claim, "The API changed.")
        self.assertEqual(claim.evidence_ids, ("chunk-1",))
        self.assertEqual(claim.confidence, 1.0)

    def test_agent_run_has_defaults_and_enforces_state_transitions(self) -> None:
        request = TaskRequest(goal="Diagnose tests")
        run = AgentRun(request=request)

        self.assertEqual(run.task_id, request.task_id)
        self.assertIs(run.task, request)
        run.transition(TaskStatus.PLANNING)
        self.assertEqual(run.status, TaskStatus.PLANNING)
        with self.assertRaises(ValueError):
            run.transition(TaskStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
