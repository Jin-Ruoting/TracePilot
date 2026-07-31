from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tracepilot.models import AgentRun, TaskContext, TaskRequest
from tracepilot.serialization import (
    CheckpointVersionError,
    checkpoint_from_dict,
    load_checkpoint,
    run_to_public_dict,
    sanitize_public_payload,
    write_checkpoint,
)


class TracePilotSerializationTest(unittest.TestCase):
    def test_v2_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = TaskRequest(goal="Choose a cache", workspace_path=tmp)
            run = AgentRun(
                task_id=request.task_id,
                request=request,
                context=TaskContext(
                    task_id=request.task_id,
                    workspace_id=request.workspace_id,
                    workspace_path=tmp,
                ),
            )
            path = Path(tmp) / "checkpoint.json"
            write_checkpoint(run, path)

            raw = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_checkpoint(path)

            self.assertEqual(raw["schema_version"], 2)
            self.assertEqual(loaded.request.goal, "Choose a cache")
            self.assertEqual(loaded.task_id, run.task_id)

    def test_legacy_bioml_checkpoint_is_mapped(self) -> None:
        run = checkpoint_from_dict(
            {
                "question": {"text": "Should we try graph transfer?", "domain": "promoter"},
                "project_name": "EpiCETRA",
                "next_node": "draft_experiment_plan",
                "candidates": [],
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "paper_id": "P1",
                        "title": "Graph paper",
                        "relevance": 0.75,
                        "supports": "Graph priors can transfer.",
                        "cautions": "Validate edges.",
                    }
                ],
                "trace": [],
            }
        )

        self.assertEqual(run.metadata["migrated_from"], 1)
        self.assertEqual(run.request.goal, "Should we try graph transfer?")
        self.assertEqual(run.plan.metadata["legacy_next_node"], "draft_experiment_plan")
        self.assertEqual(len(run.evidence), 1)

    def test_unknown_schema_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(CheckpointVersionError, "unsupported checkpoint"):
            checkpoint_from_dict({"schema_version": 99, "run": {}})

    def test_public_run_replaces_nested_local_roots_and_secrets(self) -> None:
        project_root = Path("/Users/alice/projects/TracePilot")
        workspace = project_root / "examples" / "architecture"
        request = TaskRequest(
            goal="Choose a database",
            workspace_path=str(workspace),
            metadata={
                "source_uri": f"file://{workspace}/requirements.md",
                "snapshot": str(project_root / "benchmarks" / "sources.jsonl"),
                "api_key": "private-value",
                "note": "Bearer " + ("a" * 16),
            },
        )
        run = AgentRun(
            task_id=request.task_id,
            request=request,
            context=TaskContext(
                task_id=request.task_id,
                workspace_id=request.workspace_id,
                workspace_path=str(workspace),
                source_versions={
                    f"file://{workspace}/requirements.md": "revision-1",
                },
            ),
        )

        payload = run_to_public_dict(run, project_root=project_root)
        rendered = json.dumps(payload, ensure_ascii=False)

        self.assertNotIn("/Users/alice", rendered)
        self.assertNotIn("private-value", rendered)
        self.assertNotIn("a" * 16, rendered)
        self.assertEqual(payload["request"]["workspace_path"], "workspace://")
        self.assertEqual(
            payload["request"]["metadata"]["source_uri"],
            "workspace://requirements.md",
        )
        self.assertEqual(
            payload["request"]["metadata"]["snapshot"],
            "project://benchmarks/sources.jsonl",
        )
        self.assertEqual(payload["request"]["metadata"]["api_key"], "[REDACTED]")

    def test_public_payload_sanitizes_dictionary_keys(self) -> None:
        payload = sanitize_public_payload(
            {
                "file:///Users/alice/project/source.txt": "safe",
                "headers": {"x-api-key": "secret-value"},
            },
            roots={"project": "/Users/alice/project"},
        )

        self.assertEqual(payload["project://source.txt"], "safe")
        self.assertEqual(payload["headers"]["x-api-key"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
