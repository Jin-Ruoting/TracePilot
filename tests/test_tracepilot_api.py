from __future__ import annotations

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

from tracepilot.api import TaskService, create_app


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "benchmarks" / "general_sources.jsonl"
WORKSPACE = ROOT / "examples" / "technical_decisions" / "failure_diagnosis"


class TracePilotTaskServiceTest(unittest.TestCase):
    def test_background_task_and_catalog_survive_service_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "service.db"
            service = TaskService(db, frozen_sources=SOURCES)
            created = service.submit(
                {
                    "goal": "Diagnose the worker startup failure.",
                    "workspace_path": str(WORKSPACE),
                    "execution_policy": "read_only",
                }
            )
            completed = service.wait(created.task_id, timeout=10)
            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertEqual(completed.status.value, "completed")
            workspace_id = completed.request.workspace_id
            service.close()

            reopened = TaskService(db)
            restored = reopened.find_task(created.task_id, workspace_id)
            reopened.close()

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertIn("CACHE_NAMESPACE", restored.summary)


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") is not None,
    "FastAPI optional dependency is not installed",
)
class TracePilotFastAPITest(unittest.TestCase):
    def test_app_exposes_task_sse_approval_artifact_and_memory_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(Path(tmp) / "api.db", frozen_sources=SOURCES)
            routes = {
                (method, route.path)
                for route in app.routes
                for method in getattr(route, "methods", ())
            }

            self.assertIn(("POST", "/v1/tasks"), routes)
            self.assertIn(("GET", "/v1/tasks/{task_id}/events"), routes)
            self.assertIn(
                ("POST", "/v1/tasks/{task_id}/approvals/{approval_id}"),
                routes,
            )
            self.assertIn(("GET", "/v1/tasks/{task_id}/artifacts"), routes)
            self.assertIn(("POST", "/v1/memory/promote"), routes)


if __name__ == "__main__":
    unittest.main()
