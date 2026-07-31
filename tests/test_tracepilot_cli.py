from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tracepilot.cli import main


ROOT = Path(__file__).resolve().parents[1]


class TracePilotCliTest(unittest.TestCase):
    def test_run_writes_generic_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "run",
                        "Plan the Nimbus Cache 4.0 API migration.",
                        "--workspace",
                        str(
                            ROOT
                            / "examples"
                            / "technical_decisions"
                            / "api_migration"
                        ),
                        "--workspace-id",
                        "cli-workspace",
                        "--frozen-sources",
                        str(ROOT / "benchmarks" / "general_sources.jsonl"),
                        "--db",
                        str(Path(tmp) / "tracepilot.db"),
                        "--json-out",
                        str(output),
                        "--public-output",
                        "--read-only",
                    ]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "completed")
            self.assertIn("fetch", payload["summary"])
            self.assertEqual(payload["request"]["workspace_path"], "workspace://")
            self.assertNotIn("/Users/", output.read_text(encoding="utf-8"))
            self.assertIn("Workspace id: cli-workspace", stdout.getvalue())

    def test_ingest_reports_workspace_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "ingest",
                        str(
                            ROOT
                            / "examples"
                            / "technical_decisions"
                            / "architecture_choice"
                        ),
                        "--db",
                        str(Path(tmp) / "tracepilot.db"),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIn("active documents", stdout.getvalue())
            self.assertIn("Workspace id:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
