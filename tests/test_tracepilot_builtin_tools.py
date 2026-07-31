from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from tracepilot.builtin_tools import build_default_registry
from tracepilot.rag import RagIndex


class TracePilotBuiltinToolsTest(unittest.TestCase):
    def test_registry_declares_effects_and_indexes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "client.py").write_text(
                "timeout = 30\nCACHE_NAMESPACE = 'demo'\n",
                encoding="utf-8",
            )
            with RagIndex(root / "index.db") as index:
                registry = build_default_registry(workspace_root=root, index=index)
                indexed = registry.call(
                    "index_workspace",
                    {"workspace_path": str(root), "workspace_id": "demo"},
                )
                results = registry.call(
                    "search_index",
                    {"query": "CACHE_NAMESPACE", "top_k": 8},
                )

                self.assertEqual(
                    registry.contract("run_command").effect_class.value,
                    "mutation",
                )
                self.assertGreaterEqual(indexed["documents"], 1)
                self.assertTrue(results)

    def test_file_tools_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with RagIndex(root / "index.db") as index:
                registry = build_default_registry(workspace_root=root, index=index)
                with self.assertRaisesRegex(PermissionError, "escapes workspace"):
                    registry.call(
                        "read_file",
                        {"path": str(root.parent / "secret.txt")},
                    )

    def test_nonzero_command_is_a_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with RagIndex(root / "index.db") as index:
                registry = build_default_registry(workspace_root=root, index=index)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "command exited with status 7",
                ):
                    registry.call(
                        "run_command",
                        {
                            "workspace_path": str(root),
                            "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                        },
                    )

    def test_patch_requires_and_rechecks_expected_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "config.txt"
            target.write_text("v1", encoding="utf-8")
            with RagIndex(root / "index.db") as index:
                registry = build_default_registry(workspace_root=root, index=index)
                with self.assertRaisesRegex(ValueError, "expected_sha256"):
                    registry.call(
                        "apply_patch",
                        {
                            "workspace_path": str(root),
                            "changes": [
                                {"path": "config.txt", "content": "v2"}
                            ],
                        },
                    )
                with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                    registry.call(
                        "apply_patch",
                        {
                            "workspace_path": str(root),
                            "changes": [
                                {
                                    "path": "config.txt",
                                    "content": "v2",
                                    "expected_sha256": "0" * 64,
                                }
                            ],
                        },
                    )
                result = registry.call(
                    "apply_patch",
                    {
                        "workspace_path": str(root),
                        "changes": [
                            {
                                "path": "config.txt",
                                "content": "v2",
                                "expected_sha256": hashlib.sha256(
                                    b"v1"
                                ).hexdigest(),
                            }
                        ],
                    },
                )

            self.assertEqual(target.read_text(encoding="utf-8"), "v2")
            self.assertEqual(len(result["written"]), 1)


if __name__ == "__main__":
    unittest.main()
