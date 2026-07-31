from __future__ import annotations

import unittest

from tracepilot.workbench import render_workbench


class TracePilotWorkbenchTest(unittest.TestCase):
    def test_workbench_exposes_operational_views_and_live_api_calls(self) -> None:
        html = render_workbench("task-demo")

        self.assertIn("TracePilot Workbench", html)
        self.assertIn('data-tab="evidence"', html)
        self.assertIn('data-tab="memory"', html)
        self.assertIn('data-tab="execution"', html)
        self.assertIn("/events", html)
        self.assertIn("data-approval", html)
        self.assertIn("Modify", html)
        self.assertIn("modify-approval-dialog", html)
        self.assertIn("Save decision to memory", html)
        self.assertIn("Source chunks", html)
        self.assertIn("@media (max-width: 480px)", html)
        self.assertIn("if (!state.task)", html)
        self.assertNotIn("linear-gradient", html)


if __name__ == "__main__":
    unittest.main()
