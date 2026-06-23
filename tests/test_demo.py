from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bioml_agent.demo import demo_to_dict, format_portfolio_demo, run_portfolio_demo


class DemoTest(unittest.TestCase):
    def test_portfolio_demo_passes_and_covers_core_paths(self) -> None:
        demo = run_portfolio_demo()

        self.assertTrue(demo.passed)
        self.assertEqual(demo.scenarios[0].audit_decision, "ready_for_human_review")
        self.assertEqual(demo.scenarios[1].audit_decision, "record_abstention")
        self.assertTrue(all(item.passed for item in demo.benchmarks))
        self.assertEqual(sum(item.total for item in demo.benchmarks), 60)
        self.assertEqual(demo.comparison.agent_passed_count, demo.comparison.total)
        self.assertLess(demo.comparison.baseline_passed_count, demo.comparison.agent_passed_count)
        self.assertEqual(demo.checkpoint.failure_node, "read_experiment_context")
        self.assertEqual(demo.checkpoint.resumed_review_status, "requires_approval")

    def test_portfolio_demo_is_serializable_and_human_readable(self) -> None:
        demo = run_portfolio_demo()
        data = demo_to_dict(demo)
        text = format_portfolio_demo(demo)

        self.assertTrue(data["passed"])
        self.assertIn("Portfolio Demo", text)
        self.assertIn("Baseline Comparison", text)
        self.assertIn("Failure Recovery", text)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "demo.json"
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["checkpoint"]["failure_node"], "read_experiment_context")


if __name__ == "__main__":
    unittest.main()
