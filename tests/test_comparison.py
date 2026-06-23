from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bioml_agent.benchmark import load_eval_cases
from bioml_agent.comparison import format_comparison_result, result_to_dict, run_comparison


class BaselineComparisonTest(unittest.TestCase):
    def test_default_corpus_shows_agent_improvement_over_naive_baseline(self) -> None:
        result = run_comparison(load_eval_cases("benchmarks/eval_cases.jsonl"))
        self.assertEqual(result.total, 4)
        self.assertEqual(result.agent_passed_count, 4)
        self.assertLess(result.baseline_passed_count, result.agent_passed_count)
        self.assertGreaterEqual(result.improved_count, 1)

        retrieval_case = next(item for item in result.case_results if item.case.case_id == "retrieval_conditioning")
        baseline_failed = {check.name for check in retrieval_case.baseline_checks if not check.passed}
        self.assertIn("tool_trace_complete", baseline_failed)
        self.assertIn("claim_citations_bound", baseline_failed)
        self.assertIn("review_checkpoint_valid", baseline_failed)
        self.assertIn("split_gate_present", baseline_failed)

    def test_format_comparison_result_is_human_readable(self) -> None:
        result = run_comparison(load_eval_cases("benchmarks/eval_cases.jsonl"))
        text = format_comparison_result(result)
        self.assertIn("# Baseline vs TracePilot", text)
        self.assertIn("Agent passed: 4/4", text)
        self.assertIn("Naive baseline passed:", text)
        self.assertIn("retrieval_conditioning", text)

    def test_result_to_dict_contains_failed_check_summary(self) -> None:
        result = run_comparison(load_eval_cases("benchmarks/eval_cases.jsonl"))
        data = result_to_dict(result)
        self.assertEqual(data["agent_passed_count"], 4)
        self.assertIn("cases", data)
        baseline = data["cases"][0]["baseline"]
        self.assertIn("failed_checks", baseline)
        self.assertIn("matrix", baseline)
        self.assertIn("trace_tools", baseline)

    def test_comparison_uses_experiment_log_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            log_path = root / "experiment_log.json"
            log_path.write_text(
                json.dumps(
                    {
                        "project_name": "AltProject",
                        "trusted_split": "holdout chromosome split",
                        "diagnostic_split": "sample diagnostic split",
                        "runs": [
                            {
                                "run_id": "comparison-baseline",
                                "description": "comparison experiment-log baseline",
                                "status": "trusted_baseline",
                                "split": "holdout chromosome split",
                                "metrics": {"Pearson": 0.63, "MSE": 0.74},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cases_path = root / "cases.jsonl"
            cases_path.write_text(
                json.dumps(
                    {
                        "case_id": "experiment_log_comparison",
                        "project_name": "AltProject",
                        "experiment_log": str(log_path),
                        "question": "Should we try retrieval-aware conditioning?",
                        "required_matrix_terms": ["comparison experiment-log baseline"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_comparison(load_eval_cases(cases_path))

        self.assertEqual(result.total, 1)
        self.assertTrue(result.case_results[0].agent_passed)
        self.assertIn("comparison experiment-log baseline", " ".join(result.case_results[0].agent_run.plan.matrix))
        self.assertIn("comparison experiment-log baseline", " ".join(result.case_results[0].baseline_run.plan.matrix))


if __name__ == "__main__":
    unittest.main()
