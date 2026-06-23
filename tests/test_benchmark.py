from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from bioml_agent.benchmark import (
    format_benchmark_result,
    load_eval_cases,
    result_to_dict,
    run_eval_corpus,
)
from bioml_agent.fixtures import PAPER_FIXTURES


class BenchmarkTest(unittest.TestCase):
    def test_load_and_run_default_eval_corpus(self) -> None:
        cases = load_eval_cases("benchmarks/eval_cases.jsonl")
        self.assertEqual(len(cases), 4)
        result = run_eval_corpus(cases)
        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 4)
        self.assertEqual(result.total, 4)

    def test_load_and_run_live_frozen_eval_corpus(self) -> None:
        cases = load_eval_cases("benchmarks/live_eval_cases.jsonl")
        self.assertEqual(len(cases), 3)
        result = run_eval_corpus(cases)
        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 3)
        self.assertEqual(result.total, 3)

    def test_load_and_run_abstention_eval_corpus(self) -> None:
        cases = load_eval_cases("benchmarks/abstention_eval_cases.jsonl")
        self.assertEqual(len(cases), 2)
        result = run_eval_corpus(cases)
        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 2)
        self.assertEqual(result.total, 2)

    def test_load_and_run_source_poisoning_eval_corpus(self) -> None:
        cases = load_eval_cases("benchmarks/source_poisoning_eval_cases.jsonl")
        self.assertEqual(len(cases), 1)
        result = run_eval_corpus(cases)
        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 1)
        self.assertEqual(result.total, 1)
        self.assertEqual(result.case_results[0].run.plan.recommendation, "do not run")
        self.assertIn("Source-poisoning risk", result.case_results[0].run.evidence[0].cautions)

    def test_load_and_run_stress_eval_corpus(self) -> None:
        cases = load_eval_cases("benchmarks/stress_eval_cases.jsonl")
        self.assertEqual(len(cases), 40)
        result = run_eval_corpus(cases)
        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 40)
        self.assertEqual(result.total, 40)

    def test_load_and_run_real_literature_eval_corpus(self) -> None:
        cases = load_eval_cases("benchmarks/real_literature_eval_cases.jsonl")
        self.assertEqual(len(cases), 10)
        result = run_eval_corpus(cases)
        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 10)
        self.assertEqual(result.total, 10)

    def test_result_to_dict_contains_case_summary(self) -> None:
        result = run_eval_corpus(load_eval_cases("benchmarks/eval_cases.jsonl"))
        data = result_to_dict(result)
        self.assertTrue(data["passed"])
        self.assertEqual(data["passed_count"], 4)
        self.assertEqual(data["cases"][0]["case_id"], "retrieval_conditioning")
        self.assertEqual(data["cases"][0]["top_candidate_metadata"]["source_url"], "fixture://papers/tabr-2023")
        self.assertEqual(data["cases"][0]["top_candidate_metadata"]["venue"], "fixture")
        self.assertEqual(data["cases"][0]["review"]["status"], "requires_approval")
        self.assertIn("approval:promotion-gate", data["cases"][0]["review"]["checkpoint_id"])
        self.assertEqual(data["cases"][0]["cited_claims"][0]["evidence_ids"], ["E1"])
        self.assertEqual(data["cases"][0]["cited_claims"][0]["source_refs"], ["fixture://papers/tabr-2023"])
        self.assertIn("checks", data["cases"][0])
        check_names = {check["name"] for check in data["cases"][0]["checks"]}
        self.assertIn("claim_citations_bound", check_names)
        self.assertIn("source_refs_valid", check_names)
        self.assertIn("review_checkpoint_valid", check_names)
        self.assertIn("top_candidate_source_url", check_names)
        self.assertIn("top_candidate_venue", check_names)

    def test_format_benchmark_result_is_human_readable(self) -> None:
        result = run_eval_corpus(load_eval_cases("benchmarks/eval_cases.jsonl"))
        text = format_benchmark_result(result)
        self.assertIn("# TracePilot Benchmark", text)
        self.assertIn("Cases: 4/4", text)
        self.assertIn("PASS retrieval_conditioning", text)
        self.assertIn("Review: requires_approval", text)

    def test_run_eval_corpus_with_frozen_literature_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            query = "Should we try retrieval-aware conditioning for promoter expression prediction?"
            snapshot = root / "snapshot.jsonl"
            snapshot.write_text(
                json.dumps(
                    {
                        "source": "fixture-test",
                        "query": query,
                        "top_k": 1,
                        "candidates": [asdict(PAPER_FIXTURES[1])],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases = root / "cases.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "case_id": "frozen_retrieval",
                        "question": query,
                        "literature_source": "frozen",
                        "literature_snapshot": str(snapshot),
                        "expected_recommendation": "promote to small controlled screen",
                        "expected_top_candidate": "tabr-2023",
                        "require_source_url": True,
                        "require_venue": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_eval_corpus(load_eval_cases(cases))

        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 1)

    def test_run_eval_corpus_with_context_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            context_path = root / "context.json"
            context_path.write_text(
                json.dumps(
                    {
                        "project_name": "AltProject",
                        "trusted_split": "holdout chromosome split",
                        "diagnostic_split": "sample diagnostic split",
                        "baseline": "portfolio context-file baseline",
                        "metrics": ["Pearson", "MSE"],
                        "constraints": ["main claims must use holdout chromosome split"],
                    }
                ),
                encoding="utf-8",
            )
            cases = root / "cases.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "case_id": "context_file_baseline",
                        "project_name": "AltProject",
                        "context_file": str(context_path),
                        "question": "Should we try retrieval-aware conditioning?",
                        "required_matrix_terms": ["portfolio context-file baseline"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_eval_corpus(load_eval_cases(cases))

        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 1)

    def test_run_eval_corpus_with_experiment_log(self) -> None:
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
                                "run_id": "alt-baseline",
                                "description": "portfolio experiment-log baseline",
                                "status": "trusted_baseline",
                                "split": "holdout chromosome split",
                                "metrics": {"Pearson": 0.62, "MSE": 0.77},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cases = root / "cases.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "case_id": "experiment_log_baseline",
                        "project_name": "AltProject",
                        "experiment_log": str(log_path),
                        "question": "Should we try retrieval-aware conditioning?",
                        "required_matrix_terms": ["portfolio experiment-log baseline", "Pearson=0.62"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_eval_corpus(load_eval_cases(cases))

        self.assertTrue(result.passed)
        self.assertEqual(result.passed_count, 1)

    def test_loader_reports_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.jsonl"
            path.write_text('{"case_id":"missing-question"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_eval_cases(path)


if __name__ == "__main__":
    unittest.main()
