from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bioml_agent.agent_loop import BioMLResearchAgent
from bioml_agent.context import ExperimentLogContextProvider, FileExperimentContextProvider, build_context_provider
from bioml_agent.tools import build_default_registry


class ExperimentContextProviderTest(unittest.TestCase):
    def test_file_provider_reads_single_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_context(Path(tmp_dir) / "context.json")
            context = FileExperimentContextProvider(path).read("EpiCETRA")

        self.assertEqual(context.project_name, "EpiCETRA")
        self.assertEqual(context.baseline, "custom promoter baseline")
        self.assertEqual(context.metrics, ("Pearson", "MSE"))

    def test_file_provider_reads_context_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "contexts.json"
            path.write_text(
                json.dumps(
                    {
                        "contexts": [
                            _context_record(project_name="EpiCETRA"),
                            _context_record(project_name="AltProject", baseline="alt baseline"),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            context = FileExperimentContextProvider(path).read("AltProject")

        self.assertEqual(context.project_name, "AltProject")
        self.assertEqual(context.baseline, "alt baseline")

    def test_file_provider_rejects_unknown_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_context(Path(tmp_dir) / "context.json")
            provider = FileExperimentContextProvider(path)
            with self.assertRaises(ValueError):
                provider.read("unknown")

    def test_file_provider_validates_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.json"
            path.write_text(
                json.dumps({"project_name": "EpiCETRA", "baseline": "missing fields"}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                FileExperimentContextProvider(path)

    def test_registry_uses_file_context_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_context(Path(tmp_dir) / "context.json")
            registry = build_default_registry(context_provider=build_context_provider(path))
            run = BioMLResearchAgent(registry=registry).run("Should we try retrieval-aware conditioning?")

        self.assertTrue(any("custom promoter baseline" in item for item in run.plan.matrix))
        self.assertIn("custom chromosome split", " ".join(run.review.required_actions))

    def test_experiment_log_provider_derives_context_from_trusted_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_experiment_log(Path(tmp_dir) / "experiment_log.json")
            context = ExperimentLogContextProvider(path).read("EpiCETRA")

        self.assertEqual(context.project_name, "EpiCETRA")
        self.assertEqual(context.metrics, ("Pearson", "MSE", "R2"))
        self.assertIn("baseline-best", context.baseline)
        self.assertIn("Pearson=0.61", context.baseline)
        self.assertEqual(context.trusted_split, "holdout chromosome split")
        self.assertIn("sample diagnostic split is diagnostic only", context.constraints)

    def test_experiment_log_provider_rejects_missing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "experiment_log.json"
            record = _experiment_log_record()
            record["runs"] = [
                {
                    "run_id": "diagnostic-only",
                    "description": "diagnostic run",
                    "status": "diagnostic_only",
                    "split": "sample diagnostic split",
                    "metrics": {"Pearson": 0.72},
                }
            ]
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                ExperimentLogContextProvider(path)

    def test_build_context_provider_rejects_two_context_sources(self) -> None:
        with self.assertRaises(ValueError):
            build_context_provider(context_file="context.json", experiment_log="experiment_log.json")

    def test_registry_uses_experiment_log_context_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_experiment_log(Path(tmp_dir) / "experiment_log.json")
            registry = build_default_registry(
                context_provider=build_context_provider(experiment_log=path)
            )
            run = BioMLResearchAgent(registry=registry).run("Should we try retrieval-aware conditioning?")

        self.assertTrue(any("baseline-best" in item for item in run.plan.matrix))
        self.assertTrue(any("Pearson=0.61" in item for item in run.plan.matrix))
        self.assertIn("holdout chromosome split", " ".join(run.review.required_actions))


def _write_context(path: Path) -> Path:
    path.write_text(json.dumps(_context_record()), encoding="utf-8")
    return path


def _write_experiment_log(path: Path) -> Path:
    path.write_text(json.dumps(_experiment_log_record()), encoding="utf-8")
    return path


def _context_record(project_name: str = "EpiCETRA", baseline: str = "custom promoter baseline") -> dict[str, object]:
    return {
        "project_name": project_name,
        "trusted_split": "custom chromosome split",
        "diagnostic_split": "custom diagnostic split",
        "baseline": baseline,
        "metrics": ["Pearson", "MSE"],
        "constraints": [
            "main claims must use the custom chromosome split",
            "diagnostic split is not promotion evidence",
        ],
    }


def _experiment_log_record() -> dict[str, object]:
    return {
        "project_name": "EpiCETRA",
        "trusted_split": "holdout chromosome split",
        "diagnostic_split": "sample diagnostic split",
        "runs": [
            {
                "run_id": "baseline-old",
                "description": "old promoter baseline",
                "status": "trusted_baseline",
                "split": "holdout chromosome split",
                "metrics": {"Pearson": 0.58, "MSE": 0.82, "R2": 0.31},
            },
            {
                "run_id": "baseline-best",
                "description": "best promoter baseline",
                "status": "promoted",
                "split": "holdout chromosome split",
                "metrics": {"Pearson": 0.61, "MSE": 0.79, "R2": 0.35},
            },
            {
                "run_id": "diagnostic-sample",
                "description": "diagnostic sample-random run",
                "status": "diagnostic_only",
                "split": "sample diagnostic split",
                "metrics": {"Pearson": 0.74, "MSE": 0.51, "R2": 0.55},
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
