from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from tracepilot.research.models import ExperimentBudget, ResearchSpec, Variant
from tracepilot.research.context import build_context
from tracepilot.routing.decision import CascadeRouter, DecisionScores, ROLES, SystemOneGateway
from tracepilot.workers.coding import CodingWorker, snapshot_workspace
from tracepilot.workers.gateway import CompatibleJSONGateway
from tracepilot.workers.research import ResearchWorker
from tracepilot.workers.review import ReviewWorker, validate_metrics


class ResearchDecisionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = ResearchSpec.load(Path(__file__).resolve().parents[1]
                                      / "examples/feature_scaling/project.json")

    def tearDown(self):
        self.temp.cleanup()

    def test_budget_and_heldout_selection_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceed budget"):
            replace(self.spec, budget=ExperimentBudget(max_runs=2))
        with self.assertRaisesRegex(ValueError, "validation only"):
            replace(self.spec, partition="test")
        with self.assertRaises(ValueError):
            replace(self.spec, split_seed=-1)
        with self.assertRaises(ValueError):
            ExperimentBudget(max_concurrent=0)

    def test_source_snapshot_excludes_credentials_and_rejects_links(self):
        source, target = self.root / "source", self.root / "snapshot"
        source.mkdir()
        (source / "code.py").write_text("pass\n")
        (source / ".env").write_text("PRIVATE=value")
        (source / ".ssh").mkdir()
        (source / ".ssh" / "config").write_text("private")
        manifest = snapshot_workspace(source, target)
        self.assertEqual(list(manifest), ["code.py"])
        (source / "link.py").symlink_to(source / "code.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            snapshot_workspace(source, self.root / "other")

    def test_patch_cannot_escape_or_use_stale_content(self):
        with self.assertRaises(ValueError):
            Variant("bad", ({"path": "../escape", "expected_sha256": "a" * 64, "content": "x"},))
        source = self.root / "source"
        source.mkdir()
        (source / "x.py").write_text("pass\n")
        stale = Variant("stale", ({"path": "x.py", "expected_sha256": "a" * 64, "content": "x=1\n"},))
        with self.assertRaisesRegex(ValueError, "stale"):
            CodingWorker().prepare(source, self.root / "candidate", stale)
        self.assertEqual((source / "x.py").read_text(), "pass\n")

    def test_known_events_bypass_decision_model(self):
        model = Mock()
        router = CascadeRouter(model, threshold=0.9)
        self.assertEqual(router.route({"kind": "job_succeeded"}, state_version=5).role, "review")
        self.assertEqual(router.route({"kind": "unknown", "owner": "coding"}, state_version=6).role, "coding")
        model.score.assert_not_called()

    def test_confidence_is_distinct_from_max_probability(self):
        model = Mock()
        probabilities = dict(research=0.96, coding=0.02, review=0.01, escalate=0.01)
        model.score.return_value = DecisionScores(probabilities, confidence=0.6, model="fixture")
        router = CascadeRouter(model, threshold=0.9)
        decision = router.route({"kind": "ambiguous"}, state_version=4)
        self.assertEqual(decision.role, "escalate")
        self.assertEqual(decision.confidence, 0.6)
        model.score.return_value = DecisionScores(probabilities, confidence=0.95, model="fixture")
        decision = router.route({"kind": "ambiguous"}, state_version=5)
        self.assertEqual((decision.role, decision.source, decision.state_version),
                         ("research", "decision_model", 5))

    def test_invalid_or_unavailable_decision_model_escalates(self):
        invalid = [dict.fromkeys(ROLES, 2.0), dict.fromkeys(ROLES, float("nan")),
                   {"illegal": 1.0}]
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                model = Mock()
                model.score.return_value = DecisionScores(probabilities, confidence=1.0)
                decision = CascadeRouter(model).route({"kind": "ambiguous"}, state_version=1)
                self.assertEqual(decision.role, "escalate")
                self.assertEqual(decision.probabilities, {})
        model = Mock()
        model.score.side_effect = TimeoutError("offline")
        self.assertEqual(CascadeRouter(model).route({"kind": "unknown"}, state_version=1).role, "escalate")

    def test_system_one_http_contract_and_usage(self):
        response = {"model": "jev-fixture", "answers": {"route": {
            "choice": "coding", "confidence": 0.92,
            "probabilities": dict(research=0.02, coding=0.96, review=0.01, escalate=0.01)}},
            "usage": {"input_tokens": 10}}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
            result = SystemOneGateway("https://fixture.invalid/v1/systemone", "fixture").score({"event": "x"}, ROLES)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["questions"]["route"]["criteria"], ROLES)
        self.assertEqual(result.confidence, 0.92)
        self.assertEqual(result.usage["input_tokens"], 10)

    def test_generation_http_contract_returns_proposal_without_execution(self):
        response = {"model": "fixture", "choices": [{"message": {"content": '{"hypotheses":[]}'}}],
                    "usage": {"total_tokens": 20}}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
            result = CompatibleJSONGateway("http://localhost:9999/v1/chat/completions", "fixture").generate(
                "instructions", {"goal": "test"})
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(result["executed"])
        self.assertEqual(result["usage"]["total_tokens"], 20)

    def test_research_model_cannot_invent_evidence(self):
        item = dict(hypothesis_id="h", statement="Claim", mechanism="Mechanism",
                    falsification="Fails below zero", evidence_refs=["unknown"])
        model = Mock()
        model.generate.return_value = {"proposal": {"hypotheses": [item]}, "executed": False}
        with self.assertRaisesRegex(ValueError, "invented"):
            ResearchWorker(model).propose({"evidence": [{"evidence_id": "known"}]})
        item["evidence_refs"] = ["known"]
        result = ResearchWorker(model).propose({"evidence": [{"evidence_id": "known"}]})
        self.assertEqual(result["proposal"]["hypotheses"][0]["novelty_status"], "unverified")

    def test_model_patch_requires_known_files_and_exact_hashes(self):
        content = "x=1\n"
        sha = hashlib.sha256(content.encode()).hexdigest()
        context = {"files": [{"path": "x.py", "sha256": sha, "content": content}]}
        model = Mock()
        model.generate.return_value = {"proposal": {"changes": [
            {"path": "x.py", "expected_sha256": sha, "content": "x=2\n"}]}, "executed": False}
        worker = ResearchWorker(model)
        self.assertEqual(worker.propose_patch(context)["proposal"]["status"], "proposal_only")
        model.generate.return_value["proposal"]["changes"][0]["path"] = "unlisted.py"
        with self.assertRaisesRegex(ValueError, "unknown or stale"):
            worker.propose_patch(context)

    def rows(self):
        rows = []
        for variant in self.spec.variants:
            for dataset in self.spec.datasets:
                for seed in self.spec.seeds:
                    rows.append({
                        "id": f"{variant.variant_id}-{dataset}-{seed}", "status": "succeeded",
                        "definition": {"variant": variant.variant_id, "dataset": dataset, "seed": seed},
                        "result": {"metrics_sha256": "a" * 64, "elapsed_seconds": 1, "metrics": {
                            "split_hash": "a" * 64, "protocol_hash": "b" * 64, "partition": "validation",
                            "metrics": {"accuracy": 0.5 if variant.variant_id == self.spec.baseline else 0.8},
                        }},
                    })
        return rows

    def test_missing_or_protocol_mismatched_pairs_prevent_positive_verdict(self):
        rows = self.rows()
        self.assertEqual(ReviewWorker().review(self.spec, rows)["comparisons"][0]["verdict"], "promising")
        rows[0]["result"]["metrics"]["protocol_hash"] = "changed"
        report = ReviewWorker().review(self.spec, rows)
        self.assertEqual(report["comparisons"][0]["verdict"], "inconclusive")
        self.assertEqual(report["comparisons"][0]["paired_runs"], 4)
        rows = self.rows()[1:]
        self.assertEqual(ReviewWorker().review(self.spec, rows)["comparisons"][0]["verdict"], "inconclusive")

    def test_metric_direction_and_seed_uncertainty(self):
        report = ReviewWorker().review(replace(self.spec, direction="minimize"), self.rows())
        row = report["comparisons"][0]
        self.assertAlmostEqual(row["oriented_delta_mean"], -0.3)
        self.assertEqual(row["verdict"], "deprioritize")
        one = replace(self.spec, seeds=(11,))
        rows = [r for r in self.rows() if r["definition"]["seed"] == 11]
        self.assertIsNone(ReviewWorker().review(one, rows)["comparisons"][0]["bootstrap_95_interval"])
        self.assertEqual(ReviewWorker().review(one, rows)["comparisons"][0]["verdict"], "inconclusive")

    def test_post_execution_metric_tampering_is_rejected(self):
        work = self.root / "work"
        work.mkdir()
        artifact = work / "metrics.json"
        artifact.write_text("{}")
        result = {"metrics_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        artifact.write_text('{"accuracy":1}')
        with self.assertRaisesRegex(ValueError, "differs from"):
            validate_metrics({}, result, self.root)

    def test_context_reuses_offline_rag_with_exact_code_hashes(self):
        source = self.root / "source"
        source.mkdir()
        (source / "train.py").write_text("def train():\n    # normalize features\n    pass\n")
        evidence = self.root / "paper.md"
        evidence.write_text("# Feature normalization\nNormalize training features before fitting the classifier.")
        context = build_context(source, "normalize features training classifier", evidence_paths=[str(evidence)])
        self.assertFalse(context["model_called"])
        self.assertTrue(context["evidence"])
        self.assertEqual(context["files"][0]["sha256"],
                         hashlib.sha256((source / "train.py").read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            build_context(source, "features", files=["../paper.md"])


if __name__ == "__main__":
    unittest.main()
