from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_loop import BioMLResearchAgent
from .context import build_context_provider
from .evals import EvalCheck, evaluate_run
from .literature import FrozenLiteratureProvider, build_literature_provider
from .literature import LiteratureProvider
from .schemas import AgentRun
from .tools import build_default_registry


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    project_name: str = "EpiCETRA"
    context_file: str = ""
    experiment_log: str = ""
    literature_source: str = "fixture"
    literature_snapshot: str = ""
    expected_recommendation: str = ""
    expected_top_candidate: str = ""
    expected_review_status: str = ""
    expect_no_candidates: bool = False
    required_matrix_terms: tuple[str, ...] = ()
    required_gate_terms: tuple[str, ...] = ()
    require_source_url: bool = False
    require_venue: bool = False
    require_citation_count: bool = False


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case: BenchmarkCase
    passed: bool
    checks: tuple[EvalCheck, ...]
    run: AgentRun


@dataclass(frozen=True)
class BenchmarkResult:
    passed: bool
    case_results: tuple[BenchmarkCaseResult, ...]

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.case_results if result.passed)


def load_eval_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw = json.loads(stripped)
        cases.append(_case_from_dict(raw, line_number=line_number))
    return cases


def run_eval_corpus(cases: list[BenchmarkCase], cache_dir: str | Path = ".cache/literature") -> BenchmarkResult:
    results = tuple(_run_case(case, cache_dir=cache_dir) for case in cases)
    return BenchmarkResult(
        passed=all(result.passed for result in results),
        case_results=results,
    )


def checks_for_case(case: BenchmarkCase, run: AgentRun) -> tuple[EvalCheck, ...]:
    return (*evaluate_run(run).checks, *_expectation_checks(case, run))


def result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    def top_candidate(case_result: BenchmarkCaseResult) -> dict[str, Any]:
        if not case_result.run.candidates:
            return {}
        candidate = case_result.run.candidates[0]
        return {
            "paper_id": candidate.paper_id,
            "title": candidate.title,
            "source_url": candidate.source_url,
            "doi": candidate.doi,
            "venue": candidate.venue,
            "citation_count": candidate.citation_count,
        }

    return {
        "passed": result.passed,
        "passed_count": result.passed_count,
        "total": result.total,
        "cases": [
            {
                "case_id": case_result.case.case_id,
                "project_name": case_result.case.project_name,
                "context_file": case_result.case.context_file,
                "experiment_log": case_result.case.experiment_log,
                "passed": case_result.passed,
                "recommendation": case_result.run.plan.recommendation,
                "top_candidate": case_result.run.candidates[0].paper_id if case_result.run.candidates else "",
                "top_candidate_metadata": top_candidate(case_result),
                "review": {
                    "checkpoint_id": case_result.run.review.checkpoint_id,
                    "status": case_result.run.review.status,
                    "reason": case_result.run.review.reason,
                    "required_actions": list(case_result.run.review.required_actions),
                },
                "cited_claims": [
                    {
                        "claim": claim.claim,
                        "evidence_ids": list(claim.evidence_ids),
                        "source_refs": list(claim.source_refs),
                    }
                    for claim in case_result.run.plan.cited_claims
                ],
                "checks": [
                    {
                        "name": check.name,
                        "passed": check.passed,
                        "detail": check.detail,
                    }
                    for check in case_result.checks
                ],
            }
            for case_result in result.case_results
        ],
    }


def format_benchmark_result(result: BenchmarkResult) -> str:
    lines = [
        "# TracePilot Benchmark",
        f"Passed: {result.passed}",
        f"Cases: {result.passed_count}/{result.total}",
    ]
    for case_result in result.case_results:
        status = "PASS" if case_result.passed else "FAIL"
        lines.append("")
        lines.append(f"## {status} {case_result.case.case_id}")
        lines.append(f"Question: {case_result.case.question}")
        lines.append(f"Recommendation: {case_result.run.plan.recommendation}")
        lines.append(f"Review: {case_result.run.review.status}")
        if case_result.run.candidates:
            lines.append(f"Top candidate: {case_result.run.candidates[0].paper_id}")
        for check in case_result.checks:
            check_status = "PASS" if check.passed else "FAIL"
            lines.append(f"- {check_status} {check.name}: {check.detail}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TracePilot eval corpus.")
    parser.add_argument(
        "corpus",
        nargs="?",
        default="benchmarks/eval_cases.jsonl",
        help="Path to JSONL eval corpus.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/literature",
        help="Cache directory for live literature providers.",
    )
    parser.add_argument("--json", action="store_true", help="Emit benchmark result JSON.")
    args = parser.parse_args()

    result = run_eval_corpus(load_eval_cases(args.corpus), cache_dir=args.cache_dir)
    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False))
    else:
        print(format_benchmark_result(result))
    raise SystemExit(0 if result.passed else 1)


def _run_case(case: BenchmarkCase, cache_dir: str | Path) -> BenchmarkCaseResult:
    registry = build_default_registry(
        literature_provider=_provider_for_case(case, cache_dir=cache_dir),
        context_provider=build_context_provider(
            context_file=case.context_file or None,
            experiment_log=case.experiment_log or None,
        ),
    )
    run = BioMLResearchAgent(registry=registry).run(case.question, project_name=case.project_name)
    checks = checks_for_case(case, run)
    return BenchmarkCaseResult(
        case=case,
        passed=all(check.passed for check in checks),
        checks=checks,
        run=run,
    )


def _provider_for_case(case: BenchmarkCase, cache_dir: str | Path) -> LiteratureProvider:
    if case.literature_source == "frozen":
        if not case.literature_snapshot:
            raise ValueError(f"case {case.case_id!r} uses frozen literature without literature_snapshot")
        return FrozenLiteratureProvider(case.literature_snapshot)
    return build_literature_provider(
        case.literature_source,
        cache_dir=cache_dir,
        use_cache=True,
    )


def _expectation_checks(case: BenchmarkCase, run: AgentRun) -> tuple[EvalCheck, ...]:
    checks: list[EvalCheck] = []
    if case.expected_recommendation:
        checks.append(
            EvalCheck(
                name="expected_recommendation",
                passed=run.plan.recommendation == case.expected_recommendation,
                detail=f"expected {case.expected_recommendation!r}, got {run.plan.recommendation!r}",
            )
        )
    if case.expected_top_candidate:
        actual_top = run.candidates[0].paper_id if run.candidates else ""
        checks.append(
            EvalCheck(
                name="expected_top_candidate",
                passed=actual_top == case.expected_top_candidate,
                detail=f"expected {case.expected_top_candidate!r}, got {actual_top!r}",
            )
        )
    if case.expect_no_candidates:
        checks.append(
            EvalCheck(
                name="expected_no_candidates",
                passed=not run.candidates,
                detail="expected no retrieved candidates",
            )
        )
    if case.expected_review_status:
        checks.append(
            EvalCheck(
                name="expected_review_status",
                passed=run.review.status == case.expected_review_status,
                detail=f"expected {case.expected_review_status!r}, got {run.review.status!r}",
            )
        )
    if case.require_source_url:
        source_url = run.candidates[0].source_url if run.candidates else ""
        checks.append(
            EvalCheck(
                name="top_candidate_source_url",
                passed=bool(source_url),
                detail="top candidate has source_url" if source_url else "top candidate is missing source_url",
            )
        )
    if case.require_venue:
        venue = run.candidates[0].venue if run.candidates else ""
        checks.append(
            EvalCheck(
                name="top_candidate_venue",
                passed=bool(venue),
                detail="top candidate has venue" if venue else "top candidate is missing venue",
            )
        )
    if case.require_citation_count:
        citation_count = run.candidates[0].citation_count if run.candidates else None
        checks.append(
            EvalCheck(
                name="top_candidate_citation_count",
                passed=citation_count is not None,
                detail=(
                    f"top candidate citation_count={citation_count}"
                    if citation_count is not None
                    else "top candidate is missing citation_count"
                ),
            )
        )
    matrix_text = " ".join(run.plan.matrix).lower()
    for term in case.required_matrix_terms:
        checks.append(
            EvalCheck(
                name=f"matrix_contains:{term}",
                passed=term.lower() in matrix_text,
                detail=f"matrix should contain {term!r}",
            )
        )
    gate_text = " ".join(run.plan.promotion_gate).lower()
    for term in case.required_gate_terms:
        checks.append(
            EvalCheck(
                name=f"gate_contains:{term}",
                passed=term.lower() in gate_text,
                detail=f"gate should contain {term!r}",
            )
        )
    return tuple(checks)


def _case_from_dict(raw: dict[str, Any], line_number: int) -> BenchmarkCase:
    try:
        return BenchmarkCase(
            case_id=str(raw["case_id"]),
            question=str(raw["question"]),
            project_name=str(raw.get("project_name", "EpiCETRA")),
            context_file=str(raw.get("context_file", "")),
            experiment_log=str(raw.get("experiment_log", "")),
            literature_source=str(raw.get("literature_source", "fixture")),
            literature_snapshot=str(raw.get("literature_snapshot", "")),
            expected_recommendation=str(raw.get("expected_recommendation", "")),
            expected_top_candidate=str(raw.get("expected_top_candidate", "")),
            expected_review_status=str(raw.get("expected_review_status", "")),
            expect_no_candidates=bool(raw.get("expect_no_candidates", False)),
            required_matrix_terms=tuple(raw.get("required_matrix_terms", ())),
            required_gate_terms=tuple(raw.get("required_gate_terms", ())),
            require_source_url=bool(raw.get("require_source_url", False)),
            require_venue=bool(raw.get("require_venue", False)),
            require_citation_count=bool(raw.get("require_citation_count", False)),
        )
    except KeyError as exc:
        raise ValueError(f"invalid eval case at line {line_number}: missing {exc}") from exc


if __name__ == "__main__":
    main()
