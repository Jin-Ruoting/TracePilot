from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkCase, checks_for_case, load_eval_cases, run_eval_corpus
from .context import build_context_provider
from .evals import EvalCheck
from .literature import FrozenLiteratureProvider, LiteratureProvider, build_literature_provider
from .schemas import (
    AgentRun,
    EvidenceItem,
    ExperimentPlan,
    ResearchQuestion,
    ReviewCheckpoint,
)


@dataclass(frozen=True)
class ComparisonCaseResult:
    case: BenchmarkCase
    agent_run: AgentRun
    agent_checks: tuple[EvalCheck, ...]
    baseline_run: AgentRun
    baseline_checks: tuple[EvalCheck, ...]

    @property
    def agent_passed(self) -> bool:
        return all(check.passed for check in self.agent_checks)

    @property
    def baseline_passed(self) -> bool:
        return all(check.passed for check in self.baseline_checks)

    @property
    def improved(self) -> bool:
        return self.agent_passed and not self.baseline_passed


@dataclass(frozen=True)
class ComparisonResult:
    case_results: tuple[ComparisonCaseResult, ...]

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def agent_passed_count(self) -> int:
        return sum(1 for result in self.case_results if result.agent_passed)

    @property
    def baseline_passed_count(self) -> int:
        return sum(1 for result in self.case_results if result.baseline_passed)

    @property
    def improved_count(self) -> int:
        return sum(1 for result in self.case_results if result.improved)


def run_comparison(cases: list[BenchmarkCase], cache_dir: str | Path = ".cache/literature") -> ComparisonResult:
    agent_result = run_eval_corpus(cases, cache_dir=cache_dir)
    case_results: list[ComparisonCaseResult] = []
    for agent_case_result in agent_result.case_results:
        case = agent_case_result.case
        baseline_run = run_naive_baseline(case, cache_dir=cache_dir)
        case_results.append(
            ComparisonCaseResult(
                case=case,
                agent_run=agent_case_result.run,
                agent_checks=agent_case_result.checks,
                baseline_run=baseline_run,
                baseline_checks=checks_for_case(case, baseline_run),
            )
        )
    return ComparisonResult(case_results=tuple(case_results))


def run_naive_baseline(case: BenchmarkCase, cache_dir: str | Path = ".cache/literature") -> AgentRun:
    question = ResearchQuestion(text=case.question)
    candidates = _provider_for_case(case, cache_dir=cache_dir).search(case.question, top_k=4)
    context = build_context_provider(
        context_file=case.context_file or None,
        experiment_log=case.experiment_log or None,
    ).read(case.project_name)
    evidence = _baseline_evidence(candidates)
    if candidates:
        recommendation = "promote to small controlled screen"
        summary = "Naive baseline recommends running the top retrieved method because literature was found."
        matrix = (
            f"control: {context.baseline}",
            f"variant: add method suggested by {candidates[0].title}",
        )
        promotion_gate = ("improve validation metric",)
        risks = ("No structured source, split, or approval analysis was performed.",)
    else:
        recommendation = "do not run"
        summary = "Naive baseline found no candidate literature."
        matrix = (
            "abstain: do not schedule experiment",
            "follow-up: search manually for closer evidence",
        )
        promotion_gate = ("revisit if literature appears",)
        risks = ("No tool trace explains why no literature was found.",)
    return AgentRun(
        question=question,
        candidates=candidates,
        context=context,
        evidence=evidence,
        plan=ExperimentPlan(
            summary=summary,
            matrix=matrix,
            promotion_gate=promotion_gate,
            risks=risks,
            recommendation=recommendation,
            cited_claims=(),
        ),
        review=ReviewCheckpoint(
            checkpoint_id="baseline:no-review",
            status="not_required",
            reason="Naive baseline has no human approval workflow.",
            required_actions=(),
        ),
        trace=[],
    )


def result_to_dict(result: ComparisonResult) -> dict[str, Any]:
    return {
        "total": result.total,
        "agent_passed_count": result.agent_passed_count,
        "baseline_passed_count": result.baseline_passed_count,
        "improved_count": result.improved_count,
        "cases": [
            {
                "case_id": case_result.case.case_id,
                "agent": _run_summary(case_result.agent_run, case_result.agent_checks),
                "baseline": _run_summary(case_result.baseline_run, case_result.baseline_checks),
                "improved": case_result.improved,
            }
            for case_result in result.case_results
        ],
    }


def format_comparison_result(result: ComparisonResult) -> str:
    lines = [
        "# Baseline vs TracePilot",
        f"Cases: {result.total}",
        f"Agent passed: {result.agent_passed_count}/{result.total}",
        f"Naive baseline passed: {result.baseline_passed_count}/{result.total}",
        f"Improved cases: {result.improved_count}/{result.total}",
    ]
    for case_result in result.case_results:
        lines.append("")
        lines.append(f"## {case_result.case.case_id}")
        lines.append(
            f"Agent: {_status(case_result.agent_passed)} | "
            f"{case_result.agent_run.plan.recommendation} | "
            f"failed checks: {_failed_check_names(case_result.agent_checks)}"
        )
        lines.append(
            f"Baseline: {_status(case_result.baseline_passed)} | "
            f"{case_result.baseline_run.plan.recommendation} | "
            f"failed checks: {_failed_check_names(case_result.baseline_checks)}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TracePilot against a naive baseline.")
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
    parser.add_argument("--json", action="store_true", help="Emit comparison result JSON.")
    args = parser.parse_args()

    result = run_comparison(load_eval_cases(args.corpus), cache_dir=args.cache_dir)
    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False))
    else:
        print(format_comparison_result(result))
    raise SystemExit(0 if result.agent_passed_count == result.total else 1)


def _baseline_evidence(candidates: list[object]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for rank, candidate in enumerate(candidates):
        evidence.append(
            EvidenceItem(
                evidence_id=f"B{rank + 1}",
                paper_id=getattr(candidate, "paper_id"),
                title=getattr(candidate, "title"),
                relevance=0.5,
                transfer_fit="low",
                supports="Retrieved by the same literature provider, but not scored for domain transfer.",
                cautions="No split-aware, source-grounded, or approval-gated analysis was performed.",
            )
        )
    return evidence


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


def _run_summary(run: AgentRun, checks: tuple[EvalCheck, ...]) -> dict[str, Any]:
    return {
        "passed": all(check.passed for check in checks),
        "recommendation": run.plan.recommendation,
        "top_candidate": run.candidates[0].paper_id if run.candidates else "",
        "review_status": run.review.status,
        "matrix": list(run.plan.matrix),
        "promotion_gate": list(run.plan.promotion_gate),
        "trace_tools": [event.tool for event in run.trace],
        "failed_checks": [check.name for check in checks if not check.passed],
    }


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _failed_check_names(checks: tuple[EvalCheck, ...]) -> str:
    failed = [check.name for check in checks if not check.passed]
    return ", ".join(failed) if failed else "(none)"


if __name__ == "__main__":
    main()
