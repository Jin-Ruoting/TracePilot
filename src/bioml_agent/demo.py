from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_loop import BioMLResearchAgent
from .audit import audit_run
from .benchmark import load_eval_cases, run_eval_corpus
from .comparison import run_comparison
from .literature import FrozenLiteratureProvider
from .serialization import to_jsonable
from .state_graph import initial_state, run_state_graph, state_to_run
from .tools import build_default_registry

PROMOTION_QUESTION = "Should we try retrieval-aware conditioning for promoter expression prediction?"
ABSTENTION_QUESTION = "Should we try unsupported quantum graph retrieval for promoter expression prediction?"
ABSTENTION_SNAPSHOT = "benchmarks/abstention_literature_snapshot.jsonl"


@dataclass(frozen=True)
class DemoScenario:
    name: str
    question: str
    recommendation: str
    audit_decision: str
    review_status: str
    top_candidate: str
    trace_tools: tuple[str, ...]


@dataclass(frozen=True)
class DemoBenchmark:
    corpus: str
    passed: bool
    passed_count: int
    total: int


@dataclass(frozen=True)
class DemoComparison:
    corpus: str
    agent_passed_count: int
    baseline_passed_count: int
    improved_count: int
    total: int

    @property
    def passed(self) -> bool:
        return self.agent_passed_count == self.total and self.improved_count > 0


@dataclass(frozen=True)
class DemoCheckpoint:
    failure_node: str
    failure_type: str
    resumed_recommendation: str
    resumed_review_status: str
    trace_tools: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioDemo:
    passed: bool
    scenarios: tuple[DemoScenario, ...]
    benchmarks: tuple[DemoBenchmark, ...]
    comparison: DemoComparison
    checkpoint: DemoCheckpoint


def run_portfolio_demo() -> PortfolioDemo:
    promotion = _scenario_from_run(
        name="promotion_path",
        question=PROMOTION_QUESTION,
        agent=BioMLResearchAgent(),
    )
    abstention_registry = build_default_registry(
        literature_provider=FrozenLiteratureProvider(ABSTENTION_SNAPSHOT)
    )
    abstention = _scenario_from_run(
        name="abstention_path",
        question=ABSTENTION_QUESTION,
        agent=BioMLResearchAgent(registry=abstention_registry),
    )
    benchmarks = tuple(
        _benchmark_summary(path)
        for path in (
            "benchmarks/eval_cases.jsonl",
            "benchmarks/live_eval_cases.jsonl",
            "benchmarks/real_literature_eval_cases.jsonl",
            "benchmarks/abstention_eval_cases.jsonl",
            "benchmarks/source_poisoning_eval_cases.jsonl",
            "benchmarks/stress_eval_cases.jsonl",
        )
    )
    comparison = _comparison_summary("benchmarks/eval_cases.jsonl")
    checkpoint = _checkpoint_demo()
    passed = (
        promotion.audit_decision == "ready_for_human_review"
        and abstention.audit_decision == "record_abstention"
        and all(item.passed for item in benchmarks)
        and comparison.passed
        and checkpoint.failure_node == "read_experiment_context"
        and checkpoint.resumed_review_status == "requires_approval"
    )
    return PortfolioDemo(
        passed=passed,
        scenarios=(promotion, abstention),
        benchmarks=benchmarks,
        comparison=comparison,
        checkpoint=checkpoint,
    )


def demo_to_dict(demo: PortfolioDemo) -> dict[str, Any]:
    return to_jsonable(demo)


def format_portfolio_demo(demo: PortfolioDemo) -> str:
    lines = [
        "# TracePilot Portfolio Demo",
        f"Passed: {demo.passed}",
        "",
        "## Scenario Results",
    ]
    for scenario in demo.scenarios:
        lines.extend(
            [
                f"- {scenario.name}",
                f"  Question: {scenario.question}",
                f"  Recommendation: {scenario.recommendation}",
                f"  Audit decision: {scenario.audit_decision}",
                f"  Review status: {scenario.review_status}",
                f"  Top candidate: {scenario.top_candidate or '(none)'}",
                f"  Trace tools: {', '.join(scenario.trace_tools)}",
            ]
        )
    lines.extend(["", "## Benchmark Results"])
    for benchmark in demo.benchmarks:
        lines.append(
            f"- {benchmark.corpus}: {benchmark.passed_count}/{benchmark.total} passed "
            f"({'PASS' if benchmark.passed else 'FAIL'})"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            f"- Corpus: {demo.comparison.corpus}",
            f"- Agent passed: {demo.comparison.agent_passed_count}/{demo.comparison.total}",
            f"- Naive baseline passed: {demo.comparison.baseline_passed_count}/{demo.comparison.total}",
            f"- Improved cases: {demo.comparison.improved_count}/{demo.comparison.total}",
        ]
    )
    lines.extend(
        [
            "",
            "## Failure Recovery",
            f"- Failed node: {demo.checkpoint.failure_node}",
            f"- Failure type: {demo.checkpoint.failure_type}",
            f"- Resumed recommendation: {demo.checkpoint.resumed_recommendation}",
            f"- Resumed review status: {demo.checkpoint.resumed_review_status}",
            f"- Trace tools after resume: {', '.join(demo.checkpoint.trace_tools)}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the portfolio-grade TracePilot demo.")
    parser.add_argument("--json", action="store_true", help="Emit demo result JSON.")
    parser.add_argument("--out", help="Write demo result to this path.")
    args = parser.parse_args()

    demo = run_portfolio_demo()
    text = (
        json.dumps(demo_to_dict(demo), indent=2, ensure_ascii=False)
        if args.json
        else format_portfolio_demo(demo)
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if demo.passed else 1)


def _scenario_from_run(
    name: str,
    question: str,
    agent: BioMLResearchAgent,
) -> DemoScenario:
    run = agent.run(question)
    audit = audit_run(run)
    return DemoScenario(
        name=name,
        question=question,
        recommendation=run.plan.recommendation,
        audit_decision=audit.decision,
        review_status=run.review.status,
        top_candidate=run.candidates[0].paper_id if run.candidates else "",
        trace_tools=tuple(event.tool for event in run.trace),
    )


def _benchmark_summary(path: str) -> DemoBenchmark:
    result = run_eval_corpus(load_eval_cases(path))
    return DemoBenchmark(
        corpus=path,
        passed=result.passed,
        passed_count=result.passed_count,
        total=result.total,
    )


def _comparison_summary(path: str) -> DemoComparison:
    result = run_comparison(load_eval_cases(path))
    return DemoComparison(
        corpus=path,
        agent_passed_count=result.agent_passed_count,
        baseline_passed_count=result.baseline_passed_count,
        improved_count=result.improved_count,
        total=result.total,
    )


def _checkpoint_demo() -> DemoCheckpoint:
    registry = build_default_registry()
    failed = run_state_graph(
        registry,
        initial_state(PROMOTION_QUESTION, project_name="unknown"),
        capture_errors=True,
    )
    if failed.failure is None:
        raise RuntimeError("checkpoint demo expected read_experiment_context to fail")
    failure_node = failed.failure.node
    failure_type = failed.failure.error_type

    failed.project_name = "EpiCETRA"
    resumed = run_state_graph(registry, failed)
    run = state_to_run(resumed)
    return DemoCheckpoint(
        failure_node=failure_node,
        failure_type=failure_type,
        resumed_recommendation=run.plan.recommendation,
        resumed_review_status=run.review.status,
        trace_tools=tuple(event.tool for event in run.trace),
    )


if __name__ == "__main__":
    main()
