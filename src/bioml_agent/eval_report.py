from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkCaseResult, BenchmarkResult, load_eval_cases, run_eval_corpus
from .comparison import ComparisonResult, run_comparison
from .serialization import to_jsonable

DEFAULT_CORPORA = (
    "benchmarks/eval_cases.jsonl",
    "benchmarks/live_eval_cases.jsonl",
    "benchmarks/real_literature_eval_cases.jsonl",
    "benchmarks/abstention_eval_cases.jsonl",
    "benchmarks/source_poisoning_eval_cases.jsonl",
    "benchmarks/stress_eval_cases.jsonl",
)


@dataclass(frozen=True)
class CorpusCoverage:
    corpus: str
    passed: bool
    passed_count: int
    total: int
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class FamilyCoverage:
    family: str
    passed_count: int
    total: int


@dataclass(frozen=True)
class CheckCoverage:
    check_name: str
    passed_count: int
    total: int
    failed_cases: tuple[str, ...]


@dataclass(frozen=True)
class GuardrailCoverage:
    name: str
    covered: bool
    checks: tuple[str, ...]
    cases: tuple[str, ...]


@dataclass(frozen=True)
class EvalCoverageReport:
    passed: bool
    total_cases: int
    passed_cases: int
    corpora: tuple[CorpusCoverage, ...]
    families: tuple[FamilyCoverage, ...]
    checks: tuple[CheckCoverage, ...]
    guardrails: tuple[GuardrailCoverage, ...]
    comparison: ComparisonResult | None = None


def run_eval_coverage_report(
    corpora: tuple[str, ...] = DEFAULT_CORPORA,
    comparison_corpus: str = "benchmarks/eval_cases.jsonl",
    cache_dir: str | Path = ".cache/literature",
    include_comparison: bool = True,
) -> EvalCoverageReport:
    benchmark_results: list[tuple[str, BenchmarkResult]] = [
        (corpus, run_eval_corpus(load_eval_cases(corpus), cache_dir=cache_dir))
        for corpus in corpora
    ]
    all_case_results = [
        case_result
        for _, benchmark_result in benchmark_results
        for case_result in benchmark_result.case_results
    ]
    comparison = (
        run_comparison(load_eval_cases(comparison_corpus), cache_dir=cache_dir)
        if include_comparison
        else None
    )
    total_cases = len(all_case_results)
    passed_cases = sum(1 for case_result in all_case_results if case_result.passed)
    return EvalCoverageReport(
        passed=passed_cases == total_cases and _comparison_passed(comparison),
        total_cases=total_cases,
        passed_cases=passed_cases,
        corpora=tuple(_corpus_coverage(corpus, result) for corpus, result in benchmark_results),
        families=_family_coverage(all_case_results),
        checks=_check_coverage(all_case_results),
        guardrails=_guardrail_coverage(all_case_results),
        comparison=comparison,
    )


def eval_report_to_dict(report: EvalCoverageReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "corpora": to_jsonable(report.corpora),
        "families": to_jsonable(report.families),
        "checks": to_jsonable(report.checks),
        "guardrails": to_jsonable(report.guardrails),
        "comparison": _comparison_to_dict(report.comparison),
    }


def write_eval_coverage_report(report: EvalCoverageReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_eval_coverage_report(report), encoding="utf-8")


def format_eval_coverage_report(report: EvalCoverageReport) -> str:
    lines = [
        "# TracePilot Eval Coverage Report",
        "",
        "## Summary",
        f"- Passed: `{report.passed}`",
        f"- Cases: `{report.passed_cases}/{report.total_cases}`",
        f"- Corpora: `{sum(1 for item in report.corpora if item.passed)}/{len(report.corpora)}`",
    ]
    if report.comparison is not None:
        lines.extend(
            [
                f"- Agent vs naive baseline: `{report.comparison.agent_passed_count}/{report.comparison.total}` "
                f"vs `{report.comparison.baseline_passed_count}/{report.comparison.total}`",
                f"- Improved cases: `{report.comparison.improved_count}/{report.comparison.total}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Corpus Coverage",
            "| Corpus | Passed | Cases | Case IDs |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for corpus in report.corpora:
        lines.append(
            f"| `{corpus.corpus}` | `{corpus.passed}` | "
            f"{corpus.passed_count}/{corpus.total} | {_join(corpus.case_ids)} |"
        )

    lines.extend(
        [
            "",
            "## Case Family Coverage",
            "| Family | Cases |",
            "| --- | ---: |",
        ]
    )
    for family in report.families:
        lines.append(f"| `{family.family}` | {family.passed_count}/{family.total} |")

    lines.extend(
        [
            "",
            "## Guardrail Coverage",
            "| Guardrail | Covered | Checks | Cases |",
            "| --- | --- | --- | --- |",
        ]
    )
    for guardrail in report.guardrails:
        lines.append(
            f"| `{guardrail.name}` | `{guardrail.covered}` | "
            f"{_join(guardrail.checks)} | {_join(guardrail.cases)} |"
        )

    lines.extend(
        [
            "",
            "## Check Coverage",
            "| Check | Passed | Failed Cases |",
            "| --- | ---: | --- |",
        ]
    )
    for check in report.checks:
        failed_cases = _join(check.failed_cases) if check.failed_cases else "-"
        lines.append(f"| `{check.check_name}` | {check.passed_count}/{check.total} | {failed_cases} |")

    if report.comparison is not None:
        lines.extend(
            [
                "",
                "## Baseline Delta",
                "| Case | Agent | Baseline | Improvement | Baseline Failed Checks |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for case_result in report.comparison.case_results:
            failed = tuple(check.name for check in case_result.baseline_checks if not check.passed)
            lines.append(
                f"| `{case_result.case.case_id}` | `{case_result.agent_passed}` | "
                f"`{case_result.baseline_passed}` | `{case_result.improved}` | {_join(failed) or '-'} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TracePilot eval coverage across benchmark corpora.")
    parser.add_argument(
        "corpora",
        nargs="*",
        default=list(DEFAULT_CORPORA),
        help="Benchmark JSONL corpora to include.",
    )
    parser.add_argument(
        "--comparison-corpus",
        default="benchmarks/eval_cases.jsonl",
        help="Corpus used for naive baseline comparison.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/literature",
        help="Cache directory for live literature providers.",
    )
    parser.add_argument("--no-comparison", action="store_true", help="Skip naive baseline comparison.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--out", help="Write report to this path.")
    args = parser.parse_args()

    report = run_eval_coverage_report(
        corpora=tuple(args.corpora),
        comparison_corpus=args.comparison_corpus,
        cache_dir=args.cache_dir,
        include_comparison=not args.no_comparison,
    )
    text = (
        json.dumps(eval_report_to_dict(report), indent=2, ensure_ascii=False)
        if args.json
        else format_eval_coverage_report(report)
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    raise SystemExit(0 if report.passed else 1)


def _corpus_coverage(corpus: str, result: BenchmarkResult) -> CorpusCoverage:
    return CorpusCoverage(
        corpus=corpus,
        passed=result.passed,
        passed_count=result.passed_count,
        total=result.total,
        case_ids=tuple(case_result.case.case_id for case_result in result.case_results),
    )


def _family_coverage(case_results: list[BenchmarkCaseResult]) -> tuple[FamilyCoverage, ...]:
    totals: dict[str, int] = defaultdict(int)
    passed: dict[str, int] = defaultdict(int)
    for case_result in case_results:
        family = _case_family(case_result)
        totals[family] += 1
        if case_result.passed:
            passed[family] += 1
    return tuple(
        FamilyCoverage(family=family, passed_count=passed[family], total=totals[family])
        for family in sorted(totals)
    )


def _check_coverage(case_results: list[BenchmarkCaseResult]) -> tuple[CheckCoverage, ...]:
    totals: dict[str, int] = defaultdict(int)
    passed: dict[str, int] = defaultdict(int)
    failed_cases: dict[str, list[str]] = defaultdict(list)
    for case_result in case_results:
        for check in case_result.checks:
            totals[check.name] += 1
            if check.passed:
                passed[check.name] += 1
            else:
                failed_cases[check.name].append(case_result.case.case_id)
    return tuple(
        CheckCoverage(
            check_name=name,
            passed_count=passed[name],
            total=totals[name],
            failed_cases=tuple(failed_cases[name]),
        )
        for name in sorted(totals)
    )


def _guardrail_coverage(case_results: list[BenchmarkCaseResult]) -> tuple[GuardrailCoverage, ...]:
    guardrails = (
        ("tool trace completeness", ("tool_trace_complete",), _all_cases),
        ("evidence provenance", ("evidence_provenance_traceable", "top_candidate_source_url", "top_candidate_venue"), _all_cases),
        ("claim citation binding", ("claim_citations_bound",), _all_cases),
        ("source ref format", ("source_refs_valid",), _all_cases),
        ("human approval gate", ("review_checkpoint_valid", "expected_review_status"), _all_cases),
        ("split and leakage gate", ("split_gate_present", "gate_contains:chromosome-aware split", "gate_contains:sample-random"), _all_cases),
        ("promotion support", ("promotion_supported", "expected_recommendation"), _all_cases),
        ("abstention quality", ("expected_recommendation", "matrix_contains:do not schedule experiment"), _abstention_cases),
        ("source poisoning", ("source_poisoning_guarded",), _source_poisoning_cases),
    )
    case_check_names = {
        case_result.case.case_id: {check.name for check in case_result.checks}
        for case_result in case_results
    }
    coverage: list[GuardrailCoverage] = []
    for guardrail, checks, selector in guardrails:
        cases = tuple(
            case_result.case.case_id
            for case_result in case_results
            if selector(case_result)
            and any(check in case_check_names[case_result.case.case_id] for check in checks)
        )
        coverage.append(
            GuardrailCoverage(
                name=guardrail,
                covered=bool(cases),
                checks=checks,
                cases=cases,
            )
        )
    return tuple(coverage)


def _all_cases(case_result: BenchmarkCaseResult) -> bool:
    return True


def _abstention_cases(case_result: BenchmarkCaseResult) -> bool:
    return (
        case_result.case.expected_recommendation == "do not run"
        or case_result.run.plan.recommendation == "do not run"
    )


def _source_poisoning_cases(case_result: BenchmarkCaseResult) -> bool:
    return _case_family(case_result) == "source_poisoning"


def _case_family(case_result: BenchmarkCaseResult) -> str:
    case_id = case_result.case.case_id
    if "poison" in case_id:
        return "source_poisoning"
    if case_id.startswith("live_"):
        return "real_literature"
    if "abstain" in case_id:
        return "abstention"
    if "retrieval" in case_id:
        return "retrieval"
    if "graph" in case_id:
        return "graph"
    if "moe" in case_id:
        return "moe"
    if "contrast" in case_id:
        return "contrastive"
    return "other"


def _comparison_passed(comparison: ComparisonResult | None) -> bool:
    if comparison is None:
        return True
    return comparison.agent_passed_count == comparison.total and comparison.improved_count > 0


def _comparison_to_dict(comparison: ComparisonResult | None) -> dict[str, Any] | None:
    if comparison is None:
        return None
    return {
        "total": comparison.total,
        "agent_passed_count": comparison.agent_passed_count,
        "baseline_passed_count": comparison.baseline_passed_count,
        "improved_count": comparison.improved_count,
        "cases": [
            {
                "case_id": case_result.case.case_id,
                "agent_passed": case_result.agent_passed,
                "baseline_passed": case_result.baseline_passed,
                "improved": case_result.improved,
                "agent_recommendation": case_result.agent_run.plan.recommendation,
                "baseline_recommendation": case_result.baseline_run.plan.recommendation,
                "baseline_failed_checks": [
                    check.name for check in case_result.baseline_checks if not check.passed
                ],
            }
            for case_result in comparison.case_results
        ],
    }


def _join(items: tuple[str, ...]) -> str:
    return ", ".join(f"`{item}`" for item in items)


if __name__ == "__main__":
    main()
