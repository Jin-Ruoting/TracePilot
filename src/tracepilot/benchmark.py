from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .agent import TracePilotAgent
from .models import EvidenceClaim, TaskRequest
from .rag import (
    Ingestor,
    RagIndex,
    check_citation_coverage,
    check_faithfulness,
)


@dataclass(frozen=True)
class MethodResult:
    method: str
    status: str
    summary: str
    required_terms_found: tuple[str, ...]
    required_terms_missing: tuple[str, ...]
    citation_coverage: float
    faithfulness: float
    latency_ms: float
    passed: bool


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    goal: str
    expected_decision: str
    methods: tuple[MethodResult, ...]
    passed: bool


@dataclass(frozen=True)
class BenchmarkReport:
    cases: tuple[BenchmarkCaseResult, ...]
    passed_cases: int
    total_cases: int
    passed: bool


def run_benchmark(
    cases_path: str | Path,
    sources_path: str | Path,
) -> BenchmarkReport:
    resolved_cases_path = Path(cases_path).expanduser().resolve()
    case_records = _read_jsonl(resolved_cases_path)
    source_records = _read_jsonl(sources_path)
    project_root = resolved_cases_path.parent.parent
    results: list[BenchmarkCaseResult] = []
    for raw_case in case_records:
        case = dict(raw_case)
        workspace = Path(str(case["workspace"]))
        if not workspace.is_absolute():
            case["workspace"] = str((project_root / workspace).resolve())
        selected_sources = [
            record
            for record in source_records
            if record.get("source_id") in set(case.get("source_ids", []))
        ]
        required_terms = tuple(str(item) for item in case.get("required_terms", []))
        methods = (
            _one_shot_baseline(case, required_terms),
            _single_query_rag(case, selected_sources, required_terms),
            _full_tracepilot(case, selected_sources, required_terms),
        )
        full_result = next(item for item in methods if item.method == "tracepilot")
        results.append(
            BenchmarkCaseResult(
                case_id=str(case["case_id"]),
                goal=str(case["goal"]),
                expected_decision=str(case.get("expected_decision", "")),
                methods=methods,
                passed=full_result.passed,
            )
        )
    passed_cases = sum(item.passed for item in results)
    return BenchmarkReport(
        cases=tuple(results),
        passed_cases=passed_cases,
        total_cases=len(results),
        passed=passed_cases == len(results),
    )


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return asdict(report)


def render_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# TracePilot Generic Benchmark",
        "",
        f"- Cases: `{report.total_cases}`",
        f"- Full TracePilot passed: `{report.passed_cases}/{report.total_cases}`",
        f"- Overall: `{'pass' if report.passed else 'fail'}`",
        "",
        "| Case | Method | Terms | Citations | Faithfulness | Status | Result |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case in report.cases:
        for method in case.methods:
            terms_total = (
                len(method.required_terms_found) + len(method.required_terms_missing)
            )
            lines.append(
                f"| {case.case_id} | {method.method} | "
                f"{len(method.required_terms_found)}/{terms_total} | "
                f"{method.citation_coverage:.2f} | {method.faithfulness:.2f} | "
                f"{method.status} | {'pass' if method.passed else 'fail'} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run generic TracePilot benchmarks.")
    parser.add_argument("cases")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(args.cases, args.sources)
    rendered = (
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False)
        if args.json
        else render_markdown(report)
    )
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


def _one_shot_baseline(
    case: Mapping[str, Any],
    required_terms: tuple[str, ...],
) -> MethodResult:
    started = time.perf_counter()
    summary = (
        "No project or source evidence was supplied to the one-shot baseline; "
        "the baseline abstains."
    )
    found, missing = _term_hits(summary, required_terms)
    return MethodResult(
        method="one_shot",
        status="abstained",
        summary=summary,
        required_terms_found=found,
        required_terms_missing=missing,
        citation_coverage=0.0,
        faithfulness=0.0,
        latency_ms=(time.perf_counter() - started) * 1000,
        passed=False,
    )


def _single_query_rag(
    case: Mapping[str, Any],
    sources: list[Mapping[str, Any]],
    required_terms: tuple[str, ...],
) -> MethodResult:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        with RagIndex(Path(tmp) / "rag.db") as index:
            documents = Ingestor().ingest_frozen(sources)
            index.index_documents(documents)
            chunks = index.search(str(case["goal"]), top_k=8)
            claims = [
                EvidenceClaim(
                    text=chunk.text,
                    evidence_ids=(chunk.chunk_id,),
                    supported=True,
                    confidence=1.0,
                )
                for chunk in chunks[:2]
            ]
            summary = " ".join(claim.text for claim in claims)
            coverage = check_citation_coverage(claims, chunks)
            faithfulness = check_faithfulness(claims, chunks)
    found, missing = _term_hits(summary, required_terms)
    passed = not missing and coverage.passed and faithfulness.passed
    return MethodResult(
        method="single_query_rag",
        status="completed" if claims else "abstained",
        summary=summary,
        required_terms_found=found,
        required_terms_missing=missing,
        citation_coverage=coverage.coverage,
        faithfulness=faithfulness.score,
        latency_ms=(time.perf_counter() - started) * 1000,
        passed=passed,
    )


def _full_tracepilot(
    case: Mapping[str, Any],
    sources: list[Mapping[str, Any]],
    required_terms: tuple[str, ...],
) -> MethodResult:
    started = time.perf_counter()
    workspace = Path(str(case["workspace"]))
    if not workspace.is_absolute():
        workspace = (Path.cwd() / workspace).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        request = TaskRequest(
            goal=str(case["goal"]),
            workspace_id=f"benchmark:{case['case_id']}",
            workspace_path=str(workspace),
            execution_policy="read_only",
            metadata={"benchmark_case": case["case_id"]},
        )
        with TracePilotAgent(Path(tmp) / "tracepilot.db") as agent:
            run = agent.run(request, frozen_sources=sources)
    found, missing = _term_hits(run.summary, required_terms)
    evaluation = run.metadata.get("evaluation", {})
    coverage = evaluation.get("citation_coverage", {})
    faithfulness = evaluation.get("faithfulness", {})
    passed = (
        run.status.value == "completed"
        and not missing
        and bool(coverage.get("passed"))
        and bool(faithfulness.get("passed"))
    )
    return MethodResult(
        method="tracepilot",
        status=run.status.value,
        summary=run.summary,
        required_terms_found=found,
        required_terms_missing=missing,
        citation_coverage=float(coverage.get("coverage", 0.0)),
        faithfulness=float(faithfulness.get("score", 0.0)),
        latency_ms=(time.perf_counter() - started) * 1000,
        passed=passed,
    )


def _term_hits(
    text: str,
    required_terms: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lowered = text.casefold()
    found: list[str] = []
    missing: list[str] = []
    for term in required_terms:
        (found if term.casefold() in lowered else missing).append(term)
    return tuple(found), tuple(missing)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
