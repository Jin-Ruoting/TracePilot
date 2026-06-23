from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .schemas import AgentRun

FetchStatus = tuple[int | None, str]
Fetcher = Callable[[str], FetchStatus]


@dataclass(frozen=True)
class SourceResolution:
    ref: str
    normalized_url: str
    source_type: str
    status: str
    http_status: int | None
    detail: str


@dataclass(frozen=True)
class SourceResolutionReport:
    passed: bool
    results: tuple[SourceResolution, ...]


def validate_run_sources(
    run: AgentRun,
    fetcher: Fetcher | None = None,
) -> SourceResolutionReport:
    refs = _collect_source_refs(run)
    results = tuple(_resolve_ref(ref, fetcher=fetcher or _fetch_status) for ref in refs)
    return SourceResolutionReport(
        passed=all(result.status in {"ok", "skipped"} for result in results),
        results=results,
    )


def report_to_dict(report: SourceResolutionReport) -> dict[str, object]:
    return {
        "passed": report.passed,
        "results": [asdict(result) for result in report.results],
    }


def write_source_resolution_report(report: SourceResolutionReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def format_source_resolution_report(report: SourceResolutionReport) -> str:
    lines = [
        "## Source Resolution",
        f"Passed: {report.passed}",
    ]
    for result in report.results:
        http_status = "" if result.http_status is None else f" HTTP {result.http_status}"
        lines.append(f"- {result.status.upper()} {result.ref}{http_status}: {result.detail}")
    return "\n".join(lines)


def _collect_source_refs(run: AgentRun) -> tuple[str, ...]:
    refs: list[str] = []
    for candidate in run.candidates:
        refs.extend(_refs_from_values(candidate.source_url, candidate.doi))
    for evidence in run.evidence:
        refs.extend(_refs_from_values(evidence.source_url, evidence.doi))
    for claim in run.plan.cited_claims:
        refs.extend(claim.source_refs)
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def _refs_from_values(source_url: str, doi: str) -> tuple[str, ...]:
    refs = []
    if source_url:
        refs.append(source_url)
    if doi:
        refs.append(f"doi:{doi}")
    return tuple(refs)


def _resolve_ref(ref: str, fetcher: Fetcher) -> SourceResolution:
    normalized_url, source_type = _normalize_ref(ref)
    if source_type == "fixture":
        return SourceResolution(
            ref=ref,
            normalized_url=normalized_url,
            source_type=source_type,
            status="skipped",
            http_status=None,
            detail="fixture source is offline-only",
        )
    http_status, detail = fetcher(normalized_url)
    ok = http_status is not None and 200 <= http_status < 400
    return SourceResolution(
        ref=ref,
        normalized_url=normalized_url,
        source_type=source_type,
        status="ok" if ok else "failed",
        http_status=http_status,
        detail=detail,
    )


def _normalize_ref(ref: str) -> tuple[str, str]:
    if ref.startswith("doi:"):
        return f"https://doi.org/{ref.removeprefix('doi:')}", "doi"
    if ref.startswith("fixture://"):
        return ref, "fixture"
    if "pubmed.ncbi.nlm.nih.gov" in ref:
        return ref, "pubmed"
    if "arxiv.org" in ref:
        return ref, "arxiv"
    return ref, "url"


def _fetch_status(url: str) -> FetchStatus:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "bioml-research-agent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return response.status, "reachable"
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 405}:
            return _fetch_status_get(url)
        return exc.code, exc.reason
    except urllib.error.URLError as exc:
        return None, str(exc.reason)
    except TimeoutError:
        return None, "timeout"


def _fetch_status_get(url: str) -> FetchStatus:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "bioml-research-agent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return response.status, "reachable via GET"
    except urllib.error.HTTPError as exc:
        return exc.code, exc.reason
    except urllib.error.URLError as exc:
        return None, str(exc.reason)
    except TimeoutError:
        return None, "timeout"
