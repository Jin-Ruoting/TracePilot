from __future__ import annotations

from pathlib import Path

from .models import AgentRun, EvidenceChunk, EvidenceClaim
from .serialization import to_jsonable


def render_markdown_report(run: AgentRun) -> str:
    chunks = {
        item.chunk_id: item
        for item in run.evidence
        if isinstance(item, EvidenceChunk)
    }
    claims = [item for item in run.evidence if isinstance(item, EvidenceClaim)]
    lines = [
        "# TracePilot Technical Decision",
        "",
        f"- Task: `{run.task_id}`",
        f"- Status: `{run.status.value}`",
        f"- Workspace: `{run.request.workspace_path}`",
        "",
        "## Goal",
        "",
        run.request.goal,
        "",
        "## Decision",
        "",
        run.summary or "No supported conclusion was produced.",
        "",
        "## Evidence",
        "",
    ]
    if not claims:
        lines.append(
            "No evidence-backed claim was produced. Treat this run as an abstention."
        )
    for claim in claims:
        support = "supported" if claim.supported else "uncertain"
        lines.append(
            f"- **{claim.text}** ({support}, confidence `{claim.confidence:.2f}`)"
        )
        if claim.uncertainty:
            lines.append(f"  Uncertainty: {claim.uncertainty}")
        for evidence_id in claim.evidence_ids:
            chunk = chunks.get(evidence_id)
            if chunk is None:
                lines.append(f"  Citation: missing chunk `{evidence_id}`")
                continue
            location = _location(chunk)
            lines.append(
                f"  Citation: [{chunk.title or chunk.document_id}]"
                f"({chunk.source_uri or '#'}){location} (`{chunk.chunk_id}`)"
            )
    lines.extend(["", "## Plan", ""])
    for step in run.plan.steps:
        status = step.metadata.get("status", "planned")
        lines.append(
            f"- `{step.step_id}` [{step.effect_class}/{status}] "
            f"{step.description} via `{step.tool_name}`"
        )
    lines.extend(["", "## Approval And Execution", ""])
    if not run.approvals:
        lines.append("- No mutation approval was requested.")
    else:
        for approval in run.approvals:
            lines.append(
                f"- `{approval.tool_name}`: `{approval.status.value}` "
                f"(arguments `{approval.arguments_hash[:12]}`)"
            )
    failures = [result for result in run.tool_results if not result.success]
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            lines.append(f"- `{result.step_id}` / `{result.tool_name}`: {result.error}")
    lines.extend(
        [
            "",
            "## Trace",
            "",
            f"- Events: `{len(run.trace)}`",
            f"- Tool results: `{len(run.tool_results)}`",
            f"- Plan revision: `{run.plan.revision}`",
            "",
        ]
    )
    return "\n".join(lines)


def report_payload(run: AgentRun) -> dict:
    chunks = [
        to_jsonable(item)
        for item in run.evidence
        if isinstance(item, EvidenceChunk)
    ]
    claims = [
        to_jsonable(item)
        for item in run.evidence
        if isinstance(item, EvidenceClaim)
    ]
    return {
        "schema_version": 2,
        "task_id": run.task_id,
        "run_id": run.run_id,
        "status": run.status.value,
        "goal": run.request.goal,
        "summary": run.summary,
        "claims": claims,
        "evidence_chunks": chunks,
        "plan": to_jsonable(run.plan),
        "approvals": to_jsonable(run.approvals),
        "tool_results": to_jsonable(run.tool_results),
        "trace": to_jsonable(run.trace),
    }


def write_markdown_report(run: AgentRun, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(run), encoding="utf-8")
    return output


def _location(chunk: EvidenceChunk) -> str:
    if chunk.page is not None:
        return f", page {chunk.page}"
    if chunk.start_line is not None and chunk.end_line is not None:
        return f", lines {chunk.start_line}-{chunk.end_line}"
    if chunk.start_line is not None:
        return f", line {chunk.start_line}"
    return ""


__all__ = [
    "render_markdown_report",
    "report_payload",
    "write_markdown_report",
]
