from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any

from .audit import AuditReport, audit_run
from .evals import EvalResult, evaluate_run
from .schemas import AgentRun
from .serialization import run_to_dict
from .source_validation import SourceResolutionReport, validate_run_sources


def format_decision_packet(
    run: AgentRun,
    audit_report: AuditReport | None = None,
    eval_result: EvalResult | None = None,
    source_resolution: SourceResolutionReport | None = None,
) -> str:
    """Render a human-reviewable Markdown packet for one completed agent run."""
    audit = audit_report or audit_run(run)
    evals = eval_result or evaluate_run(run)
    data = run_to_dict(run)
    plan = data["plan"]
    review = data["review"]
    lines = [
        "# TracePilot Decision Packet",
        "",
        "## Decision Summary",
        f"- Question: {_inline(data['question']['text'])}",
        f"- Recommendation: `{_inline(plan['recommendation'])}`",
        f"- Review status: `{_inline(review['status'])}`",
        f"- Audit decision: `{_inline(audit.decision)}`",
        f"- Audit passed: `{audit.passed}`",
        f"- Eval passed: `{evals.passed}`",
        f"- Trace tools: {_inline(_trace_tools(data['trace']))}",
        "",
        "## Evidence Table",
        "| Evidence | Fit | Relevance | Source | Support | Caution |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    if data["evidence"]:
        for item in data["evidence"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _cell(f"{item['evidence_id']} / {item['paper_id']}"),
                        _cell(item["transfer_fit"]),
                        _cell(f"{item['relevance']:.2f}"),
                        _cell(_source_label(item)),
                        _cell(item["supports"]),
                        _cell(item["cautions"]),
                    )
                )
                + " |"
            )
    else:
        lines.append("| - | - | - | - | No evidence retrieved. | Run abstains. |")

    lines.extend(
        [
            "",
            "## Cited Claims",
        ]
    )
    if plan["cited_claims"]:
        for claim in plan["cited_claims"]:
            lines.append(f"- {_inline(claim['claim'])}")
            lines.append(f"  - Evidence ids: {_inline(', '.join(claim['evidence_ids']))}")
            lines.append(f"  - Source refs: {_inline('; '.join(claim['source_refs']))}")
    else:
        lines.append("- No cited claims. This is expected only for no-evidence abstention.")

    lines.extend(
        [
            "",
            "## Experiment Matrix",
            *_bullet_lines(plan["matrix"]),
            "",
            "## Promotion Gate",
            *_bullet_lines(plan["promotion_gate"]),
            "",
            "## Risks",
            *_bullet_lines(plan["risks"]),
            "",
            "## Review Checkpoint",
            f"- Checkpoint id: `{_inline(review['checkpoint_id'])}`",
            f"- Status: `{_inline(review['status'])}`",
            f"- Reason: {_inline(review['reason'])}",
        ]
    )
    if review["required_actions"]:
        lines.append("- Required actions:")
        lines.extend(f"  - {_inline(item)}" for item in review["required_actions"])

    lines.extend(
        [
            "",
            "## Eval Checks",
            f"- Overall: `{evals.passed}`",
        ]
    )
    for check in evals.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"- `{status}` `{_inline(check.name)}`: {_inline(check.detail)}")

    lines.extend(
        [
            "",
            "## Decision Audit",
            f"- Overall: `{audit.passed}`",
            f"- Decision: `{_inline(audit.decision)}`",
            f"- Score: `{audit.score:.3f}`",
            f"- Summary: {_inline(audit.summary)}",
        ]
    )
    for finding in audit.findings:
        lines.append(
            f"- `{finding.status.upper()}` `{_inline(finding.name)}` "
            f"[{_inline(finding.severity)}]: {_inline(finding.detail)}"
        )
    if audit.required_actions:
        lines.append("- Required audit actions:")
        lines.extend(f"  - {_inline(item)}" for item in audit.required_actions)

    if source_resolution is not None:
        lines.extend(
            [
                "",
                "## Source Resolution",
                f"- Overall: `{source_resolution.passed}`",
            ]
        )
        for result in source_resolution.results:
            http_status = "" if result.http_status is None else f" HTTP {result.http_status}"
            lines.append(
                f"- `{_inline(result.status.upper())}` {_inline(result.ref)}{http_status}: "
                f"{_inline(result.detail)}"
            )

    lines.extend(
        [
            "",
            "## Tool Trace",
            "| Step | Tool | Inputs | Output summary |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for index, event in enumerate(data["trace"], start=1):
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(str(index)),
                    _cell(event["tool"]),
                    _cell(_compact_json(event["inputs"])),
                    _cell(_compact_json(event["outputs"])),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_decision_packet(
    run: AgentRun,
    path: str | Path,
    audit_report: AuditReport | None = None,
    eval_result: EvalResult | None = None,
    source_resolution: SourceResolutionReport | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_decision_packet(
            run,
            audit_report=audit_report,
            eval_result=eval_result,
            source_resolution=source_resolution,
        ),
        encoding="utf-8",
    )


def format_decision_packet_html(
    run: AgentRun,
    audit_report: AuditReport | None = None,
    eval_result: EvalResult | None = None,
    source_resolution: SourceResolutionReport | None = None,
) -> str:
    """Render a self-contained static HTML review artifact for one agent run."""
    audit = audit_report or audit_run(run)
    evals = eval_result or evaluate_run(run)
    data = run_to_dict(run)
    plan = data["plan"]
    review = data["review"]
    recommendation = str(plan["recommendation"])
    review_status = str(review["status"])
    audit_decision = audit.decision
    evidence_rows = _evidence_rows_html(data["evidence"])
    claims = _claims_html(plan["cited_claims"])
    matrix = _list_html(plan["matrix"])
    gates = _list_html(plan["promotion_gate"])
    risks = _list_html(plan["risks"])
    actions = _list_html(review["required_actions"]) if review["required_actions"] else "<p class=\"muted\">No actions required.</p>"
    eval_rows = "\n".join(
        f"""
        <tr>
          <td><span class="badge {_status_class(check.passed)}">{_e('PASS' if check.passed else 'FAIL')}</span></td>
          <td><code>{_e(check.name)}</code></td>
          <td>{_e(check.detail)}</td>
        </tr>
        """.strip()
        for check in evals.checks
    )
    audit_rows = "\n".join(
        f"""
        <tr>
          <td><span class="badge {_status_class(finding.status == 'pass')}">{_e(finding.status.upper())}</span></td>
          <td><code>{_e(finding.name)}</code></td>
          <td>{_e(finding.severity)}</td>
          <td>{_e(finding.detail)}</td>
        </tr>
        """.strip()
        for finding in audit.findings
    )
    source_resolution_block = (
        _source_resolution_html(source_resolution)
        if source_resolution is not None
        else ""
    )
    trace_rows = _trace_rows_html(data["trace"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TracePilot Decision Review</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-alt: #eef3f7;
      --text: #1f2933;
      --muted: #5f6b7a;
      --line: #d8dee7;
      --blue: #1f6feb;
      --green: #137333;
      --red: #b42318;
      --amber: #8a5a00;
      --code: #f0f2f5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }}
    main, .header-inner {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 19px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 18px;
      margin: 18px 0;
    }}
    .question {{
      max-width: 900px;
      color: var(--muted);
      margin: 0;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      background: var(--surface-alt);
      min-height: 78px;
    }}
    .metric-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .metric-value {{
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fafbfc;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    li + li {{ margin-top: 8px; }}
    code {{
      background: var(--code);
      border-radius: 4px;
      padding: 1px 4px;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      margin: 8px 0 0;
      font-size: 12px;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      background: #fbfcfd;
    }}
    details + details {{ margin-top: 10px; }}
    summary {{
      cursor: pointer;
      font-weight: 650;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid currentColor;
      white-space: nowrap;
    }}
    .pass {{ color: var(--green); background: #eaf5ed; }}
    .fail {{ color: var(--red); background: #faeded; }}
    .info {{ color: var(--blue); background: #edf4ff; }}
    .warn {{ color: var(--amber); background: #fff6df; }}
    .muted {{ color: var(--muted); }}
    .section-note {{
      margin: -6px 0 14px;
      color: var(--muted);
    }}
    @media (max-width: 820px) {{
      main, .header-inner {{ padding: 16px; }}
      .layout {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 23px; }}
      table {{ table-layout: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>TracePilot Decision Review</h1>
      <p class="question">{_e(data['question']['text'])}</p>
      <div class="summary-grid">
        <div class="metric">
          <span class="metric-label">Recommendation</span>
          <span class="metric-value">{_e(recommendation)}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Review Status</span>
          <span class="metric-value">{_status_badge_html(review_status)}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Audit Decision</span>
          <span class="metric-value">{_e(audit_decision)}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Eval / Audit</span>
          <span class="metric-value">{_status_badge_html('pass' if evals.passed else 'fail')} {_status_badge_html('pass' if audit.passed else 'fail')}</span>
        </div>
      </div>
    </div>
  </header>
  <main>
    <section>
      <h2>Evidence</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 16%;">Evidence</th>
            <th style="width: 10%;">Fit</th>
            <th style="width: 10%;">Relevance</th>
            <th>Source</th>
            <th>Support</th>
            <th>Caution</th>
          </tr>
        </thead>
        <tbody>
          {evidence_rows}
        </tbody>
      </table>
    </section>
    <div class="layout">
      <section>
        <h2>Cited Claims</h2>
        {claims}
      </section>
      <section>
        <h2>Review Checkpoint</h2>
        <p><strong>ID:</strong> <code>{_e(review['checkpoint_id'])}</code></p>
        <p><strong>Status:</strong> {_status_badge_html(review_status)}</p>
        <p><strong>Reason:</strong> {_e(review['reason'])}</p>
        <h3>Required Actions</h3>
        {actions}
      </section>
    </div>
    <div class="layout">
      <section>
        <h2>Experiment Matrix</h2>
        {matrix}
      </section>
      <section>
        <h2>Promotion Gate</h2>
        {gates}
      </section>
    </div>
    <section>
      <h2>Risks</h2>
      {risks}
    </section>
    <div class="layout">
      <section>
        <h2>Eval Checks</h2>
        <table>
          <thead><tr><th style="width: 90px;">Status</th><th style="width: 210px;">Check</th><th>Detail</th></tr></thead>
          <tbody>{eval_rows}</tbody>
        </table>
      </section>
      <section>
        <h2>Decision Audit</h2>
        <p class="section-note">Decision: <strong>{_e(audit.decision)}</strong>; score <strong>{audit.score:.3f}</strong>.</p>
        <p>{_e(audit.summary)}</p>
        <table>
          <thead><tr><th>Status</th><th>Finding</th><th>Severity</th><th>Detail</th></tr></thead>
          <tbody>{audit_rows}</tbody>
        </table>
      </section>
    </div>
    {source_resolution_block}
    <section>
      <h2>Tool Trace</h2>
      {trace_rows}
    </section>
  </main>
</body>
</html>
"""


def write_decision_packet_html(
    run: AgentRun,
    path: str | Path,
    audit_report: AuditReport | None = None,
    eval_result: EvalResult | None = None,
    source_resolution: SourceResolutionReport | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_decision_packet_html(
            run,
            audit_report=audit_report,
            eval_result=eval_result,
            source_resolution=source_resolution,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a decision packet from an AgentRun JSON trace.")
    parser.add_argument("trace_in", help="AgentRun JSON produced by --trace-out.")
    parser.add_argument("--out", help="Write Markdown packet to this path instead of stdout.")
    parser.add_argument(
        "--format",
        choices=("markdown", "html"),
        help="Output format. Defaults to html for .html outputs, otherwise markdown.",
    )
    parser.add_argument(
        "--validate-sources",
        action="store_true",
        help="Resolve source refs while rendering the packet.",
    )
    args = parser.parse_args()

    run = _run_from_dict(json.loads(Path(args.trace_in).read_text(encoding="utf-8")))
    source_resolution = validate_run_sources(run) if args.validate_sources else None
    output_format = args.format or ("html" if args.out and str(args.out).endswith(".html") else "markdown")
    text = (
        format_decision_packet_html(run, source_resolution=source_resolution)
        if output_format == "html"
        else format_decision_packet(run, source_resolution=source_resolution)
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _run_from_dict(data: dict[str, Any]) -> AgentRun:
    from .schemas import (
        CitedClaim,
        EvidenceItem,
        ExperimentContext,
        ExperimentPlan,
        PaperCandidate,
        ResearchQuestion,
        ReviewCheckpoint,
        TraceEvent,
    )

    plan_data = data["plan"]
    candidates = [
        PaperCandidate(
            paper_id=item["paper_id"],
            title=item["title"],
            year=item["year"],
            tags=tuple(item["tags"]),
            claim=item["claim"],
            risk=item["risk"],
            source_url=item.get("source_url", ""),
            doi=item.get("doi", ""),
            venue=item.get("venue", ""),
            citation_count=item.get("citation_count"),
        )
        for item in data["candidates"]
    ]
    context_data = data["context"]
    review_data = data["review"]
    return AgentRun(
        question=ResearchQuestion(**data["question"]),
        candidates=candidates,
        context=ExperimentContext(
            project_name=context_data["project_name"],
            trusted_split=context_data["trusted_split"],
            diagnostic_split=context_data["diagnostic_split"],
            baseline=context_data["baseline"],
            metrics=tuple(context_data["metrics"]),
            constraints=tuple(context_data["constraints"]),
        ),
        evidence=[EvidenceItem(**item) for item in data["evidence"]],
        plan=ExperimentPlan(
            summary=plan_data["summary"],
            matrix=tuple(plan_data["matrix"]),
            promotion_gate=tuple(plan_data["promotion_gate"]),
            risks=tuple(plan_data["risks"]),
            recommendation=plan_data["recommendation"],
            cited_claims=tuple(CitedClaim(**item) for item in plan_data.get("cited_claims", ())),
        ),
        review=ReviewCheckpoint(
            checkpoint_id=review_data["checkpoint_id"],
            status=review_data["status"],
            reason=review_data["reason"],
            required_actions=tuple(review_data["required_actions"]),
        ),
        trace=[TraceEvent(**item) for item in data["trace"]],
    )


def _trace_tools(trace: list[dict[str, object]]) -> str:
    tools = [str(event["tool"]) for event in trace]
    return ", ".join(tools) if tools else "(none)"


def _source_label(item: dict[str, object]) -> str:
    parts: list[str] = []
    if item.get("source_url"):
        parts.append(str(item["source_url"]))
    if item.get("doi"):
        parts.append(f"doi:{item['doi']}")
    if item.get("venue"):
        parts.append(f"venue:{item['venue']}")
    if item.get("citation_count") is not None:
        parts.append(f"citations:{item['citation_count']}")
    return "; ".join(parts) if parts else "-"


def _bullet_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- (none)"]
    return [f"- {_inline(item)}" for item in items]


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cell(value: object) -> str:
    text = _inline(str(value))
    return text.replace("|", "\\|")


def _inline(value: object) -> str:
    return str(value).replace("\n", "<br>")


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _status_class(passed: bool) -> str:
    return "pass" if passed else "fail"


def _status_badge_html(value: str) -> str:
    normalized = value.lower()
    if normalized in {"pass", "passed", "true", "requires_approval"}:
        css_class = "pass" if normalized != "requires_approval" else "warn"
    elif normalized in {"fail", "failed", "false"}:
        css_class = "fail"
    else:
        css_class = "info"
    return f"<span class=\"badge {css_class}\">{_e(value)}</span>"


def _evidence_rows_html(items: list[dict[str, object]]) -> str:
    if not items:
        return """
        <tr>
          <td>-</td><td>-</td><td>-</td><td>-</td>
          <td>No evidence retrieved.</td><td>Run abstains.</td>
        </tr>
        """.strip()
    rows = []
    for item in items:
        rows.append(
            f"""
            <tr>
              <td><code>{_e(item['evidence_id'])}</code><br><span class="muted">{_e(item['paper_id'])}</span></td>
              <td>{_status_badge_html(str(item['transfer_fit']))}</td>
              <td>{float(item['relevance']):.2f}</td>
              <td>{_e(_source_label(item))}</td>
              <td>{_e(item['supports'])}</td>
              <td>{_e(item['cautions'])}</td>
            </tr>
            """.strip()
        )
    return "\n".join(rows)


def _claims_html(claims: list[dict[str, object]]) -> str:
    if not claims:
        return "<p class=\"muted\">No cited claims. This is expected only for no-evidence abstention.</p>"
    lines = ["<ul>"]
    for claim in claims:
        evidence_ids = ", ".join(str(item) for item in claim["evidence_ids"])
        source_refs = "; ".join(str(item) for item in claim["source_refs"])
        lines.append(
            f"<li>{_e(claim['claim'])}<br>"
            f"<span class=\"muted\">Evidence: {_e(evidence_ids)}; Sources: {_e(source_refs)}</span></li>"
        )
    lines.append("</ul>")
    return "\n".join(lines)


def _list_html(items: list[str]) -> str:
    if not items:
        return "<p class=\"muted\">None.</p>"
    return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>"


def _source_resolution_html(report: SourceResolutionReport) -> str:
    rows = []
    for result in report.results:
        status = result.status.upper()
        http_status = "" if result.http_status is None else f" HTTP {result.http_status}"
        rows.append(
            f"""
            <tr>
              <td>{_status_badge_html(status)}</td>
              <td>{_e(result.ref)}</td>
              <td>{_e(result.source_type)}</td>
              <td>{_e(http_status)}</td>
              <td>{_e(result.detail)}</td>
            </tr>
            """.strip()
        )
    return f"""
    <section>
      <h2>Source Resolution</h2>
      <p class="section-note">Overall: {_status_badge_html('pass' if report.passed else 'fail')}</p>
      <table>
        <thead><tr><th>Status</th><th>Reference</th><th>Type</th><th>HTTP</th><th>Detail</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """.strip()


def _trace_rows_html(trace: list[dict[str, object]]) -> str:
    if not trace:
        return "<p class=\"muted\">No trace events.</p>"
    rows = []
    for index, event in enumerate(trace, start=1):
        rows.append(
            f"""
            <details>
              <summary>{index}. {_e(event['tool'])}</summary>
              <h3>Inputs</h3>
              <pre>{_e(_compact_json(event['inputs']))}</pre>
              <h3>Output Summary</h3>
              <pre>{_e(_compact_json(event['outputs']))}</pre>
            </details>
            """.strip()
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
