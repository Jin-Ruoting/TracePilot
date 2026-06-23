from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

from .audit import audit_run
from .evals import evaluate_run
from .reporting import _run_from_dict
from .schemas import AgentRun
from .serialization import to_jsonable, run_to_dict


DEFAULT_TRACE = "traces/promotion-run.json"
DEFAULT_HTML = "traces/review-workbench.html"
DEFAULT_JSON = "traces/review-workbench.json"


@dataclass(frozen=True)
class WorkbenchSummary:
    passed: bool
    question: str
    recommendation: str
    review_status: str
    audit_decision: str
    eval_passed: bool
    audit_passed: bool
    evidence_count: int
    cited_claim_count: int
    required_action_count: int
    trace_step_count: int
    high_fit_evidence_count: int


def load_run(path: str | Path) -> AgentRun:
    return _run_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_workbench_summary(run: AgentRun) -> WorkbenchSummary:
    audit = audit_run(run)
    evals = evaluate_run(run)
    data = run_to_dict(run)
    evidence = data["evidence"]
    cited_claims = data["plan"]["cited_claims"]
    review = data["review"]
    high_fit = [item for item in evidence if item["transfer_fit"] == "high"]
    passed = (
        evals.passed
        and audit.passed
        and bool(evidence)
        and bool(cited_claims)
        and review["status"] == "requires_approval"
        and bool(review["required_actions"])
    )
    return WorkbenchSummary(
        passed=passed,
        question=data["question"]["text"],
        recommendation=data["plan"]["recommendation"],
        review_status=review["status"],
        audit_decision=audit.decision,
        eval_passed=evals.passed,
        audit_passed=audit.passed,
        evidence_count=len(evidence),
        cited_claim_count=len(cited_claims),
        required_action_count=len(review["required_actions"]),
        trace_step_count=len(data["trace"]),
        high_fit_evidence_count=len(high_fit),
    )


def workbench_summary_to_dict(summary: WorkbenchSummary) -> dict[str, Any]:
    return to_jsonable(summary)


def format_workbench_html(run: AgentRun) -> str:
    audit = audit_run(run)
    evals = evaluate_run(run)
    data = run_to_dict(run)
    summary = build_workbench_summary(run)
    evidence_rows = "\n".join(_evidence_row(item) for item in data["evidence"]) or _empty_row(
        "No evidence returned.", 6
    )
    claim_rows = "\n".join(_claim_row(item) for item in data["plan"]["cited_claims"]) or _empty_row(
        "No cited claims returned.", 3
    )
    action_items = "\n".join(
        f"""
        <label class="check-row">
          <input type="checkbox">
          <span>{_e(action)}</span>
        </label>
        """.strip()
        for action in data["review"]["required_actions"]
    ) or '<p class="muted">No approval actions were required.</p>'
    gate_items = _list_html(data["plan"]["promotion_gate"])
    risk_items = _list_html(data["plan"]["risks"])
    eval_rows = "\n".join(
        f"""
        <tr>
          <td>{_badge('PASS' if check.passed else 'FAIL', check.passed)}</td>
          <td><code>{_e(check.name)}</code></td>
          <td>{_e(check.detail)}</td>
        </tr>
        """.strip()
        for check in evals.checks
    )
    audit_rows = "\n".join(
        f"""
        <tr>
          <td>{_badge(finding.status.upper(), finding.status == 'pass')}</td>
          <td><code>{_e(finding.name)}</code></td>
          <td>{_e(finding.severity)}</td>
          <td>{_e(finding.detail)}</td>
        </tr>
        """.strip()
        for finding in audit.findings
    )
    trace_rows = "\n".join(
        f"""
        <tr>
          <td>{index}</td>
          <td><code>{_e(event['tool'])}</code></td>
          <td>{_e(_compact_json(event['inputs']))}</td>
          <td>{_e(_compact_json(event['outputs']))}</td>
        </tr>
        """.strip()
        for index, event in enumerate(data["trace"], start=1)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TracePilot Review Workbench</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --surface: #ffffff;
      --line: #d7dee8;
      --text: #1f2933;
      --muted: #5d6978;
      --blue: #1f6feb;
      --green: #137333;
      --red: #b42318;
      --amber: #8a5a00;
      --code: #eef2f6;
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
      max-width: 1220px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 18px;
      margin: 18px 0;
    }}
    .lead {{
      color: var(--muted);
      max-width: 920px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric {{
      min-height: 78px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f9fbfd;
      padding: 11px 12px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .metric strong {{
      display: block;
      overflow-wrap: anywhere;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(300px, 0.95fr);
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
      background: #fbfcfd;
      font-size: 12px;
      font-weight: 700;
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
    .badge {{
      display: inline-block;
      border: 1px solid currentColor;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .pass {{ color: var(--green); background: #eaf5ed; }}
    .fail {{ color: var(--red); background: #faeded; }}
    .warn {{ color: var(--amber); background: #fff6df; }}
    .info {{ color: var(--blue); background: #edf4ff; }}
    .muted {{ color: var(--muted); }}
    .check-row {{
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      margin-top: 8px;
      background: #fbfcfd;
    }}
    .check-row input {{
      margin-top: 3px;
      width: 16px;
      height: 16px;
    }}
    @media (max-width: 820px) {{
      main, .header-inner {{ padding: 16px; }}
      h1 {{ font-size: 24px; }}
      .layout {{ grid-template-columns: 1fr; }}
      table {{ table-layout: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>TracePilot Review Workbench</h1>
      <p class="lead">{_e(summary.question)}</p>
      <div class="metric-grid">
        {_metric("Recommendation", _e(summary.recommendation))}
        {_metric("Review Status", _badge(summary.review_status, summary.review_status == "requires_approval", "warn"))}
        {_metric("Eval / Audit", f"{_badge('EVAL PASS' if summary.eval_passed else 'EVAL FAIL', summary.eval_passed)} {_badge('AUDIT PASS' if summary.audit_passed else 'AUDIT FAIL', summary.audit_passed)}")}
        {_metric("Evidence", f"{summary.evidence_count} items, {summary.high_fit_evidence_count} high fit")}
        {_metric("Claims", f"{summary.cited_claim_count} cited claims")}
        {_metric("Trace", f"{summary.trace_step_count} tool steps")}
      </div>
    </div>
  </header>
  <main>
    <div class="layout">
      <section>
        <h2>Approval Queue</h2>
        <p class="muted">{_e(data['review']['reason'])}</p>
        {action_items}
      </section>
      <section>
        <h2>Decision Summary</h2>
        <p><strong>Audit decision:</strong> {_e(summary.audit_decision)}</p>
        <p><strong>Plan summary:</strong> {_e(data['plan']['summary'])}</p>
      </section>
    </div>
    <section>
      <h2>Evidence Coverage</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 12%;">Evidence</th>
            <th style="width: 12%;">Fit</th>
            <th style="width: 10%;">Score</th>
            <th>Source</th>
            <th>Support</th>
            <th>Caution</th>
          </tr>
        </thead>
        <tbody>{evidence_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Claim Grounding</h2>
      <table>
        <thead><tr><th>Claim</th><th style="width: 22%;">Evidence IDs</th><th style="width: 28%;">Source refs</th></tr></thead>
        <tbody>{claim_rows}</tbody>
      </table>
    </section>
    <div class="layout">
      <section>
        <h2>Promotion Gate</h2>
        {gate_items}
      </section>
      <section>
        <h2>Risk Register</h2>
        {risk_items}
      </section>
    </div>
    <div class="layout">
      <section>
        <h2>Eval Checks</h2>
        <table>
          <thead><tr><th style="width: 92px;">Status</th><th style="width: 210px;">Check</th><th>Detail</th></tr></thead>
          <tbody>{eval_rows}</tbody>
        </table>
      </section>
      <section>
        <h2>Decision Audit</h2>
        <table>
          <thead><tr><th>Status</th><th>Finding</th><th>Severity</th><th>Detail</th></tr></thead>
          <tbody>{audit_rows}</tbody>
        </table>
      </section>
    </div>
    <section>
      <h2>Tool Trace Sequence</h2>
      <table>
        <thead><tr><th style="width: 58px;">Step</th><th style="width: 190px;">Tool</th><th>Inputs</th><th>Output summary</th></tr></thead>
        <tbody>{trace_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def write_workbench(
    run: AgentRun,
    path: str | Path,
    json_output: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(workbench_summary_to_dict(build_workbench_summary(run)), indent=2, ensure_ascii=False)
        if json_output
        else format_workbench_html(run)
    )
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TracePilot decision review workbench.")
    parser.add_argument("trace_in", nargs="?", default=DEFAULT_TRACE, help="AgentRun JSON trace to render.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary instead of HTML.")
    parser.add_argument("--out", help="Write output to this path.")
    args = parser.parse_args()

    run = load_run(args.trace_in)
    output = args.out or (DEFAULT_JSON if args.json else DEFAULT_HTML)
    write_workbench(run, output, json_output=args.json)
    summary = build_workbench_summary(run)
    raise SystemExit(0 if summary.passed else 1)


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _badge(label: str, passed: bool, passed_class: str = "pass") -> str:
    badge_class = passed_class if passed else "fail"
    return f'<span class="badge {badge_class}">{_e(label)}</span>'


def _metric(label: str, value: object) -> str:
    return f"""
    <div class="metric">
      <span>{_e(label)}</span>
      <strong>{value}</strong>
    </div>
    """.strip()


def _evidence_row(item: dict[str, Any]) -> str:
    source = item.get("source_url") or item.get("doi") or item.get("venue") or item.get("paper_id", "")
    return f"""
    <tr>
      <td><code>{_e(item['evidence_id'])}</code><br>{_e(item['paper_id'])}</td>
      <td>{_e(item['transfer_fit'])}</td>
      <td>{_e(f"{item['relevance']:.2f}")}</td>
      <td>{_e(source)}</td>
      <td>{_e(item['supports'])}</td>
      <td>{_e(item['cautions'])}</td>
    </tr>
    """.strip()


def _claim_row(item: dict[str, Any]) -> str:
    return f"""
    <tr>
      <td>{_e(item['claim'])}</td>
      <td>{_e(', '.join(item['evidence_ids']))}</td>
      <td>{_e('; '.join(item['source_refs']))}</td>
    </tr>
    """.strip()


def _empty_row(message: str, columns: int) -> str:
    return f'<tr><td colspan="{columns}" class="muted">{_e(message)}</td></tr>'


def _list_html(items: list[str]) -> str:
    if not items:
        return '<p class="muted">None.</p>'
    return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>"


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
