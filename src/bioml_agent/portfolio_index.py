from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
from typing import Any


DEFAULT_TRACE_DIR = "traces"
DEFAULT_INDEX = "traces/index.html"


@dataclass(frozen=True)
class PortfolioArtifact:
    title: str
    path: str
    description: str
    kind: str
    required: bool = True


DEFAULT_ARTIFACTS = (
    PortfolioArtifact(
        title="Quality Gate Report",
        path="quality-gate-report.md",
        description="Local CI-style gate covering compilation, tests, contracts, eval coverage, decision artifacts, and demo generation.",
        kind="gate",
    ),
    PortfolioArtifact(
        title="Reference Catalog Report",
        path="reference-catalog-report.md",
        description="GitHub and web source matrix explaining which design patterns I borrowed and which ones I did not adopt.",
        kind="guide",
    ),
    PortfolioArtifact(
        title="Service Interface Report",
        path="service-interface-report.md",
        description="MCP-style descriptors and HTTP route specs derived from the canonical tool contracts for service-boundary discussion.",
        kind="guide",
    ),
    PortfolioArtifact(
        title="Service Runtime Demo",
        path="service-runtime-demo.md",
        description="Session-backed service route execution proving contract-derived routes can run the full workflow.",
        kind="demo",
    ),
    PortfolioArtifact(
        title="Service API Readiness",
        path="service-api-readiness.md",
        description="Optional FastAPI adapter endpoint plan, OpenAPI preview, dependency status, and launch command.",
        kind="guide",
    ),
    PortfolioArtifact(
        title="Service MCP Readiness",
        path="service-mcp-readiness.md",
        description="Optional FastMCP adapter tool bindings, resources, prompts, transports, dependency status, and launch commands.",
        kind="guide",
    ),
    PortfolioArtifact(
        title="Eval Coverage Report",
        path="eval-coverage-report.md",
        description="Multi-corpus coverage summary with case families, guardrail checks, and naive-baseline delta.",
        kind="eval",
    ),
    PortfolioArtifact(
        title="Promotion Decision Review",
        path="promotion-decision-review.html",
        description="Static HTML review for the promotion-path run, including evidence, gate, audit, and tool trace.",
        kind="review",
    ),
    PortfolioArtifact(
        title="Review Workbench",
        path="review-workbench.html",
        description="Product-style workbench for approval actions, evidence coverage, claim grounding, evals, audit, and trace.",
        kind="review",
    ),
    PortfolioArtifact(
        title="Promotion Decision Packet",
        path="promotion-decision-packet.md",
        description="Markdown human-review packet for the same promotion-path run.",
        kind="review",
    ),
    PortfolioArtifact(
        title="Promotion Trace JSON",
        path="promotion-run.json",
        description="Full AgentRun trace behind the review artifacts.",
        kind="trace",
    ),
    PortfolioArtifact(
        title="Portfolio Demo JSON",
        path="portfolio-demo.json",
        description="One-command portfolio demo covering promotion, abstention, benchmark, baseline comparison, and failure recovery paths.",
        kind="demo",
    ),
    PortfolioArtifact(
        title="Eval Coverage JSON",
        path="eval-coverage-report.json",
        description="Machine-readable version of the eval coverage report.",
        kind="eval",
    ),
)


def format_portfolio_index(trace_dir: str | Path = DEFAULT_TRACE_DIR) -> str:
    root = Path(trace_dir)
    quality = _load_json(root / "quality-gate-report.json")
    eval_report = _load_json(root / "eval-coverage-report.json")
    demo = _load_json(root / "portfolio-demo.json")
    cards = "\n".join(_artifact_card(root, artifact) for artifact in DEFAULT_ARTIFACTS)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TracePilot Dashboard</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%230f172a'/%3E%3Cpath d='M15 37h12l7-18 5 29 5-11h5' stroke='%2338bdf8' stroke-width='5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
  <style>
    :root {{
      --bg: #f5f7fa;
      --surface: #ffffff;
      --line: #d7dee8;
      --text: #1f2933;
      --muted: #617082;
      --blue: #1f6feb;
      --green: #137333;
      --amber: #8a5a00;
      --red: #b42318;
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
      max-width: 1180px;
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
      margin: 0 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; }}
    .lead {{
      color: var(--muted);
      max-width: 900px;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 18px;
      margin: 18px 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #f9fbfd;
      min-height: 82px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .metric strong {{
      display: block;
      font-size: 18px;
      overflow-wrap: anywhere;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      background: #fbfcfd;
      min-height: 170px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .card p {{
      color: var(--muted);
      margin: 8px 0 12px;
    }}
    a {{
      color: var(--blue);
      text-decoration: none;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    a:hover {{ text-decoration: underline; }}
    .badge {{
      display: inline-block;
      border: 1px solid currentColor;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      width: fit-content;
    }}
    .pass {{ color: var(--green); background: #eaf5ed; }}
    .warn {{ color: var(--amber); background: #fff6df; }}
    .fail {{ color: var(--red); background: #faeded; }}
    .info {{ color: var(--blue); background: #edf4ff; }}
    code {{
      background: var(--code);
      border-radius: 4px;
      padding: 1px 4px;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    li + li {{ margin-top: 8px; }}
    @media (max-width: 760px) {{
      main, .header-inner {{ padding: 16px; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>TracePilot Dashboard</h1>
      <p class="lead">I use this dashboard to inspect the generated outputs for a source-grounded planning agent: tool traces, explicit state, evaluation coverage, approval gates, service-boundary artifacts, and a gene-expression planning case study.</p>
      <div class="metrics">
        {_metric("Quality Gate", _quality_summary(quality))}
        {_metric("Eval Coverage", _eval_summary(eval_report))}
        {_metric("Baseline Delta", _comparison_summary(eval_report))}
        {_metric("Portfolio Demo", _demo_summary(demo))}
      </div>
    </div>
  </header>
  <main>
    <section>
      <h2>Project Artifacts</h2>
      <div class="grid">
        {cards}
      </div>
    </section>
    <section>
      <h2>Suggested Project Path</h2>
      <ul>
        <li>I start with <code>quality-gate-report.md</code> to confirm the generated artifacts match the current source.</li>
        <li>I open <code>promotion-decision-review.html</code> to inspect evidence, claims, split gates, audit findings, and trace.</li>
        <li>I use <code>eval-coverage-report.md</code> to verify coverage across benchmark corpora and guardrails.</li>
        <li>I use <code>reference-catalog-report.md</code> to inspect which GitHub projects and external sources informed the design.</li>
        <li>I use <code>service-interface-report.md</code>, <code>service-runtime-demo.md</code>, <code>service-api-readiness.md</code>, and <code>service-mcp-readiness.md</code> to inspect the service boundary.</li>
        <li>I open <code>review-workbench.html</code> to inspect approval actions, evidence coverage, claim grounding, audit, evals, and trace.</li>
        <li>I use <code>portfolio-demo.json</code> to inspect promotion, abstention, benchmark, baseline comparison, and failure recovery paths.</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


def write_portfolio_index(trace_dir: str | Path = DEFAULT_TRACE_DIR, out: str | Path = DEFAULT_INDEX) -> None:
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_portfolio_index(trace_dir=trace_dir), encoding="utf-8")


def serve_directory(directory: str | Path = ".", host: str = "127.0.0.1", port: int = 8765) -> None:
    root = Path(directory).resolve()
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(root), **kwargs)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving {root} at http://{host}:{port}/")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or serve the TracePilot portfolio index.")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Build the static portfolio index.")
    build.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR, help="Directory containing generated artifacts.")
    build.add_argument("--out", default=DEFAULT_INDEX, help="Output HTML path.")

    serve = subparsers.add_parser("serve", help="Serve the repository root over localhost.")
    serve.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR, help="Directory to serve.")
    serve.add_argument("--root", default=".", help="Repository root to serve.")
    serve.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    serve.add_argument("--port", type=int, default=8765, help="Port to bind.")

    args = parser.parse_args()
    if args.command == "serve":
        write_portfolio_index(trace_dir=args.trace_dir, out=Path(args.trace_dir) / "index.html")
        serve_directory(directory=args.root, host=args.host, port=args.port)
        return
    write_portfolio_index(
        trace_dir=getattr(args, "trace_dir", DEFAULT_TRACE_DIR),
        out=getattr(args, "out", DEFAULT_INDEX),
    )


def _artifact_card(root: Path, artifact: PortfolioArtifact) -> str:
    exists = (root / artifact.path).is_file()
    status = "ready" if exists else "missing"
    badge_class = "pass" if exists else ("fail" if artifact.required else "warn")
    link = (
        f"<a href=\"{_e(artifact.path)}\">Open artifact</a>"
        if exists
        else "<span class=\"badge fail\">Missing</span>"
    )
    return f"""
    <article class="card">
      <div>
        <span class="badge {badge_class}">{_e(status)}</span>
        <h3>{_e(artifact.title)}</h3>
        <p>{_e(artifact.description)}</p>
      </div>
      <div>
        <p><code>{_e(artifact.path)}</code></p>
        {link}
      </div>
    </article>
    """.strip()


def _metric(label: str, value: str) -> str:
    return f"""
    <div class="metric">
      <span>{_e(label)}</span>
      <strong>{value}</strong>
    </div>
    """.strip()


def _quality_summary(data: dict[str, Any]) -> str:
    if not data:
        return "<span class=\"badge warn\">not generated</span>"
    passed = bool(data.get("passed"))
    return f"<span class=\"badge {'pass' if passed else 'fail'}\">{_e(str(data.get('passed_count', 0)))}/{_e(str(data.get('total', 0)))}</span>"


def _eval_summary(data: dict[str, Any]) -> str:
    if not data:
        return "<span class=\"badge warn\">not generated</span>"
    return f"{_e(str(data.get('passed_cases', 0)))}/{_e(str(data.get('total_cases', 0)))} cases"


def _comparison_summary(data: dict[str, Any]) -> str:
    comparison = data.get("comparison") if data else None
    if not isinstance(comparison, dict):
        return "<span class=\"badge warn\">not generated</span>"
    agent = f"{comparison.get('agent_passed_count', 0)}/{comparison.get('total', 0)}"
    baseline = f"{comparison.get('baseline_passed_count', 0)}/{comparison.get('total', 0)}"
    return f"agent {_e(agent)} vs baseline {_e(baseline)}"


def _demo_summary(data: dict[str, Any]) -> str:
    if not data:
        return "<span class=\"badge warn\">not generated</span>"
    passed = bool(data.get("passed"))
    return f"<span class=\"badge {'pass' if passed else 'fail'}\">{_e(str(passed))}</span>"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _e(value: object) -> str:
    return escape(str(value), quote=True)


if __name__ == "__main__":
    main()
