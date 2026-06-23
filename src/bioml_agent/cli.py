from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent_loop import BioMLResearchAgent
from .audit import audit_run, audit_to_dict, format_audit_report, write_audit_report
from .context import build_context_provider
from .evals import EvalResult, evaluate_run
from .literature import build_literature_provider
from .openai_controller import OpenAIToolCallingController
from .reporting import write_decision_packet, write_decision_packet_html
from .serialization import run_to_dict, write_review_packet, write_run_json
from .source_validation import (
    format_source_resolution_report,
    validate_run_sources,
    write_source_resolution_report,
)
from .state_graph import (
    DONE_NODE,
    GRAPH_NODES,
    AgentGraphState,
    initial_state,
    load_graph_checkpoint,
    run_state_graph,
    state_to_dict,
    state_to_run,
    write_graph_checkpoint,
)
from .tools import build_default_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TracePilot.")
    parser.add_argument("question", nargs="?", help="Research question to analyze.")
    parser.add_argument("--project", help="Experiment context name. Also overrides checkpoint project on resume.")
    parser.add_argument(
        "--literature-source",
        choices=("fixture", "arxiv", "semantic_scholar", "pubmed", "frozen"),
        default="fixture",
        help="Literature search source. arxiv, semantic_scholar, and pubmed use live APIs.",
    )
    parser.add_argument(
        "--literature-snapshot",
        default="",
        help="JSONL snapshot path required when --literature-source frozen.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/literature",
        help="Cache directory for live literature providers.",
    )
    parser.add_argument(
        "--no-literature-cache",
        action="store_true",
        help="Disable cache for live literature providers.",
    )
    parser.add_argument(
        "--context-file",
        help="JSON experiment context file. Defaults to the built-in EpiCETRA fixture.",
    )
    parser.add_argument(
        "--experiment-log",
        help="JSON experiment log used to derive project baseline, metrics, split, and constraints.",
    )
    parser.add_argument(
        "--controller",
        choices=("offline", "openai"),
        default="offline",
        help="Controller runtime. offline is deterministic and does not require API keys.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL"),
        help="OpenAI model for --controller openai. Can also be set with OPENAI_MODEL.",
    )
    parser.add_argument("--trace-out", help="Write full AgentRun JSON trace to this path.")
    parser.add_argument("--review-out", help="Write a compact human-review checkpoint packet to this path.")
    parser.add_argument("--report-out", help="Write a Markdown decision packet for human review.")
    parser.add_argument("--report-html-out", help="Write a self-contained HTML decision review artifact.")
    parser.add_argument("--checkpoint-out", help="Write resumable graph checkpoint JSON to this path.")
    parser.add_argument("--resume-checkpoint", help="Resume an offline graph run from checkpoint JSON.")
    parser.add_argument(
        "--checkpoint-on-error",
        action="store_true",
        help="Capture graph node failures in checkpoint state instead of raising immediately.",
    )
    parser.add_argument(
        "--stop-after",
        choices=GRAPH_NODES,
        help="Stop an offline graph run after this node and optionally write --checkpoint-out.",
    )
    parser.add_argument("--validate-sources", action="store_true", help="Resolve real source URLs and DOI refs.")
    parser.add_argument("--source-validation-out", help="Write source resolution report JSON to this path.")
    parser.add_argument("--eval", action="store_true", help="Print lightweight eval checks after the report.")
    parser.add_argument("--audit", action="store_true", help="Print decision-quality audit after the report.")
    parser.add_argument("--audit-out", help="Write decision-quality audit JSON to this path.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of a compact report.")
    args = parser.parse_args()
    project_name = args.project or "EpiCETRA"

    uses_state_graph = bool(args.resume_checkpoint or args.stop_after or args.checkpoint_out or args.checkpoint_on_error)
    if not args.question and not args.resume_checkpoint:
        parser.error("question is required unless --resume-checkpoint is used")
    if args.context_file and args.experiment_log:
        parser.error("--context-file and --experiment-log are mutually exclusive")
    if uses_state_graph and args.controller != "offline":
        parser.error(
            "--checkpoint-out, --checkpoint-on-error, --stop-after, and --resume-checkpoint "
            "are supported only with --controller offline"
        )

    controller = None
    if args.controller == "openai":
        if not args.model:
            parser.error("--controller openai requires --model or OPENAI_MODEL")
        controller = OpenAIToolCallingController(model=args.model)

    literature_source = args.literature_source
    if literature_source == "frozen":
        if not args.literature_snapshot:
            parser.error("--literature-source frozen requires --literature-snapshot")
        literature_source = f"frozen:{args.literature_snapshot}"

    registry = build_default_registry(
        literature_provider=build_literature_provider(
            literature_source,
            cache_dir=args.cache_dir,
            use_cache=not args.no_literature_cache,
        ),
        context_provider=build_context_provider(
            context_file=args.context_file,
            experiment_log=args.experiment_log,
        ),
    )
    if uses_state_graph:
        state = (
            load_graph_checkpoint(args.resume_checkpoint)
            if args.resume_checkpoint
            else initial_state(args.question or "", project_name=project_name)
        )
        if args.resume_checkpoint and args.project:
            state.project_name = args.project
        state = run_state_graph(
            registry,
            state,
            stop_after=args.stop_after,
            capture_errors=args.checkpoint_on_error,
        )
        if args.checkpoint_out:
            write_graph_checkpoint(state, Path(args.checkpoint_out))
        if state.failure is not None:
            if args.json:
                print(json.dumps(state_to_dict(state), indent=2, ensure_ascii=False))
            else:
                print(_format_checkpoint_status(state, checkpoint_path=args.checkpoint_out))
            raise SystemExit(2)
        if state.next_node != DONE_NODE:
            if args.json:
                print(json.dumps(state_to_dict(state), indent=2, ensure_ascii=False))
            else:
                print(_format_checkpoint_status(state, checkpoint_path=args.checkpoint_out))
            return
        run = state_to_run(state)
    else:
        run = BioMLResearchAgent(registry=registry, controller=controller).run(
            args.question or "",
            project_name=project_name,
        )
    if args.trace_out:
        write_run_json(run, Path(args.trace_out))
    if args.review_out:
        write_review_packet(run, Path(args.review_out))
    audit_report = audit_run(run) if args.audit or args.audit_out or args.report_out or args.report_html_out else None
    if audit_report is not None and args.audit_out:
        write_audit_report(audit_report, Path(args.audit_out))
    eval_result = evaluate_run(run) if args.eval or args.report_out or args.report_html_out else None
    source_resolution = None
    if args.validate_sources or args.source_validation_out:
        source_resolution = validate_run_sources(run)
        if args.source_validation_out:
            write_source_resolution_report(source_resolution, Path(args.source_validation_out))
    if args.report_out:
        write_decision_packet(
            run,
            Path(args.report_out),
            audit_report=audit_report,
            eval_result=eval_result,
            source_resolution=source_resolution,
        )
    if args.report_html_out:
        write_decision_packet_html(
            run,
            Path(args.report_html_out),
            audit_report=audit_report,
            eval_result=eval_result,
            source_resolution=source_resolution,
        )
    if args.json:
        data: dict[str, object] = run_to_dict(run)
        if audit_report is not None:
            data = {"run": data, "audit": audit_to_dict(audit_report)}
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    print(_format_report(run))
    if args.eval:
        print()
        print(_format_eval_result(eval_result or evaluate_run(run)))
    if args.audit and audit_report is not None:
        print()
        print(format_audit_report(audit_report))
    if source_resolution is not None:
        print()
        print(format_source_resolution_report(source_resolution))


def _format_report(run: object) -> str:
    data = run_to_dict(run)  # type: ignore[arg-type]
    plan = data["plan"]
    review = data["review"]
    lines = [
        "# TracePilot Report",
        "",
        f"Question: {data['question']['text']}",
        f"Recommendation: {plan['recommendation']}",
        "",
        "## Summary",
        plan["summary"],
        "",
        "## Evidence",
    ]
    for item in data["evidence"]:
        lines.append(
            f"- {item['evidence_id']} | {item['transfer_fit']} fit: {item['title']} ({item['paper_id']})"
        )
        source_parts = _source_parts(item)
        if source_parts:
            lines.append(f"  Source: {'; '.join(source_parts)}")
        lines.append(f"  Supports: {item['supports']}")
        lines.append(f"  Cautions: {item['cautions']}")
    if plan["cited_claims"]:
        lines.extend(["", "## Cited Claims"])
        for item in plan["cited_claims"]:
            evidence_ids = ", ".join(item["evidence_ids"])
            source_refs = "; ".join(item["source_refs"])
            lines.append(f"- {item['claim']}")
            lines.append(f"  Evidence: {evidence_ids}")
            lines.append(f"  Sources: {source_refs}")
    lines.extend(["", "## Review Checkpoint"])
    lines.append(f"- ID: {review['checkpoint_id']}")
    lines.append(f"- Status: {review['status']}")
    lines.append(f"- Reason: {review['reason']}")
    if review["required_actions"]:
        lines.append("- Required actions:")
        lines.extend(f"  - {item}" for item in review["required_actions"])
    lines.extend(["", "## Experiment Matrix"])
    lines.extend(f"- {item}" for item in plan["matrix"])
    lines.extend(["", "## Promotion Gate"])
    lines.extend(f"- {item}" for item in plan["promotion_gate"])
    lines.extend(["", "## Trace"])
    lines.extend(f"- {event['tool']}: {event['outputs']}" for event in data["trace"])
    return "\n".join(lines)


def _format_checkpoint_status(state: AgentGraphState, checkpoint_path: str | None = None) -> str:
    lines = [
        "# TracePilot Checkpoint",
        "",
        f"Question: {state.question.text}",
        f"Next node: {state.next_node}",
        f"Completed tools: {_completed_tools(state)}",
        f"Candidates: {len(state.candidates)}",
        f"Evidence items: {len(state.evidence)}",
    ]
    if state.failure is not None:
        lines.extend(
            [
                f"Failure node: {state.failure.node}",
                f"Failure type: {state.failure.error_type}",
                f"Failure message: {state.failure.message}",
            ]
        )
    if checkpoint_path:
        lines.append(f"Checkpoint: {checkpoint_path}")
    return "\n".join(lines)


def _completed_tools(state: AgentGraphState) -> str:
    tools = [event.tool for event in state.trace]
    return ", ".join(tools) if tools else "(none)"


def _source_parts(item: dict[str, object]) -> list[str]:
    parts: list[str] = []
    for label, key in (
        ("url", "source_url"),
        ("doi", "doi"),
        ("venue", "venue"),
        ("citations", "citation_count"),
    ):
        value = item.get(key)
        if value is not None and value != "":
            parts.append(f"{label}={value}")
    return parts


def _format_eval_result(result: EvalResult) -> str:
    lines = [
        "## Eval Smoke",
        f"Passed: {result.passed}",
    ]
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"- {status} {check.name}: {check.detail}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
