from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serialization import to_jsonable


DEFAULT_CHECKED_ON = "2026-06-22"
DEFAULT_MARKDOWN = "traces/reference-catalog-report.md"
DEFAULT_JSON = "traces/reference-catalog-report.json"


@dataclass(frozen=True)
class ReferenceCatalogEntry:
    source_id: str
    title: str
    url: str
    source_kind: str
    current_signal: str
    learn: str
    borrowed_pattern: str
    project_evidence: tuple[str, ...]
    selection_guidance: str
    design_note: str


@dataclass(frozen=True)
class ReferenceCatalogReport:
    checked_on: str
    entries: tuple[ReferenceCatalogEntry, ...]
    project_fit_summary: str

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def github_count(self) -> int:
        return sum(1 for entry in self.entries if entry.source_kind == "github")


REFERENCE_ENTRIES = (
    ReferenceCatalogEntry(
        source_id="openai_agents_sdk",
        title="openai/openai-agents-python",
        url="https://github.com/openai/openai-agents-python",
        source_kind="github",
        current_signal=(
            "Current README presents it as a lightweight multi-agent workflow SDK with tools, "
            "guardrails, human-in-the-loop, sessions, tracing, MCP, and sandbox agents."
        ),
        learn="Tool contracts, guardrails, handoffs, sessions, tracing, and human approval as first-class objects.",
        borrowed_pattern="Controller/tools/trace vocabulary without making the LLM the source of truth.",
        project_evidence=(
            "src/bioml_agent/controllers.py",
            "src/bioml_agent/tool_contracts.py",
            "src/bioml_agent/openai_controller.py",
            "src/bioml_agent/service_interface.py",
        ),
        selection_guidance="Use as runtime vocabulary and eventual SDK path; keep deterministic mode for offline eval.",
        design_note=(
            "I borrowed the explicit tools/guardrails/tracing model, but kept a local controller protocol so "
            "evals still run without an API key."
        ),
    ),
    ReferenceCatalogEntry(
        source_id="anthropic_effective_agents",
        title="Anthropic: Building effective agents",
        url="https://www.anthropic.com/engineering/building-effective-agents",
        source_kind="article",
        current_signal=(
            "The article distinguishes workflows from agents, recommends simple solutions first, and emphasizes "
            "well-documented agent-computer interfaces."
        ),
        learn="When to use workflows, when to add agent autonomy, and why tool design is part of agent design.",
        borrowed_pattern="Start with a deterministic workflow, then add model-driven control where eval evidence justifies it.",
        project_evidence=(
            "README.md",
            "src/bioml_agent/agent_loop.py",
        ),
        selection_guidance="Use as the main architecture explanation in design notes.",
        design_note=(
            "The project starts as a workflow on purpose: it locks down source provenance, state, and eval before "
            "introducing model autonomy."
        ),
    ),
    ReferenceCatalogEntry(
        source_id="langgraph",
        title="langchain-ai/langgraph",
        url="https://github.com/langchain-ai/langgraph",
        source_kind="github",
        current_signal=(
            "GitHub README points to durable execution that resumes after failures and human-in-the-loop state "
            "inspection or modification."
        ),
        learn="State graph semantics, checkpointing, long-running execution, and human review of mutable state.",
        borrowed_pattern="Typed state graph with stop-after, checkpoint-on-error, and resume paths.",
        project_evidence=(
            "src/bioml_agent/state_graph.py",
            "tests/test_state_graph.py",
            "traces/state-failure-resume.json",
        ),
        selection_guidance="Adopt when persistence backend and concurrent long-running runs become real product needs.",
        design_note="I implemented the minimal state graph first, so I can explain exactly what LangGraph would replace.",
    ),
    ReferenceCatalogEntry(
        source_id="mcp",
        title="Model Context Protocol",
        url="https://github.com/modelcontextprotocol",
        source_kind="github",
        current_signal=(
            "GitHub org frames MCP as a protocol for integrating LLM applications with external data sources, "
            "with specification docs and official SDKs."
        ),
        learn="Tool/data/workflow protocol boundaries, server/client separation, discovery, and permission surfaces.",
        borrowed_pattern="Tool manifest and approval policy designed so local tools can later become MCP tools.",
        project_evidence=(
            "src/bioml_agent/tool_contracts.py",
            "src/bioml_agent/service_mcp.py",
            "tests/test_service_mcp.py",
        ),
        selection_guidance="Use for Phase 6, after local tools and approval semantics are stable.",
        design_note=(
            "I did not start by shipping an MCP server; I first made tool contracts explicit enough that an MCP "
            "adapter has a clean boundary."
        ),
    ),
    ReferenceCatalogEntry(
        source_id="ragas",
        title="vibrantlabsai/ragas",
        url="https://docs.ragas.io/en/stable/",
        source_kind="docs",
        current_signal=(
            "Docs position Ragas as a move from vibe checks to systematic evaluation loops and list agent/tool-use "
            "metrics such as tool call accuracy and agent goal accuracy."
        ),
        learn="Evaluation datasets, metrics, experiment loops, and separating retrieval, grounding, and tool-use quality.",
        borrowed_pattern="Case corpora, guardrail checks, audit rubric, and naive baseline comparison.",
        project_evidence=(
            "src/bioml_agent/evals.py",
            "src/bioml_agent/benchmark.py",
            "src/bioml_agent/eval_report.py",
            "traces/eval-coverage-report.md",
        ),
        selection_guidance="Use the evaluation mindset now; keep heavyweight evaluator integrations optional.",
        design_note="The project is not judged by one demo prompt; it has fixed corpora, guardrail checks, and baseline delta.",
    ),
    ReferenceCatalogEntry(
        source_id="crewai",
        title="crewAIInc/crewAI",
        url="https://github.com/crewAIInc/crewAI",
        source_kind="github",
        current_signal=(
            "README distinguishes autonomous Crews from production-ready event-driven Flows with fine-grained control "
            "and state management."
        ),
        learn="How to decide between multi-agent role collaboration and controlled workflow execution.",
        borrowed_pattern="Avoid decorative multi-agent design; use controlled flow unless role decomposition earns its complexity.",
        project_evidence=(
            "README.md",
            "src/bioml_agent/demo.py",
        ),
        selection_guidance="Study the crews/flows distinction; do not use multi-agent roles as a portfolio gimmick.",
        design_note="I can explain why this project is not multi-agent yet: the current bottleneck is eval and provenance.",
    ),
    ReferenceCatalogEntry(
        source_id="openhands",
        title="OpenHands/OpenHands",
        url="https://github.com/OpenHands/OpenHands",
        source_kind="github",
        current_signal="GitHub repository describes OpenHands as AI-driven development with CLI and developer-tooling scope.",
        learn="Workspace-agent product surfaces: files, commands, long tasks, local runtime, and user-facing UI.",
        borrowed_pattern="Generated artifact index and local dashboard instead of a single notebook or chat transcript.",
        project_evidence=(
            "src/bioml_agent/portfolio_index.py",
            "traces/index.html",
        ),
        selection_guidance="Borrow product-shape lessons when adding experiment execution or UI.",
        design_note="A serious agent project needs an inspectable product surface, not only an implementation path.",
    ),
    ReferenceCatalogEntry(
        source_id="smolagents",
        title="huggingface/smolagents",
        url="https://github.com/huggingface/smolagents",
        source_kind="github",
        current_signal=(
            "README emphasizes agents that think in code, minimal abstractions, and sandboxed execution options."
        ),
        learn="Why code actions need sandboxing and why small abstractions are easier to inspect.",
        borrowed_pattern="Keep the local loop small, typed, and auditable before adding remote execution.",
        project_evidence=(
            "src/bioml_agent/agent_loop.py",
            "src/bioml_agent/tools.py",
            "src/bioml_agent/source_safety.py",
        ),
        selection_guidance="Use as a warning that code/experiment execution must come with sandbox and permission scope.",
        design_note="The project deliberately plans experiment execution later, after source and permission gates exist.",
    ),
    ReferenceCatalogEntry(
        source_id="autogen",
        title="microsoft/autogen",
        url="https://github.com/microsoft/autogen",
        source_kind="github",
        current_signal=(
            "README now directs new users to Microsoft Agent Framework and describes AutoGen as maintenance mode."
        ),
        learn="Multi-agent API history, migration risk, and why framework status must be checked before choosing a stack.",
        borrowed_pattern="Use AutoGen as historical reference, not as the primary dependency for a new portfolio project.",
        project_evidence=(
            "README.md",
            "traces/reference-catalog-report.md",
        ),
        selection_guidance="Do not build new core architecture on a maintenance-mode framework.",
        design_note="I checked current project status before choosing what to borrow; not every popular repo should be adopted.",
    ),
)


def build_reference_catalog_report(checked_on: str = DEFAULT_CHECKED_ON) -> ReferenceCatalogReport:
    return ReferenceCatalogReport(
        checked_on=checked_on,
        entries=REFERENCE_ENTRIES,
        project_fit_summary=(
            "The strongest project story is not framework adoption. It is a source-grounded research workflow "
            "with explicit tools, state, eval, audit, approval, and reproducible artifacts. Frameworks are used as "
            "design references and future integration paths."
        ),
    )


def reference_catalog_to_dict(report: ReferenceCatalogReport) -> dict[str, Any]:
    data = to_jsonable(report)
    data["total"] = report.total
    data["github_count"] = report.github_count
    return data


def write_reference_catalog_report(
    report: ReferenceCatalogReport,
    path: str | Path,
    json_output: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(reference_catalog_to_dict(report), indent=2, ensure_ascii=False)
        if json_output
        else format_reference_catalog_report(report)
    )
    output_path.write_text(text, encoding="utf-8")


def format_reference_catalog_report(report: ReferenceCatalogReport) -> str:
    lines = [
        "# Agent Reference Catalog",
        "",
        "## Summary",
        f"- Checked on: `{report.checked_on}`",
        f"- Sources: `{report.total}`",
        f"- GitHub-backed sources: `{report.github_count}`",
        f"- Project fit: {report.project_fit_summary}",
        "",
        "## Source Matrix",
        "| Source | Current signal | Learn | Borrowed pattern | Project evidence | Selection guidance |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report.entries:
        lines.append(
            f"| [{_cell(entry.title)}]({entry.url}) | {_cell(entry.current_signal)} | {_cell(entry.learn)} | "
            f"{_cell(entry.borrowed_pattern)} | {_join(entry.project_evidence)} | {_cell(entry.selection_guidance)} |"
        )

    lines.extend(["", "## Design Notes"])
    for entry in report.entries:
        lines.append(f"- `{entry.source_id}`: {entry.design_note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the source-backed agent reference catalog.")
    parser.add_argument("--checked-on", default=DEFAULT_CHECKED_ON, help="Date attached to the generated report.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--out", help="Write report to this path.")
    args = parser.parse_args()

    report = build_reference_catalog_report(checked_on=args.checked_on)
    text = (
        json.dumps(reference_catalog_to_dict(report), indent=2, ensure_ascii=False)
        if args.json
        else format_reference_catalog_report(report)
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _join(items: tuple[str, ...]) -> str:
    return ", ".join(f"`{_cell(item)}`" for item in items)


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
