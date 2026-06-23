from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import (
    AgentRun,
    CitedClaim,
    EvidenceItem,
    ExperimentContext,
    ExperimentPlan,
    PaperCandidate,
    ResearchQuestion,
    ReviewCheckpoint,
    TraceEvent,
)
from .serialization import to_jsonable
from .tools import ToolRegistry

GRAPH_NODES = (
    "search_literature",
    "read_experiment_context",
    "score_transfer_fit",
    "draft_experiment_plan",
    "create_review_checkpoint",
)
DONE_NODE = "done"


@dataclass
class GraphFailure:
    node: str
    error_type: str
    message: str
    recoverable: bool = True


@dataclass
class AgentGraphState:
    question: ResearchQuestion
    project_name: str = "EpiCETRA"
    next_node: str = GRAPH_NODES[0]
    candidates: list[PaperCandidate] = field(default_factory=list)
    context: ExperimentContext | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    plan: ExperimentPlan | None = None
    review: ReviewCheckpoint | None = None
    failure: GraphFailure | None = None
    trace: list[TraceEvent] = field(default_factory=list)


def initial_state(
    question: str | ResearchQuestion,
    project_name: str = "EpiCETRA",
) -> AgentGraphState:
    parsed_question = question if isinstance(question, ResearchQuestion) else ResearchQuestion(text=question)
    return AgentGraphState(question=parsed_question, project_name=project_name)


def run_state_graph(
    registry: ToolRegistry,
    state: AgentGraphState,
    stop_after: str | None = None,
    capture_errors: bool = False,
) -> AgentGraphState:
    if stop_after is not None and stop_after not in GRAPH_NODES:
        raise ValueError(f"unknown stop_after node: {stop_after}")
    _validate_next_node(state.next_node)
    if state.next_node != DONE_NODE:
        state.failure = None

    while state.next_node != DONE_NODE:
        node = state.next_node
        try:
            _run_node(registry, state, node)
        except Exception as exc:
            state.failure = GraphFailure(
                node=node,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            if capture_errors:
                return state
            raise
        state.next_node = _next_node_after(node)
        if stop_after == node:
            return state
    return state


def state_to_run(state: AgentGraphState) -> AgentRun:
    if state.failure is not None:
        raise RuntimeError(f"graph state is failed at {state.failure.node!r}: {state.failure.message}")
    if state.next_node != DONE_NODE:
        raise RuntimeError(f"graph state is incomplete; next node is {state.next_node!r}")
    if state.context is None:
        raise RuntimeError("graph state is missing experiment context")
    if state.plan is None:
        raise RuntimeError("graph state is missing experiment plan")
    if state.review is None:
        raise RuntimeError("graph state is missing review checkpoint")
    return AgentRun(
        question=state.question,
        candidates=state.candidates,
        context=state.context,
        evidence=state.evidence,
        plan=state.plan,
        review=state.review,
        trace=state.trace,
    )


def state_to_dict(state: AgentGraphState) -> dict[str, Any]:
    return to_jsonable(state)


def write_graph_checkpoint(state: AgentGraphState, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(state_to_dict(state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_graph_checkpoint(path: str | Path) -> AgentGraphState:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _state_from_dict(raw)


def _run_node(registry: ToolRegistry, state: AgentGraphState, node: str) -> None:
    if node == "search_literature":
        state.candidates = registry.call(
            "search_literature",
            state.trace,
            query=state.question.text,
            top_k=4,
        )
        return
    if node == "read_experiment_context":
        state.context = registry.call(
            "read_experiment_context",
            state.trace,
            project_name=state.project_name,
        )
        return
    if node == "score_transfer_fit":
        context = _require(state.context, "experiment context", node)
        state.evidence = registry.call(
            "score_transfer_fit",
            state.trace,
            candidates=state.candidates,
            context=context,
        )
        return
    if node == "draft_experiment_plan":
        context = _require(state.context, "experiment context", node)
        state.plan = registry.call(
            "draft_experiment_plan",
            state.trace,
            evidence=state.evidence,
            context=context,
        )
        return
    if node == "create_review_checkpoint":
        context = _require(state.context, "experiment context", node)
        plan = _require(state.plan, "experiment plan", node)
        state.review = registry.call(
            "create_review_checkpoint",
            state.trace,
            plan=plan,
            evidence=state.evidence,
            context=context,
        )
        return
    raise ValueError(f"unknown graph node: {node}")


def _require(value: Any | None, name: str, node: str) -> Any:
    if value is None:
        raise RuntimeError(f"cannot run {node}: missing {name}")
    return value


def _next_node_after(node: str) -> str:
    index = GRAPH_NODES.index(node)
    if index == len(GRAPH_NODES) - 1:
        return DONE_NODE
    return GRAPH_NODES[index + 1]


def _validate_next_node(node: str) -> None:
    if node not in (*GRAPH_NODES, DONE_NODE):
        raise ValueError(f"unknown next_node: {node}")


def _state_from_dict(raw: dict[str, Any]) -> AgentGraphState:
    next_node = raw.get("next_node", GRAPH_NODES[0])
    _validate_next_node(next_node)
    return AgentGraphState(
        question=_question_from_dict(raw["question"]),
        project_name=raw.get("project_name", "EpiCETRA"),
        next_node=next_node,
        candidates=[_paper_from_dict(item) for item in raw.get("candidates", [])],
        context=_context_from_dict(raw["context"]) if raw.get("context") is not None else None,
        evidence=[_evidence_from_dict(item) for item in raw.get("evidence", [])],
        plan=_plan_from_dict(raw["plan"]) if raw.get("plan") is not None else None,
        review=_review_from_dict(raw["review"]) if raw.get("review") is not None else None,
        failure=_failure_from_dict(raw["failure"]) if raw.get("failure") is not None else None,
        trace=[_trace_from_dict(item) for item in raw.get("trace", [])],
    )


def _question_from_dict(raw: dict[str, Any]) -> ResearchQuestion:
    return ResearchQuestion(
        text=raw["text"],
        domain=raw.get("domain", "promoter_expression_prediction"),
    )


def _paper_from_dict(raw: dict[str, Any]) -> PaperCandidate:
    return PaperCandidate(
        paper_id=raw["paper_id"],
        title=raw["title"],
        year=int(raw["year"]),
        tags=tuple(raw.get("tags", ())),
        claim=raw["claim"],
        risk=raw["risk"],
        source_url=raw.get("source_url", ""),
        doi=raw.get("doi", ""),
        venue=raw.get("venue", ""),
        citation_count=raw.get("citation_count"),
    )


def _context_from_dict(raw: dict[str, Any]) -> ExperimentContext:
    return ExperimentContext(
        project_name=raw["project_name"],
        trusted_split=raw["trusted_split"],
        diagnostic_split=raw["diagnostic_split"],
        baseline=raw["baseline"],
        metrics=tuple(raw.get("metrics", ())),
        constraints=tuple(raw.get("constraints", ())),
    )


def _evidence_from_dict(raw: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=raw["evidence_id"],
        paper_id=raw["paper_id"],
        title=raw["title"],
        relevance=float(raw["relevance"]),
        transfer_fit=raw["transfer_fit"],
        supports=raw["supports"],
        cautions=raw["cautions"],
        source_url=raw.get("source_url", ""),
        doi=raw.get("doi", ""),
        venue=raw.get("venue", ""),
        citation_count=raw.get("citation_count"),
    )


def _claim_from_dict(raw: dict[str, Any]) -> CitedClaim:
    return CitedClaim(
        claim=raw["claim"],
        evidence_ids=tuple(raw.get("evidence_ids", ())),
        source_refs=tuple(raw.get("source_refs", ())),
    )


def _plan_from_dict(raw: dict[str, Any]) -> ExperimentPlan:
    return ExperimentPlan(
        summary=raw["summary"],
        matrix=tuple(raw.get("matrix", ())),
        promotion_gate=tuple(raw.get("promotion_gate", ())),
        risks=tuple(raw.get("risks", ())),
        recommendation=raw["recommendation"],
        cited_claims=tuple(_claim_from_dict(item) for item in raw.get("cited_claims", ())),
    )


def _review_from_dict(raw: dict[str, Any]) -> ReviewCheckpoint:
    return ReviewCheckpoint(
        checkpoint_id=raw["checkpoint_id"],
        status=raw["status"],
        reason=raw["reason"],
        required_actions=tuple(raw.get("required_actions", ())),
    )


def _failure_from_dict(raw: dict[str, Any]) -> GraphFailure:
    return GraphFailure(
        node=raw["node"],
        error_type=raw["error_type"],
        message=raw["message"],
        recoverable=bool(raw.get("recoverable", True)),
    )


def _trace_from_dict(raw: dict[str, Any]) -> TraceEvent:
    return TraceEvent(
        step=raw["step"],
        tool=raw["tool"],
        inputs=raw.get("inputs", {}),
        outputs=raw.get("outputs", {}),
    )
