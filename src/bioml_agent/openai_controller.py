from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schemas import (
    AgentRun,
    EvidenceItem,
    ExperimentContext,
    ExperimentPlan,
    PaperCandidate,
    ResearchQuestion,
    ReviewCheckpoint,
)
from .serialization import to_jsonable
from .tool_contracts import build_openai_tool_specs_from_contracts
from .tools import ToolRegistry


def build_openai_tool_specs() -> list[dict[str, Any]]:
    """Return strict Responses API function-tool schemas for the project tools."""
    return build_openai_tool_specs_from_contracts()


@dataclass
class _ToolState:
    candidates: list[PaperCandidate] | None = None
    context: ExperimentContext | None = None
    evidence: list[EvidenceItem] | None = None
    plan: ExperimentPlan | None = None
    review: ReviewCheckpoint | None = None


class OpenAIToolCallingController:
    """Optional Responses API controller.

    The controller keeps the project tools local and deterministic. The model
    decides which tool to call and with what arguments; tool execution remains
    inside this process so trace, tests, and guardrails stay project-owned.
    """

    def __init__(self, model: str, max_tool_rounds: int = 8) -> None:
        if not model:
            raise ValueError("model is required for OpenAIToolCallingController")
        self.model = model
        self.max_tool_rounds = max_tool_rounds

    def run(
        self,
        question: ResearchQuestion,
        registry: ToolRegistry,
        project_name: str = "EpiCETRA",
    ) -> AgentRun:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "OpenAI runtime requires the optional OpenAI SDK. "
                "Install this project with the llm extra or install openai."
            ) from exc

        client = OpenAI()
        state = _ToolState()
        trace = []
        instructions = (
            "You are a BioML research planning controller. Use tools only; do not invent "
            "paper evidence or experiment context. Call search_literature, "
            "read_experiment_context, score_transfer_fit, draft_experiment_plan, "
            "and create_review_checkpoint. Stop after create_review_checkpoint has produced a review gate."
        )
        response = client.responses.create(
            model=self.model,
            instructions=instructions,
            input=question.text,
            tools=build_openai_tool_specs(),
        )

        for _ in range(self.max_tool_rounds):
            calls = _extract_function_calls(response)
            if not calls:
                break
            tool_outputs = []
            for call in calls:
                output = self._execute_tool_call(
                    call=call,
                    question=question,
                    registry=registry,
                    project_name=project_name,
                    state=state,
                    trace=trace,
                )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(output, ensure_ascii=False),
                    }
                )
            if state.review is not None:
                break
            response = client.responses.create(
                model=self.model,
                instructions=instructions,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=build_openai_tool_specs(),
            )

        if state.plan is not None and state.review is None and state.evidence is not None and state.context is not None:
            state.review = registry.call(
                "create_review_checkpoint",
                trace,
                plan=state.plan,
                evidence=state.evidence,
                context=state.context,
            )

        if (
            not state.candidates
            or state.context is None
            or not state.evidence
            or state.plan is None
            or state.review is None
        ):
            raise RuntimeError("OpenAI controller ended before producing a complete AgentRun")
        return AgentRun(
            question=question,
            candidates=state.candidates,
            context=state.context,
            evidence=state.evidence,
            plan=state.plan,
            review=state.review,
            trace=trace,
        )

    def _execute_tool_call(
        self,
        call: dict[str, Any],
        question: ResearchQuestion,
        registry: ToolRegistry,
        project_name: str,
        state: _ToolState,
        trace: list[Any],
    ) -> Any:
        name = call["name"]
        arguments = call["arguments"]
        if name == "search_literature":
            query = str(arguments.get("query") or question.text)
            top_k = int(arguments.get("top_k") or 4)
            state.candidates = registry.call(name, trace, query=query, top_k=top_k)
            return to_jsonable(state.candidates)
        if name == "read_experiment_context":
            requested_project = str(arguments.get("project_name") or project_name)
            state.context = registry.call(name, trace, project_name=requested_project)
            return to_jsonable(state.context)
        if name == "score_transfer_fit":
            if state.candidates is None or state.context is None:
                return {"error": "search_literature and read_experiment_context must run first"}
            state.evidence = registry.call(
                name,
                trace,
                candidates=state.candidates,
                context=state.context,
            )
            return to_jsonable(state.evidence)
        if name == "draft_experiment_plan":
            if state.evidence is None or state.context is None:
                return {"error": "score_transfer_fit and read_experiment_context must run first"}
            state.plan = registry.call(
                name,
                trace,
                evidence=state.evidence,
                context=state.context,
            )
            return to_jsonable(state.plan)
        if name == "create_review_checkpoint":
            if state.plan is None or state.evidence is None or state.context is None:
                return {"error": "draft_experiment_plan, score_transfer_fit, and read_experiment_context must run first"}
            state.review = registry.call(
                name,
                trace,
                plan=state.plan,
                evidence=state.evidence,
                context=state.context,
            )
            return to_jsonable(state.review)
        return {"error": f"unknown tool: {name}"}


def _extract_function_calls(response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        item_type = _get_field(item, "type")
        if item_type != "function_call":
            continue
        name = _get_field(item, "name")
        call_id = _get_field(item, "call_id")
        raw_arguments = _get_field(item, "arguments") or "{}"
        calls.append(
            {
                "name": name,
                "call_id": call_id,
                "arguments": json.loads(raw_arguments),
            }
        )
    return calls


def _get_field(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)
