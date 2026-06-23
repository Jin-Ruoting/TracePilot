from __future__ import annotations

from typing import Protocol

from .schemas import AgentRun, ResearchQuestion
from .state_graph import initial_state, run_state_graph, state_to_run
from .tools import ToolRegistry


class ResearchController(Protocol):
    def run(
        self,
        question: ResearchQuestion,
        registry: ToolRegistry,
        project_name: str = "EpiCETRA",
    ) -> AgentRun:
        ...


class DeterministicResearchController:
    """Offline controller that keeps tool contracts testable without an LLM."""

    def run(
        self,
        question: ResearchQuestion,
        registry: ToolRegistry,
        project_name: str = "EpiCETRA",
    ) -> AgentRun:
        state = initial_state(question, project_name=project_name)
        return state_to_run(run_state_graph(registry, state))
