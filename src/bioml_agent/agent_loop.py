from __future__ import annotations

from .controllers import DeterministicResearchController, ResearchController
from .schemas import AgentRun, ResearchQuestion
from .tools import ToolRegistry, build_default_registry


class TracePilotAgent:
    """Facade over a TracePilot planning controller and a shared tool registry."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        controller: ResearchController | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.controller = controller or DeterministicResearchController()

    def run(self, question_text: str, project_name: str = "EpiCETRA") -> AgentRun:
        question = ResearchQuestion(text=question_text)
        return self.controller.run(question, self.registry, project_name=project_name)


class BioMLResearchAgent(TracePilotAgent):
    """Backward-compatible alias for the original portfolio package API."""
