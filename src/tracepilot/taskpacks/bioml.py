"""Legacy BioML task pack retained as an optional TracePilot example."""

from __future__ import annotations

from typing import Any

BIOML_TASKPACK = {
    "name": "bioml",
    "description": "Evidence-grounded BioML experiment-planning compatibility task pack.",
    "default_project": "EpiCETRA",
    "legacy_package": "bioml_agent",
}


def build_legacy_bioml_agent(**kwargs: Any) -> Any:
    """Build the pre-0.2 BioML agent without coupling it to the generic core."""

    from bioml_agent.agent_loop import BioMLResearchAgent

    return BioMLResearchAgent(**kwargs)
