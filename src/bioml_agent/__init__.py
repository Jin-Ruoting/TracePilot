"""Deprecated BioML compatibility package for TracePilot 0.2."""

from __future__ import annotations

import warnings

from tracepilot.agent import TracePilotAgent as GenericTracePilotAgent

from .agent_loop import BioMLResearchAgent, TracePilotAgent

warnings.warn(
    "bioml_agent is deprecated; use the tracepilot package for new projects",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BioMLResearchAgent",
    "GenericTracePilotAgent",
    "TracePilotAgent",
    "__version__",
]

__version__ = "0.2.0"
