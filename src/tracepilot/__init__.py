"""TracePilot: a traceable agent for technical research and decisions."""

from .agent import TracePilotAgent
from .execution import ExecutionRuntime
from .memory import MemoryStore
from .models import (
    AgentRun,
    ApprovalRequest,
    EvidenceChunk,
    EvidenceClaim,
    MemoryRecord,
    PlanStep,
    SourceDocument,
    TaskContext,
    TaskPlan,
    TaskRequest,
    ToolResult,
    TraceEvent,
)
from .planning import DeterministicPlanner
from .rag import RagIndex
from .tooling import EffectClass, ToolContract, ToolRegistry

__all__ = [
    "AgentRun",
    "ApprovalRequest",
    "EvidenceChunk",
    "EvidenceClaim",
    "EffectClass",
    "ExecutionRuntime",
    "MemoryRecord",
    "MemoryStore",
    "PlanStep",
    "RagIndex",
    "SourceDocument",
    "TaskContext",
    "TaskPlan",
    "TaskRequest",
    "ToolResult",
    "ToolContract",
    "ToolRegistry",
    "TraceEvent",
    "TracePilotAgent",
    "DeterministicPlanner",
    "__version__",
]

__version__ = "0.2.0"
