from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """Return a compact, timezone-aware timestamp suitable for persistence."""

    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def content_digest(content: str | bytes) -> str:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(payload).hexdigest()


class TaskStatus(str, Enum):
    CREATED = "created"
    INDEXING = "indexing"
    PLANNING = "planning"
    RESEARCHING = "researching"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    NEEDS_REVISION = "needs_revision"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class MemoryKind(str, Enum):
    WORKING = "working"
    RUN = "run"
    DECISION = "decision"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    DISABLED = "disabled"


_STATUS_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset(
        {TaskStatus.INDEXING, TaskStatus.PLANNING, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.INDEXING: frozenset(
        {TaskStatus.PLANNING, TaskStatus.NEEDS_REVISION, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.PLANNING: frozenset(
        {TaskStatus.RESEARCHING, TaskStatus.NEEDS_REVISION, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.RESEARCHING: frozenset(
        {
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.VERIFYING,
            TaskStatus.NEEDS_REVISION,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.AWAITING_APPROVAL: frozenset(
        {TaskStatus.EXECUTING, TaskStatus.NEEDS_REVISION, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.EXECUTING: frozenset(
        {TaskStatus.VERIFYING, TaskStatus.NEEDS_REVISION, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.VERIFYING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.NEEDS_REVISION, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.NEEDS_REVISION: frozenset(
        {TaskStatus.PLANNING, TaskStatus.RESEARCHING, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True)
class TaskRequest:
    goal: str = ""
    workspace_id: str = "default"
    workspace_path: str = "."
    source_urls: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    execution_policy: str = "approval_required"
    task_id: str = field(default_factory=lambda: new_id("task"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_path", str(Path(self.workspace_path).expanduser().resolve()))
        object.__setattr__(self, "source_urls", tuple(self.source_urls))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SourceDocument:
    document_id: str = field(default_factory=lambda: new_id("doc"))
    source_type: str = "text"
    uri: str = ""
    title: str = ""
    content: str = ""
    content_hash: str = ""
    version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", content_digest(self.content))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str = field(default_factory=lambda: new_id("chunk"))
    document_id: str = ""
    text: str = ""
    content_hash: str = ""
    source_uri: str = ""
    title: str = ""
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", content_digest(self.text))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content(self) -> str:
        """Compatibility alias used by context assemblers."""

        return self.text


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str = field(default_factory=lambda: new_id("claim"))
    text: str = ""
    evidence_ids: tuple[str, ...] = ()
    supported: bool = True
    confidence: float = 1.0
    uncertainty: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def claim(self) -> str:
        return self.text


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str = field(default_factory=lambda: new_id("memory"))
    workspace_id: str = "default"
    kind: MemoryKind | str = MemoryKind.WORKING
    content: Any = field(default_factory=dict)
    status: MemoryStatus | str = MemoryStatus.ACTIVE
    source_task_id: str = ""
    workspace_revision: str = ""
    source_versions: dict[str, str] = field(default_factory=dict)
    supersedes_id: str = ""
    confirmed: bool = False
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        object.__setattr__(self, "status", MemoryStatus(self.status))
        object.__setattr__(self, "source_versions", dict(self.source_versions))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class TaskContext:
    task_id: str = ""
    workspace_id: str = "default"
    workspace_path: str = "."
    workspace_revision: str = ""
    workspace_hash: str = ""
    source_versions: dict[str, str] = field(default_factory=dict)
    evidence: list[EvidenceChunk] = field(default_factory=list)
    memories: list[MemoryRecord] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace_path = str(Path(self.workspace_path).expanduser().resolve())
        self.source_versions = dict(self.source_versions)
        self.evidence = list(self.evidence)
        self.memories = list(self.memories)
        self.state = dict(self.state)
        self.metadata = dict(self.metadata)


@dataclass(frozen=True)
class PlanStep:
    step_id: str = field(default_factory=lambda: new_id("step"))
    description: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    effect_class: str = "read"
    required_evidence: bool = False
    result_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def id(self) -> str:
        return self.step_id

    @property
    def tool(self) -> str:
        return self.tool_name

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.depends_on


@dataclass(frozen=True)
class TaskPlan:
    task_id: str = ""
    steps: tuple[PlanStep, ...] = ()
    revision: int = 1
    rationale: str = ""
    max_concurrency: int = 4
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def step(self, step_id: str) -> PlanStep:
        for item in self.steps:
            if item.step_id == step_id:
                return item
        raise KeyError(f"unknown plan step: {step_id}")


@dataclass(frozen=True)
class ToolResult:
    result_id: str = field(default_factory=lambda: new_id("result"))
    task_id: str = ""
    run_id: str = ""
    step_id: str = ""
    tool_name: str = ""
    effect_class: str = "read"
    success: bool = True
    output: Any = None
    error: str = ""
    idempotency_key: str = ""
    started_at: str = field(default_factory=utc_now)
    finished_at: str = field(default_factory=utc_now)
    duration_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str = field(default_factory=lambda: new_id("approval"))
    task_id: str = ""
    run_id: str = ""
    step_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_hash: str = ""
    status: ApprovalStatus | str = ApprovalStatus.PENDING
    requested_at: str = field(default_factory=utc_now)
    decided_at: str = ""
    decided_by: str = ""
    rejection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "status", ApprovalStatus(self.status))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TraceEvent:
    event_id: str = field(default_factory=lambda: new_id("event"))
    task_id: str = ""
    run_id: str = ""
    step_id: str = ""
    event_type: str = ""
    timestamp: str = field(default_factory=utc_now)
    tool_name: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    error: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "outputs", dict(self.outputs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tool(self) -> str:
        return self.tool_name


@dataclass
class AgentRun:
    run_id: str = field(default_factory=lambda: new_id("run"))
    task_id: str = ""
    request: TaskRequest = field(default_factory=TaskRequest)
    context: TaskContext = field(default_factory=TaskContext)
    plan: TaskPlan = field(default_factory=TaskPlan)
    evidence: list[EvidenceChunk | EvidenceClaim] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    status: TaskStatus | str = TaskStatus.CREATED
    summary: str = ""
    error: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = self.request.task_id
        self.status = TaskStatus(self.status)
        self.evidence = list(self.evidence)
        self.tool_results = list(self.tool_results)
        self.approvals = list(self.approvals)
        self.trace = list(self.trace)
        self.metadata = dict(self.metadata)

    @property
    def task(self) -> TaskRequest:
        return self.request

    def transition(self, status: TaskStatus | str) -> None:
        target = TaskStatus(status)
        if target == self.status:
            return
        if target not in _STATUS_TRANSITIONS[self.status]:
            raise ValueError(f"invalid task status transition: {self.status.value} -> {target.value}")
        self.status = target
        self.updated_at = utc_now()


def update_approval(request: ApprovalRequest, **changes: Any) -> ApprovalRequest:
    """Typed helper used by persistence and execution adapters."""

    return replace(request, **changes)


__all__ = [
    "AgentRun",
    "ApprovalRequest",
    "ApprovalStatus",
    "EvidenceChunk",
    "EvidenceClaim",
    "MemoryKind",
    "MemoryRecord",
    "MemoryStatus",
    "PlanStep",
    "SourceDocument",
    "TaskContext",
    "TaskPlan",
    "TaskRequest",
    "TaskStatus",
    "ToolResult",
    "TraceEvent",
    "content_digest",
    "new_id",
    "update_approval",
    "utc_now",
]
