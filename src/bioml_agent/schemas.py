from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResearchQuestion:
    text: str
    domain: str = "promoter_expression_prediction"


@dataclass(frozen=True)
class PaperCandidate:
    paper_id: str
    title: str
    year: int
    tags: tuple[str, ...]
    claim: str
    risk: str
    source_url: str = ""
    doi: str = ""
    venue: str = ""
    citation_count: int | None = None


@dataclass(frozen=True)
class ExperimentContext:
    project_name: str
    trusted_split: str
    diagnostic_split: str
    baseline: str
    metrics: tuple[str, ...]
    constraints: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    paper_id: str
    title: str
    relevance: float
    transfer_fit: str
    supports: str
    cautions: str
    source_url: str = ""
    doi: str = ""
    venue: str = ""
    citation_count: int | None = None


@dataclass(frozen=True)
class CitedClaim:
    claim: str
    evidence_ids: tuple[str, ...]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentPlan:
    summary: str
    matrix: tuple[str, ...]
    promotion_gate: tuple[str, ...]
    risks: tuple[str, ...]
    recommendation: str
    cited_claims: tuple[CitedClaim, ...] = ()


@dataclass(frozen=True)
class ReviewCheckpoint:
    checkpoint_id: str
    status: str
    reason: str
    required_actions: tuple[str, ...]


@dataclass
class TraceEvent:
    step: str
    tool: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


@dataclass
class AgentRun:
    question: ResearchQuestion
    candidates: list[PaperCandidate]
    context: ExperimentContext
    evidence: list[EvidenceItem]
    plan: ExperimentPlan
    review: ReviewCheckpoint
    trace: list[TraceEvent] = field(default_factory=list)
