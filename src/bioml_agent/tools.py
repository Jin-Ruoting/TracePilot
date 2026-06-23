from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .context import ExperimentContextProvider, FixtureExperimentContextProvider
from .literature import FixtureLiteratureProvider, LiteratureProvider
from .schemas import (
    CitedClaim,
    EvidenceItem,
    ExperimentContext,
    ExperimentPlan,
    PaperCandidate,
    ReviewCheckpoint,
    TraceEvent,
)
from .source_safety import candidate_safety_findings

ToolFn = Callable[..., Any]

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = fn

    def call(self, name: str, trace: list[TraceEvent], **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        output = self._tools[name](**kwargs)
        trace.append(
            TraceEvent(
                step=f"call:{name}",
                tool=name,
                inputs=kwargs,
                outputs=_summarize_output(output),
            )
        )
        return output

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def build_default_registry(
    literature_provider: LiteratureProvider | None = None,
    context_provider: ExperimentContextProvider | None = None,
) -> ToolRegistry:
    provider = literature_provider or FixtureLiteratureProvider()
    context_reader = context_provider or FixtureExperimentContextProvider()
    registry = ToolRegistry()
    registry.register(
        "search_literature",
        lambda query, top_k=4: search_literature(query=query, top_k=top_k, provider=provider),
    )
    registry.register(
        "read_experiment_context",
        lambda project_name="EpiCETRA": read_experiment_context(project_name, provider=context_reader),
    )
    registry.register("score_transfer_fit", score_transfer_fit)
    registry.register("draft_experiment_plan", draft_experiment_plan)
    registry.register("create_review_checkpoint", create_review_checkpoint)
    return registry


def search_literature(
    query: str,
    top_k: int = 4,
    provider: LiteratureProvider | None = None,
) -> list[PaperCandidate]:
    return (provider or FixtureLiteratureProvider()).search(query=query, top_k=top_k)


def read_experiment_context(
    project_name: str = "EpiCETRA",
    provider: ExperimentContextProvider | None = None,
) -> ExperimentContext:
    return (provider or FixtureExperimentContextProvider()).read(project_name)


def score_transfer_fit(
    candidates: list[PaperCandidate],
    context: ExperimentContext,
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for rank, paper in enumerate(candidates):
        safety_findings = candidate_safety_findings(paper)
        if safety_findings:
            relevance = 0.1
            fit = "low"
            matched = ", ".join(f"{finding.kind}:{finding.matched_text}" for finding in safety_findings[:2])
            supports = "Source content contains instruction-like text and is not trusted as method evidence."
            cautions = f"Source-poisoning risk detected ({matched}); do not promote without manual source review."
        else:
            relevance = min(1.0, _estimate_relevance(paper) + max(0.0, 0.15 - rank * 0.05))
            if "retrieval" in paper.tags:
                fit = "high"
                supports = "Retrieval-style conditioning maps cleanly to tissue/sample-neighbor hypotheses."
                cautions = "Neighbor selection must respect chromosome-aware and gene-level boundaries."
            elif "graph" in paper.tags:
                fit = "medium"
                supports = "Graph priors align with enhancer-promoter regulation and distal context."
                cautions = "Requires reliable edge construction; start with diagnostic ablation before scaling."
            elif "moe" in paper.tags:
                fit = "medium"
                supports = "Routing may help heterogeneous tissue or EID-conditioned regimes."
                cautions = "Needs router diagnostics to avoid average-metric-only gains."
            else:
                fit = "low"
                supports = "Representation idea may transfer after a controlled objective ablation."
                cautions = "Weak direct overlap with the current baseline; keep as later-stage idea."
        evidence.append(
            EvidenceItem(
                evidence_id=f"E{rank + 1}",
                paper_id=paper.paper_id,
                title=paper.title,
                relevance=relevance,
                transfer_fit=fit,
                supports=supports,
                cautions=f"{cautions} Context gate: {context.trusted_split}.",
                source_url=paper.source_url,
                doi=paper.doi,
                venue=paper.venue,
                citation_count=paper.citation_count,
            )
        )
    return evidence


def draft_experiment_plan(
    evidence: list[EvidenceItem],
    context: ExperimentContext,
) -> ExperimentPlan:
    ranked = sorted(evidence, key=lambda item: item.relevance, reverse=True)
    top = evidence[0] if evidence else None
    if top and top.transfer_fit == "high":
        recommendation = "promote to small controlled screen"
        summary = "Start with retrieval-aware conditioning because it has the clearest transfer path and testable leakage controls."
    elif top and _should_run_diagnostic(top):
        recommendation = "run diagnostic-only screen"
        summary = f"Evidence for {top.title} is plausible but indirect; keep the first experiment small and diagnostic."
    elif top:
        recommendation = "do not run"
        summary = f"Evidence for {top.title} is too weak or off-task; abstain from experiment implementation."
    else:
        recommendation = "do not run"
        summary = "No supporting evidence was found."

    matrix = _experiment_matrix_for(top, context, recommendation=recommendation)
    promotion_gate = (
        "Pearson improves on chromosome-aware split",
        "MSE and R2 remain scale-consistent with the trusted baseline",
        "sample-random gains are not used as main evidence",
        "no neighbor or metadata lookup crosses evaluation boundaries",
    )
    risks = _risks_for(ranked=ranked, recommendation=recommendation)
    return ExperimentPlan(
        summary=summary,
        matrix=matrix,
        promotion_gate=promotion_gate,
        risks=risks,
        recommendation=recommendation,
        cited_claims=_cited_claims_for(evidence=evidence, recommendation=recommendation),
    )


def create_review_checkpoint(
    plan: ExperimentPlan,
    evidence: list[EvidenceItem],
    context: ExperimentContext,
) -> ReviewCheckpoint:
    if "promote" in plan.recommendation:
        return ReviewCheckpoint(
            checkpoint_id="approval:promotion-gate",
            status="requires_approval",
            reason="Promotion requires human approval before experiment execution.",
            required_actions=(
                "review cited claims and source refs",
                f"confirm main evaluation uses {context.trusted_split}",
                "confirm no retrieval, metadata, or graph lookup crosses evaluation boundaries",
            ),
        )
    if any(item.transfer_fit == "medium" for item in evidence):
        return ReviewCheckpoint(
            checkpoint_id="approval:diagnostic-record",
            status="not_required",
            reason="Diagnostic-only screen can be recorded without promotion approval.",
            required_actions=(),
        )
    return ReviewCheckpoint(
        checkpoint_id="approval:no-run",
        status="not_required",
        reason="No promotion decision is proposed.",
        required_actions=(),
    )


def _estimate_relevance(paper: PaperCandidate) -> float:
    score = 0.25
    if "gene_expression" in paper.tags or "retrieval" in paper.tags:
        score += 0.35
    if "conditioning" in paper.tags or "biology_prior" in paper.tags:
        score += 0.2
    if "foundation_model" in paper.tags or "sequence_modeling" in paper.tags:
        score += 0.15
    if paper.year >= 2023:
        score += 0.1
    return min(score, 1.0)


def _should_run_diagnostic(item: EvidenceItem) -> bool:
    if item.transfer_fit == "medium":
        return True
    text = item.title.lower()
    background_terms = ("review", "survey", "primer", "tutorial", "overview")
    if any(term in text for term in background_terms):
        return False
    diagnostic_terms = (
        "contrast",
        "rank",
        "representation",
        "regression",
        "regulatory",
        "sequence",
        "genomic",
        "foundation",
        "graph",
        "gene expression",
        "promoter",
    )
    return item.transfer_fit == "low" and item.relevance >= 0.4 and any(term in text for term in diagnostic_terms)


def _risks_for(ranked: list[EvidenceItem], recommendation: str) -> tuple[str, ...]:
    if not ranked:
        return ("No task-aligned evidence was found; do not spend training budget.",)
    if recommendation == "do not run":
        return tuple(item.cautions for item in ranked[:2]) + (
            "Keep this as an abstention record, not a model contribution.",
        )
    return tuple(item.cautions for item in ranked[:3]) + (
        "If gains only appear on sample-random split, mark as leakage-risk diagnostic.",
    )


def _experiment_matrix_for(
    top: EvidenceItem | None,
    context: ExperimentContext,
    recommendation: str,
) -> tuple[str, ...]:
    control = f"control: {context.baseline}"
    if recommendation == "do not run":
        return (
            "abstain: do not schedule experiment",
            "follow-up: search for closer promoter-expression evidence before implementation",
            "record: keep as negative-screening note, not a model contribution",
            "gate: revisit only if new evidence maps to chromosome-aware evaluation",
        )
    if top and top.transfer_fit == "high":
        variants = (
            "variant A: add retrieval features with train-split-only neighbor index",
            "variant B: retrieval features with shuffled-neighbor negative control",
            "variant C: remove new module while keeping preprocessing identical",
        )
    elif top and ("graph" in top.paper_id.lower() or "graph" in top.title.lower()):
        variants = (
            "variant A: add regulatory graph prior built only from training metadata",
            "variant B: edge-drop graph ablation to test distal-edge dependence",
            "variant C: no-graph control with identical promoter and distal inputs",
        )
    elif top and "moe" in top.paper_id:
        variants = (
            "variant A: add EID/tissue-conditioned router with expert usage logging",
            "variant B: entropy-regularized router to reduce expert collapse",
            "variant C: shared-expert control with the same parameter budget",
        )
    else:
        variants = (
            "variant A: add the candidate representation objective",
            "variant B: shuffled-label or shuffled-pair negative control",
            "variant C: remove the new objective with identical data preprocessing",
        )
    return (
        control,
        *variants,
        "stratify: report by chromosome, EID/tissue, and expression band",
    )


def _cited_claims_for(evidence: list[EvidenceItem], recommendation: str) -> tuple[CitedClaim, ...]:
    if not evidence:
        return ()
    top = evidence[0]
    claims = [
        CitedClaim(
            claim=f"Recommendation '{recommendation}' is grounded in the transfer fit of {top.title}.",
            evidence_ids=(top.evidence_id,),
            source_refs=_source_refs(top),
        )
    ]
    if top.transfer_fit == "high":
        claims.append(
            CitedClaim(
                claim="Retrieval-style methods need split-aware neighbor controls before promotion.",
                evidence_ids=(top.evidence_id,),
                source_refs=_source_refs(top),
            )
        )
    elif recommendation == "do not run":
        claims.append(
            CitedClaim(
                claim="Weak or off-task evidence should be recorded as abstention instead of an experiment.",
                evidence_ids=(top.evidence_id,),
                source_refs=_source_refs(top),
            )
        )
    else:
        claims.append(
            CitedClaim(
                claim="The candidate method should stay diagnostic until stronger task-aligned evidence appears.",
                evidence_ids=(top.evidence_id,),
                source_refs=_source_refs(top),
            )
        )
    return tuple(claims)


def _source_refs(item: EvidenceItem) -> tuple[str, ...]:
    refs = []
    if item.source_url:
        refs.append(item.source_url)
    if item.doi:
        refs.append(f"doi:{item.doi}")
    return tuple(refs)


def _summarize_output(output: Any) -> dict[str, Any]:
    if isinstance(output, list):
        return {"type": "list", "count": len(output), "items": [_brief(item) for item in output[:3]]}
    return {"type": type(output).__name__, "value": _brief(output)}


def _brief(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        data = asdict(value)
        return {key: data[key] for key in list(data)[:4]}
    return value
