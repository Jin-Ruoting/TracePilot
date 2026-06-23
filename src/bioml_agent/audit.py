from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import AgentRun
from .serialization import to_jsonable
from .source_safety import candidate_safety_findings, is_poisoning_guarded


@dataclass(frozen=True)
class AuditFinding:
    name: str
    status: str
    severity: str
    detail: str


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    decision: str
    score: float
    summary: str
    findings: tuple[AuditFinding, ...]
    required_actions: tuple[str, ...]


def audit_run(run: AgentRun) -> AuditReport:
    findings = (
        _audit_promotion_evidence(run),
        _audit_abstention_quality(run),
        _audit_split_controls(run),
        _audit_experiment_matrix(run),
        _audit_review_checkpoint(run),
        _audit_source_grounding(run),
        _audit_source_poisoning(run),
    )
    failures = [finding for finding in findings if finding.status == "fail"]
    warnings = [finding for finding in findings if finding.status == "warn"]
    passed = not failures
    score = round(sum(1 for finding in findings if finding.status == "pass") / len(findings), 3)
    decision = _decision_for(run, passed=passed)
    required_actions = tuple(finding.detail for finding in (*failures, *warnings))
    return AuditReport(
        passed=passed,
        decision=decision,
        score=score,
        summary=_summary_for(run, decision=decision, failures=failures, warnings=warnings),
        findings=findings,
        required_actions=required_actions,
    )


def audit_to_dict(report: AuditReport) -> dict[str, object]:
    return to_jsonable(report)


def write_audit_report(report: AuditReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def format_audit_report(report: AuditReport) -> str:
    lines = [
        "## Decision Audit",
        f"Passed: {report.passed}",
        f"Decision: {report.decision}",
        f"Score: {report.score:.3f}",
        f"Summary: {report.summary}",
    ]
    for finding in report.findings:
        lines.append(f"- {finding.status.upper()} {finding.name} [{finding.severity}]: {finding.detail}")
    if report.required_actions:
        lines.append("Required actions:")
        lines.extend(f"- {item}" for item in report.required_actions)
    return "\n".join(lines)


def _audit_promotion_evidence(run: AgentRun) -> AuditFinding:
    if "promote" not in run.plan.recommendation:
        return _pass("promotion_evidence", "info", "run does not request promotion")
    top = run.evidence[0] if run.evidence else None
    cited_ids = {
        evidence_id
        for claim in run.plan.cited_claims
        for evidence_id in claim.evidence_ids
    }
    if top is None:
        return _fail("promotion_evidence", "blocker", "promotion has no supporting evidence")
    if top.transfer_fit != "high":
        return _fail(
            "promotion_evidence",
            "blocker",
            f"promotion top evidence is {top.transfer_fit!r}, expected high transfer fit",
        )
    if top.evidence_id not in cited_ids:
        return _fail("promotion_evidence", "blocker", "promotion top evidence is not cited by plan claims")
    return _pass("promotion_evidence", "critical", "promotion is grounded in cited high-fit top evidence")


def _audit_abstention_quality(run: AgentRun) -> AuditFinding:
    if run.plan.recommendation != "do not run":
        return _pass("abstention_quality", "info", "run does not abstain")
    if not run.evidence:
        return _pass("abstention_quality", "critical", "abstention is justified by no retrieved evidence")
    top = run.evidence[0]
    matrix_text = " ".join(run.plan.matrix).lower()
    if top.transfer_fit in {"high", "medium"}:
        return _fail(
            "abstention_quality",
            "blocker",
            f"abstention conflicts with {top.transfer_fit} transfer-fit top evidence",
        )
    if "do not schedule" not in matrix_text or "negative-screening" not in matrix_text:
        return _fail(
            "abstention_quality",
            "major",
            "abstention lacks do-not-schedule and negative-screening record actions",
        )
    return _pass("abstention_quality", "critical", "weak evidence is kept as abstention and negative screening")


def _audit_split_controls(run: AgentRun) -> AuditFinding:
    gate_text = " ".join(run.plan.promotion_gate).lower()
    required_terms = ("chromosome-aware", "sample-random", "crosses evaluation boundaries")
    missing = [term for term in required_terms if term not in gate_text]
    if missing:
        return _fail("split_controls", "blocker", f"missing split/leakage gate terms: {', '.join(missing)}")
    return _pass("split_controls", "critical", "promotion gate separates main split, diagnostic split, and leakage boundary")


def _audit_experiment_matrix(run: AgentRun) -> AuditFinding:
    matrix_text = " ".join(run.plan.matrix).lower()
    if run.plan.recommendation == "do not run":
        required_terms = ("abstain", "do not schedule", "follow-up")
        success_detail = "abstention matrix records no-run decision, follow-up search, and revisit gate"
    else:
        required_terms = ("control:", "variant a:", "variant b:", "variant c:", "stratify:")
        success_detail = "matrix has baseline/control, variants, and reporting stratification"
    missing = [term for term in required_terms if term not in matrix_text]
    if missing:
        return _fail("experiment_matrix", "major", f"matrix is missing required terms: {', '.join(missing)}")
    if "promote" in run.plan.recommendation and _top_has_retrieval_tag(run):
        retrieval_terms = ("train-split-only", "shuffled-neighbor")
        missing_retrieval = [term for term in retrieval_terms if term not in matrix_text]
        if missing_retrieval:
            return _fail(
                "experiment_matrix",
                "blocker",
                f"retrieval promotion lacks leakage controls: {', '.join(missing_retrieval)}",
            )
    return _pass("experiment_matrix", "major", success_detail)


def _audit_review_checkpoint(run: AgentRun) -> AuditFinding:
    if "promote" in run.plan.recommendation:
        if run.review.status != "requires_approval":
            return _fail("review_checkpoint", "blocker", "promotion does not require human approval")
        actions = " ".join(run.review.required_actions).lower()
        missing = [term for term in ("review cited claims", "chromosome-aware", "crosses evaluation boundaries") if term not in actions]
        if missing:
            return _fail("review_checkpoint", "blocker", f"approval actions are underspecified: {', '.join(missing)}")
        return _pass("review_checkpoint", "critical", "promotion is gated behind specific human approval actions")
    if run.review.status == "requires_approval":
        return _fail("review_checkpoint", "major", "non-promotion plan should not require promotion approval")
    return _pass("review_checkpoint", "major", "non-promotion decision is recorded without promotion approval")


def _audit_source_grounding(run: AgentRun) -> AuditFinding:
    if not run.evidence and run.plan.recommendation == "do not run":
        return _pass("source_grounding", "major", "no-evidence abstention avoids unsupported citations")
    if run.evidence and not run.plan.cited_claims:
        return _fail("source_grounding", "major", "plan uses evidence but has no cited claims")
    evidence_ids = {item.evidence_id for item in run.evidence}
    failures: list[str] = []
    for claim in run.plan.cited_claims:
        if not claim.source_refs:
            failures.append("claim missing source refs")
        unknown = [evidence_id for evidence_id in claim.evidence_ids if evidence_id not in evidence_ids]
        if unknown:
            failures.append(f"claim cites unknown evidence ids: {', '.join(unknown)}")
    if failures:
        return _fail("source_grounding", "major", "; ".join(failures[:3]))
    return _pass("source_grounding", "major", "plan claims are tied to evidence ids and source refs")


def _audit_source_poisoning(run: AgentRun) -> AuditFinding:
    poisoned_ids = {
        candidate.paper_id
        for candidate in run.candidates
        if candidate_safety_findings(candidate)
    }
    if not poisoned_ids:
        return _pass("source_poisoning", "info", "no instruction-like source text detected")
    if "promote" in run.plan.recommendation:
        return _fail("source_poisoning", "blocker", "instruction-like source text must not be promoted")
    evidence_by_paper = {item.paper_id: item for item in run.evidence}
    unguarded = [
        paper_id
        for paper_id in poisoned_ids
        if paper_id not in evidence_by_paper or not is_poisoning_guarded(evidence_by_paper[paper_id])
    ]
    if unguarded:
        return _fail(
            "source_poisoning",
            "blocker",
            f"source-poisoning risk is not explicitly guarded for: {', '.join(sorted(unguarded))}",
        )
    return _pass(
        "source_poisoning",
        "critical",
        "instruction-like source text is downgraded and blocked from promotion",
    )


def _top_has_retrieval_tag(run: AgentRun) -> bool:
    if not run.evidence:
        return False
    top_paper_id = run.evidence[0].paper_id
    for candidate in run.candidates:
        if candidate.paper_id == top_paper_id:
            return "retrieval" in candidate.tags
    return False


def _decision_for(run: AgentRun, passed: bool) -> str:
    if not passed:
        return "needs_revision"
    if "promote" in run.plan.recommendation:
        return "ready_for_human_review"
    if run.plan.recommendation == "do not run":
        return "record_abstention"
    return "record_diagnostic"


def _summary_for(
    run: AgentRun,
    decision: str,
    failures: list[AuditFinding],
    warnings: list[AuditFinding],
) -> str:
    if failures:
        return f"{len(failures)} blocking audit issue(s) must be fixed before trusting this decision."
    if warnings:
        return f"{len(warnings)} audit warning(s) should be reviewed before downstream use."
    if decision == "ready_for_human_review":
        return "Promotion is evidence-grounded but still requires human approval before experiment execution."
    if decision == "record_abstention":
        return "Abstention is supported and should be recorded as a negative-screening result."
    if decision == "record_diagnostic":
        return "Diagnostic plan is supported but should not be promoted without stronger evidence."
    return f"Audit passed for recommendation: {run.plan.recommendation}."


def _pass(name: str, severity: str, detail: str) -> AuditFinding:
    return AuditFinding(name=name, status="pass", severity=severity, detail=detail)


def _fail(name: str, severity: str, detail: str) -> AuditFinding:
    return AuditFinding(name=name, status="fail", severity=severity, detail=detail)
