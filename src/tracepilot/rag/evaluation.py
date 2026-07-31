from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from tracepilot.models import EvidenceChunk, EvidenceClaim

from .safety import has_prompt_injection


_WORD_RE = re.compile(r"[^\W_]+(?:[.+#/-][^\W_]+)*", re.UNICODE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|cannot|can't|doesn't|isn't|unsupported)\b|"
    r"(?:不支持|不能|无法|未提供|没有)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


@dataclass(frozen=True)
class CitationCoverageResult:
    passed: bool
    coverage: float
    total_supported_claims: int
    cited_claims: int
    missing_claim_ids: tuple[str, ...]
    unknown_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClaimFaithfulness:
    claim_id: str
    score: float
    passed: bool
    evidence_ids: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class FaithfulnessResult:
    passed: bool
    score: float
    claims: tuple[ClaimFaithfulness, ...]
    unsupported_claim_ids: tuple[str, ...]
    contaminated_claim_ids: tuple[str, ...]


def bind_claim(
    text: str,
    chunks: Iterable[EvidenceChunk],
    *,
    claim_id: str | None = None,
    max_evidence: int = 3,
    min_support: float = 0.8,
    supported: bool | None = None,
    uncertainty: str = "",
) -> EvidenceClaim:
    if max_evidence < 1:
        raise ValueError("max_evidence must be positive")
    candidates: list[tuple[float, EvidenceChunk]] = []
    for chunk in chunks:
        score = lexical_support_score(text, chunk.text)
        if (
            score >= min_support
            and _numbers_supported(text, chunk.text)
            and _negation_consistent(text, chunk.text)
            and not _chunk_is_contaminated(chunk)
        ):
            candidates.append((score, chunk))
    candidates.sort(
        key=lambda item: (-item[0], -float(item[1].score), item[1].chunk_id)
    )
    selected = candidates[:max_evidence]
    evidence_ids = tuple(chunk.chunk_id for _, chunk in selected)
    inferred_supported = bool(evidence_ids)
    is_supported = inferred_supported if supported is None else bool(supported and evidence_ids)
    confidence = max((score for score, _ in selected), default=0.0)
    if not is_supported and not uncertainty:
        uncertainty = "No indexed evidence met the support threshold; abstaining."
    identifier = claim_id or (
        "claim-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    )
    return EvidenceClaim(
        claim_id=identifier,
        text=text,
        evidence_ids=evidence_ids,
        supported=is_supported,
        confidence=confidence,
        uncertainty=uncertainty,
        metadata={
            "binding_method": "lexical_overlap",
            "support_scores": {
                chunk.chunk_id: round(score, 6)
                for score, chunk in selected
            },
        },
    )


bind_claim_to_chunks = bind_claim


def check_citation_coverage(
    claims: Iterable[EvidenceClaim],
    chunks: Iterable[EvidenceChunk] | None = None,
) -> CitationCoverageResult:
    claim_values = tuple(claims)
    known_ids = (
        {chunk.chunk_id for chunk in chunks}
        if chunks is not None
        else None
    )
    supported_claims = [claim for claim in claim_values if claim.supported]
    missing = tuple(
        claim.claim_id
        for claim in supported_claims
        if not claim.evidence_ids
    )
    unknown = tuple(
        sorted(
            {
                evidence_id
                for claim in supported_claims
                for evidence_id in claim.evidence_ids
                if known_ids is not None and evidence_id not in known_ids
            }
        )
    )
    cited = sum(
        bool(claim.evidence_ids)
        and not (
            known_ids is not None
            and any(evidence_id not in known_ids for evidence_id in claim.evidence_ids)
        )
        for claim in supported_claims
    )
    total = len(supported_claims)
    coverage = cited / total if total else 1.0
    return CitationCoverageResult(
        passed=not missing and not unknown,
        coverage=coverage,
        total_supported_claims=total,
        cited_claims=cited,
        missing_claim_ids=missing,
        unknown_evidence_ids=unknown,
    )


citation_coverage = check_citation_coverage


def check_faithfulness(
    claims: Iterable[EvidenceClaim],
    chunks: Iterable[EvidenceChunk],
    *,
    threshold: float = 0.8,
) -> FaithfulnessResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    checks: list[ClaimFaithfulness] = []
    unsupported: list[str] = []
    contaminated: list[str] = []
    for claim in claims:
        if not claim.supported:
            continue
        cited = [
            chunk_by_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in chunk_by_id
        ]
        if not cited:
            unsupported.append(claim.claim_id)
            checks.append(
                ClaimFaithfulness(
                    claim_id=claim.claim_id,
                    score=0.0,
                    passed=False,
                    evidence_ids=claim.evidence_ids,
                    reason="no resolvable evidence",
                )
            )
            continue
        if any(_chunk_is_contaminated(chunk) for chunk in cited):
            contaminated.append(claim.claim_id)
        combined_evidence = "\n".join(chunk.text for chunk in cited)
        score = lexical_support_score(claim.text, combined_evidence)
        clause_scores = _clause_support_scores(claim.text, combined_evidence)
        numbers_supported = _numbers_supported(claim.text, combined_evidence)
        negation_consistent = _negation_consistent(
            claim.text,
            combined_evidence,
        )
        claim_contaminated = has_prompt_injection(claim.text)
        if claim_contaminated:
            contaminated.append(claim.claim_id)
        passed = (
            score >= threshold
            and all(item >= threshold for item in clause_scores)
            and numbers_supported
            and negation_consistent
            and claim.claim_id not in contaminated
        )
        if not passed:
            unsupported.append(claim.claim_id)
        reason_parts: list[str] = []
        if score < threshold:
            reason_parts.append("insufficient lexical support")
        if any(item < threshold for item in clause_scores):
            reason_parts.append("one or more claim clauses lack lexical support")
        if not numbers_supported:
            reason_parts.append("claim numbers are absent from evidence")
        if not negation_consistent:
            reason_parts.append("claim and evidence disagree on negation")
        if claim.claim_id in contaminated:
            reason_parts.append(
                "claim or cited evidence contains prompt-injection markers"
            )
        checks.append(
            ClaimFaithfulness(
                claim_id=claim.claim_id,
                score=score,
                passed=passed,
                evidence_ids=claim.evidence_ids,
                reason="; ".join(reason_parts),
            )
        )
    aggregate = (
        sum(check.score for check in checks) / len(checks)
        if checks
        else 1.0
    )
    return FaithfulnessResult(
        passed=all(check.passed for check in checks),
        score=aggregate,
        claims=tuple(checks),
        unsupported_claim_ids=tuple(dict.fromkeys(unsupported)),
        contaminated_claim_ids=tuple(dict.fromkeys(contaminated)),
    )


faithfulness_check = check_faithfulness


def lexical_support_score(claim: str, evidence: str) -> float:
    claim_terms = _terms(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = _terms(evidence)
    overlap = claim_terms & evidence_terms
    return len(overlap) / len(claim_terms)


def _clause_support_scores(claim: str, evidence: str) -> tuple[float, ...]:
    clauses = tuple(
        item.strip()
        for item in re.split(r"(?:[;!?。；！？]+|\.(?=\s|$))", claim)
        if _terms(item)
    )
    return tuple(
        lexical_support_score(clause, evidence)
        for clause in clauses
    ) or (lexical_support_score(claim, evidence),)


def _terms(text: str) -> set[str]:
    terms = {
        match.group(0).casefold()
        for match in _WORD_RE.finditer(text)
        if match.group(0).casefold() not in _STOPWORDS
    }
    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        terms.update(
            sequence[index : index + 2]
            for index in range(len(sequence) - 1)
        )
    return terms


def _numbers_supported(claim: str, evidence: str) -> bool:
    claim_numbers = set(_NUMBER_RE.findall(claim))
    if not claim_numbers:
        return True
    return claim_numbers <= set(_NUMBER_RE.findall(evidence))


def _negation_consistent(claim: str, evidence: str) -> bool:
    return bool(_NEGATION_RE.search(claim)) == bool(_NEGATION_RE.search(evidence))


def _chunk_is_contaminated(chunk: EvidenceChunk) -> bool:
    marker = chunk.metadata.get("prompt_injection_detected")
    return bool(marker) or has_prompt_injection(chunk.text)
