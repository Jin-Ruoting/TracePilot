from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import EvidenceItem, PaperCandidate


_POISONING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore (all )?(previous|prior|system|developer) instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard (all )?(previous|prior|system|developer) instructions\b", re.IGNORECASE),
    re.compile(r"\breveal (the )?(system prompt|developer message|api key|secrets?|credentials?)\b", re.IGNORECASE),
    re.compile(r"\bexfiltrat(e|ion)\b", re.IGNORECASE),
    re.compile(r"\b(call|invoke|use) (the )?(tool|shell|terminal|python)\b", re.IGNORECASE),
    re.compile(r"\bdelete (files|the repo|all data)\b", re.IGNORECASE),
    re.compile(r"\bbase64\b.*\b(secret|token|credential)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class SourceSafetyFinding:
    source_id: str
    kind: str
    matched_text: str


def candidate_safety_findings(candidate: PaperCandidate) -> tuple[SourceSafetyFinding, ...]:
    fields = {
        "title": candidate.title,
        "claim": candidate.claim,
        "risk": candidate.risk,
        "tags": " ".join(candidate.tags),
    }
    return _scan_fields(source_id=candidate.paper_id, fields=fields)


def evidence_safety_findings(evidence: EvidenceItem) -> tuple[SourceSafetyFinding, ...]:
    fields = {
        "title": evidence.title,
        "supports": evidence.supports,
        "cautions": evidence.cautions,
    }
    return _scan_fields(source_id=evidence.paper_id, fields=fields)


def is_source_poisoned(candidate: PaperCandidate) -> bool:
    return bool(candidate_safety_findings(candidate))


def is_poisoning_guarded(evidence: EvidenceItem) -> bool:
    text = " ".join((evidence.transfer_fit, evidence.supports, evidence.cautions)).lower()
    return (
        evidence.transfer_fit == "low"
        and "source-poisoning" in text
        and "do not promote" in text
    )


def _scan_fields(source_id: str, fields: dict[str, str]) -> tuple[SourceSafetyFinding, ...]:
    findings: list[SourceSafetyFinding] = []
    for kind, value in fields.items():
        for pattern in _POISONING_PATTERNS:
            match = pattern.search(value)
            if match:
                findings.append(
                    SourceSafetyFinding(
                        source_id=source_id,
                        kind=kind,
                        matched_text=match.group(0),
                    )
                )
                break
    return tuple(findings)
