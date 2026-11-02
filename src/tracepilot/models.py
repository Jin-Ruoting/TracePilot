from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"

def content_digest(content: str | bytes) -> str:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(payload).hexdigest()

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
