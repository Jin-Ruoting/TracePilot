from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..rag.index import RagIndex
from ..rag.ingestion import Ingestor
from ..workers.coding import file_manifest
from .models import digest


def build_context(workspace: str | Path, goal: str, *, evidence_paths: list[str] | None = None,
                  files: list[str] | None = None) -> dict[str, Any]:
    """Reuse TracePilot's offline hybrid RAG and retain byte-exact code for later patch proposals."""
    root = Path(workspace).expanduser().resolve()
    manifest = file_manifest(root)
    ingestor = Ingestor()
    documents = []
    for relative in manifest:
        documents.extend(ingestor.ingest_path(root / relative))
    for source in evidence_paths or []:
        documents.extend(ingestor.ingest_path(source))
    selected = files if files is not None else [name for name in manifest if name.endswith(".py")][:8]
    code, total = [], 0
    for relative in selected:
        if relative not in manifest:
            raise ValueError("context file is not in the contained code manifest: " + relative)
        raw = (root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[relative]:
            raise ValueError("workspace changed while building model context")
        total += len(raw)
        if total > 250_000:
            raise ValueError("selected model-context code exceeds 250 KB; choose fewer files")
        code.append({"path": relative, "sha256": manifest[relative], "content": raw.decode("utf-8")})
    with RagIndex(":memory:") as index:
        index.index_documents(documents, namespace="research-context")
        chunks = index.search(goal, top_k=8, namespace="research-context")
    return {
        "schema_version": 1, "goal": goal, "workspace": str(root), "code_hash": digest(manifest),
        "files": code, "evidence": [
            {"evidence_id": chunk.chunk_id, "source_uri": chunk.source_uri, "title": chunk.title,
             "content_hash": chunk.content_hash, "text": chunk.text,
             "start_line": chunk.start_line, "end_line": chunk.end_line, "page": chunk.page}
            for chunk in chunks
        ],
        "retrieval": "Existing TracePilot FTS5 + deterministic hash vectors + reciprocal rank fusion; no neural embedding",
        "novelty_status": "unverified", "model_called": False,
    }
