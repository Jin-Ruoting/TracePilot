from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tracepilot.models import EvidenceChunk, SourceDocument

from .chunking import StructuredChunker
from .embeddings import DeterministicHashEmbedder, cosine_similarity, tokenize
from .ingestion import sha256_text


_RERANK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)


class Embedder(Protocol):
    def embed(self, text: str) -> Iterable[float]:
        ...


@dataclass(frozen=True)
class IndexingReport:
    indexed: int = 0
    updated: int = 0
    unchanged: int = 0
    deduplicated: int = 0
    chunks_indexed: int = 0
    stale_revisions: int = 0

    @property
    def documents_seen(self) -> int:
        return self.indexed + self.updated + self.unchanged + self.deduplicated


@dataclass(frozen=True)
class HybridSearchResult:
    chunk: EvidenceChunk
    score: float
    bm25_rank: int | None
    vector_rank: int | None
    query_parts: tuple[str, ...]


def decompose_query(query: str, *, max_parts: int = 4) -> tuple[str, ...]:
    normalized = " ".join(query.split())
    if not normalized:
        return ()
    raw_parts = re.split(
        r"(?:[?？]|[;；\n]+|\b(?:and then|also|versus|vs\.?)\b|(?:以及|并且|然后|对比))",
        normalized,
        flags=re.IGNORECASE,
    )
    parts = tuple(
        dict.fromkeys(
            part.strip(" ,.:")
            for part in raw_parts
            if len(part.strip(" ,.:")) >= 3
        )
    )
    if len(parts) <= 1:
        return (normalized,)
    return parts[:max_parts]


class RagIndex:
    """SQLite-backed hybrid index with FTS5, hash vectors, and RRF."""

    def __init__(
        self,
        db_path: str | Path,
        embedder: Embedder | None = None,
    ) -> None:
        raw_db_path = str(db_path)
        self.db_path = (
            raw_db_path
            if raw_db_path == ":memory:"
            else str(Path(raw_db_path).expanduser().resolve())
        )
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or DeterministicHashEmbedder()
        self.chunker = StructuredChunker()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "RagIndex":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def index_documents(
        self,
        documents: Iterable[SourceDocument],
        *,
        namespace: str = "",
    ) -> IndexingReport:
        totals = {
            "indexed": 0,
            "updated": 0,
            "unchanged": 0,
            "deduplicated": 0,
            "chunks_indexed": 0,
            "stale_revisions": 0,
        }
        with self._lock, self._connection:
            for document in documents:
                scoped = _scope_document(document, namespace)
                result, chunk_count, stale_revision = self._index_document(
                    scoped,
                    namespace=namespace,
                )
                totals[result] += 1
                totals["chunks_indexed"] += chunk_count
                totals["stale_revisions"] += int(stale_revision)
        return IndexingReport(**totals)

    def sync_documents(
        self,
        documents: Iterable[SourceDocument],
        *,
        source_type: str | None = None,
        namespace: str = "",
    ) -> IndexingReport:
        values = tuple(documents)
        scoped_values = tuple(
            _scope_document(document, namespace) for document in values
        )
        report = self.index_documents(values, namespace=namespace)
        active_ids = {document.document_id for document in scoped_values}
        with self._lock, self._connection:
            if active_ids:
                placeholders = ",".join("?" for _ in active_ids)
                alias_rows = self._connection.execute(
                    "SELECT canonical_document_id FROM document_aliases "
                    f"WHERE alias_document_id IN ({placeholders})",
                    tuple(active_ids),
                ).fetchall()
                active_ids.update(
                    str(row["canonical_document_id"])
                    for row in alias_rows
                )
            if source_type is None:
                rows = self._connection.execute(
                    "SELECT document_id FROM documents "
                    "WHERE stale = 0 AND namespace = ?",
                    (namespace,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT document_id FROM documents "
                    "WHERE stale = 0 AND source_type = ? AND namespace = ?",
                    (source_type, namespace),
                ).fetchall()
            for row in rows:
                if row["document_id"] not in active_ids:
                    self._mark_document_stale(row["document_id"])
        return report

    def search(
        self,
        query: str,
        top_k: int = 8,
        *,
        include_stale: bool = False,
        namespace: str = "",
    ) -> list[EvidenceChunk]:
        return [
            result.chunk
            for result in self.search_with_scores(
                query,
                top_k=top_k,
                include_stale=include_stale,
                namespace=namespace,
            )
        ]

    def search_with_scores(
        self,
        query: str,
        top_k: int = 8,
        *,
        include_stale: bool = False,
        namespace: str = "",
    ) -> list[HybridSearchResult]:
        parts = decompose_query(query)
        if not parts or top_k <= 0:
            return []
        limit = min(int(top_k), 8)
        candidate_limit = 20
        rrf_constant = 60.0
        scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}

        with self._lock:
            for part in parts:
                lexical_ids = self._lexical_search(
                    part,
                    candidate_limit,
                    include_stale=include_stale,
                    namespace=namespace,
                )
                for rank, chunk_id in enumerate(lexical_ids, start=1):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                        1.0 / (rrf_constant + rank)
                    )
                    bm25_ranks[chunk_id] = min(
                        bm25_ranks.get(chunk_id, rank),
                        rank,
                    )
                vector_ids = self._vector_search(
                    part,
                    candidate_limit,
                    include_stale=include_stale,
                    namespace=namespace,
                )
                for rank, (chunk_id, _similarity) in enumerate(vector_ids, start=1):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                        1.0 / (rrf_constant + rank)
                    )
                    vector_ranks[chunk_id] = min(
                        vector_ranks.get(chunk_id, rank),
                        rank,
                    )

            candidate_rows: dict[str, sqlite3.Row] = {}
            rerank_scores: dict[str, float] = {}
            rerank_metadata: dict[str, dict[str, Any]] = {}
            query_terms = {
                token
                for token in tokenize(query)
                if not token.startswith("#") and token not in _RERANK_STOPWORDS
            }
            for chunk_id, rrf_score in scores.items():
                row = self._chunk_row(
                    chunk_id,
                    include_stale=include_stale,
                    namespace=namespace,
                )
                if row is None:
                    continue
                candidate_rows[chunk_id] = row
                text_terms = {
                    token
                    for token in tokenize(f"{row['title']} {row['text']}")
                    if not token.startswith("#") and token not in _RERANK_STOPWORDS
                }
                lexical_coverage = (
                    len(query_terms & text_terms) / len(query_terms)
                    if query_terms
                    else 0.0
                )
                metadata = _load_json(row["metadata_json"], {})
                poisoning_penalty = (
                    0.025 if metadata.get("prompt_injection_detected") else 0.0
                )
                rerank_scores[chunk_id] = (
                    rrf_score + 0.01 * lexical_coverage - poisoning_penalty
                )
                rerank_metadata[chunk_id] = {
                    "rrf_score": rrf_score,
                    "lexical_coverage": lexical_coverage,
                    "poisoning_penalty": poisoning_penalty,
                }
            ranked_ids = sorted(
                candidate_rows,
                key=lambda chunk_id: (-rerank_scores[chunk_id], chunk_id),
            )
            output: list[HybridSearchResult] = []
            seen_content_hashes: set[str] = set()
            seen_documents: set[str] = set()
            for chunk_id in ranked_ids:
                row = candidate_rows[chunk_id]
                content_hash = str(row["content_hash"])
                if content_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(content_hash)
                document_id = str(row["document_id"])
                diversity_penalty = 0.002 if document_id in seen_documents else 0.0
                seen_documents.add(document_id)
                score = rerank_scores[chunk_id] - diversity_penalty
                chunk = self._chunk_from_row(
                    row,
                    score=score,
                    retrieval_metadata={
                        "method": "bm25_vector_rrf",
                        "bm25_rank": bm25_ranks.get(chunk_id),
                        "vector_rank": vector_ranks.get(chunk_id),
                        "query_parts": list(parts),
                        "reranker": "structured_v1",
                        **rerank_metadata[chunk_id],
                        "diversity_penalty": diversity_penalty,
                    },
                )
                output.append(
                    HybridSearchResult(
                        chunk=chunk,
                        score=score,
                        bm25_rank=bm25_ranks.get(chunk_id),
                        vector_rank=vector_ranks.get(chunk_id),
                        query_parts=parts,
                    )
                )
                if len(output) >= limit:
                    break
        return output

    def mark_source_stale(self, source_id: str, *, namespace: str = "") -> int:
        scoped_source_id = _scoped_document_id(source_id, namespace)
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT document_id FROM documents "
                "WHERE namespace = ? AND (document_id = ? OR uri = ?)",
                (namespace, scoped_source_id, source_id),
            ).fetchall()
            alias_rows = self._connection.execute(
                "SELECT canonical_document_id FROM document_aliases "
                "WHERE namespace = ? AND (alias_document_id = ? OR uri = ?)",
                (namespace, scoped_source_id, source_id),
            ).fetchall()
            document_ids = {
                str(row["document_id"])
                for row in rows
            } | {
                str(row["canonical_document_id"])
                for row in alias_rows
            }
            for document_id in document_ids:
                self._mark_document_stale(document_id)
            self._connection.execute(
                "UPDATE document_aliases SET stale = 1 "
                "WHERE namespace = ? AND (alias_document_id = ? OR uri = ?)",
                (namespace, scoped_source_id, source_id),
            )
            return len(document_ids)

    def get_document(
        self,
        document_id: str,
        *,
        include_stale: bool = True,
        namespace: str = "",
    ) -> SourceDocument | None:
        scoped_document_id = _scoped_document_id(document_id, namespace)
        with self._lock:
            sql = "SELECT * FROM documents WHERE document_id = ? AND namespace = ?"
            params: tuple[Any, ...] = (scoped_document_id, namespace)
            if not include_stale:
                sql += " AND stale = 0"
            row = self._connection.execute(sql, params).fetchone()
            if row is None:
                return None
            return self._document_from_row(row)

    def list_documents(
        self,
        *,
        include_stale: bool = False,
        namespace: str = "",
    ) -> list[SourceDocument]:
        with self._lock:
            sql = "SELECT * FROM documents WHERE namespace = ?"
            parameters: tuple[Any, ...] = (namespace,)
            if not include_stale:
                sql += " AND stale = 0"
            sql += " ORDER BY document_id"
            return [
                self._document_from_row(row)
                for row in self._connection.execute(sql, parameters).fetchall()
            ]

    def list_revisions(
        self,
        document_id: str,
        *,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        scoped_document_id = _scoped_document_id(document_id, namespace)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM document_revisions "
                "WHERE document_id = ? AND namespace = ? ORDER BY revision_id",
                (scoped_document_id, namespace),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "metadata": _load_json(row["metadata_json"], {}),
                }
                for row in rows
            ]

    def stats(self, *, namespace: str | None = None) -> dict[str, int]:
        document_scope = "" if namespace is None else " WHERE namespace = ?"
        active_document_scope = (
            " WHERE stale = 0"
            if namespace is None
            else " WHERE stale = 0 AND namespace = ?"
        )
        chunk_scope = "" if namespace is None else " WHERE namespace = ?"
        active_chunk_scope = (
            " WHERE stale = 0"
            if namespace is None
            else " WHERE stale = 0 AND namespace = ?"
        )
        parameters: tuple[Any, ...] = () if namespace is None else (namespace,)
        with self._lock:
            return {
                "documents": int(
                    self._connection.execute(
                        f"SELECT COUNT(*) FROM documents{document_scope}",
                        parameters,
                    ).fetchone()[0]
                ),
                "active_documents": int(
                    self._connection.execute(
                        f"SELECT COUNT(*) FROM documents{active_document_scope}",
                        parameters,
                    ).fetchone()[0]
                ),
                "chunks": int(
                    self._connection.execute(
                        f"SELECT COUNT(*) FROM chunks{chunk_scope}",
                        parameters,
                    ).fetchone()[0]
                ),
                "active_chunks": int(
                    self._connection.execute(
                        f"SELECT COUNT(*) FROM chunks{active_chunk_scope}",
                        parameters,
                    ).fetchone()[0]
                ),
                "aliases": int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM document_aliases"
                        + (
                            ""
                            if namespace is None
                            else " WHERE namespace = ?"
                        ),
                        parameters,
                    ).fetchone()[0]
                ),
                "stale_revisions": int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM document_revisions"
                        + (
                            ""
                            if namespace is None
                            else " WHERE namespace = ?"
                        ),
                        parameters,
                    ).fetchone()[0]
                ),
            }

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            try:
                self._connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL DEFAULT '',
                        source_type TEXT NOT NULL,
                        uri TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL UNIQUE,
                        version TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        stale INTEGER NOT NULL DEFAULT 0,
                        indexed_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_documents_uri
                        ON documents(uri);
                    CREATE INDEX IF NOT EXISTS idx_documents_stale
                        ON documents(stale);

                    CREATE TABLE IF NOT EXISTS document_aliases (
                        alias_document_id TEXT PRIMARY KEY,
                        canonical_document_id TEXT NOT NULL,
                        namespace TEXT NOT NULL DEFAULT '',
                        uri TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        stale INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS document_revisions (
                        revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        namespace TEXT NOT NULL DEFAULT '',
                        source_type TEXT NOT NULL,
                        uri TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        version TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        stale INTEGER NOT NULL DEFAULT 1,
                        superseded_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_revisions_document
                        ON document_revisions(document_id);

                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        namespace TEXT NOT NULL DEFAULT '',
                        text TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        source_uri TEXT NOT NULL,
                        title TEXT NOT NULL,
                        start_line INTEGER,
                        end_line INTEGER,
                        page INTEGER,
                        score REAL NOT NULL DEFAULT 0.0,
                        metadata_json TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        stale INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY(document_id) REFERENCES documents(document_id)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_chunks_document
                        ON chunks(document_id);
                    CREATE INDEX IF NOT EXISTS idx_chunks_stale
                        ON chunks(stale);

                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        title,
                        text,
                        tokenize = 'unicode61'
                    );
                    """
                )
            except sqlite3.OperationalError as exc:
                raise RuntimeError("SQLite FTS5 support is required") from exc
            self._ensure_namespace_columns()
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_namespace "
                "ON documents(namespace)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_namespace "
                "ON chunks(namespace)"
            )
            self._connection.commit()

    def _index_document(
        self,
        document: SourceDocument,
        *,
        namespace: str,
    ) -> tuple[str, int, bool]:
        if not document.document_id:
            raise ValueError("document_id must not be empty")
        if not document.uri:
            raise ValueError("document uri must not be empty")
        if not document.content.strip():
            raise ValueError(f"document {document.document_id} has no content")
        actual_content_hash = sha256_text(document.content)
        expected_content_hash = str(
            document.metadata.get(
                "_tracepilot_original_content_hash",
                document.content_hash,
            )
        )
        if (
            expected_content_hash
            and expected_content_hash.casefold() != actual_content_hash
        ):
            raise ValueError(
                f"content_hash mismatch for document {document.document_id}"
            )
        content_hash = document.content_hash or actual_content_hash
        existing = self._connection.execute(
            "SELECT * FROM documents WHERE document_id = ? AND namespace = ?",
            (document.document_id, namespace),
        ).fetchone()
        duplicate = self._connection.execute(
            "SELECT document_id FROM documents "
            "WHERE content_hash = ? AND document_id <> ? AND namespace = ?",
            (content_hash, document.document_id, namespace),
        ).fetchone()
        if duplicate is not None:
            stale_revision = False
            if existing is not None:
                self._archive_revision(existing)
                self._delete_chunks(document.document_id)
                self._connection.execute(
                    "DELETE FROM documents WHERE document_id = ?",
                    (document.document_id,),
                )
                stale_revision = True
            self._connection.execute(
                """
                INSERT INTO document_aliases(
                    alias_document_id, canonical_document_id, namespace, uri, title,
                    metadata_json, stale
                ) VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(alias_document_id) DO UPDATE SET
                    canonical_document_id = excluded.canonical_document_id,
                    namespace = excluded.namespace,
                    uri = excluded.uri,
                    title = excluded.title,
                    metadata_json = excluded.metadata_json,
                    stale = 0
                """,
                (
                    document.document_id,
                    duplicate["document_id"],
                    namespace,
                    document.uri,
                    document.title,
                    _dump_json(document.metadata),
                ),
            )
            return "deduplicated", 0, stale_revision

        now = _now()
        if existing is not None and existing["content_hash"] == content_hash:
            self._connection.execute(
                """
                UPDATE documents
                SET source_type = ?, uri = ?, title = ?, content = ?,
                    version = ?, metadata_json = ?, stale = 0, indexed_at = ?
                WHERE document_id = ?
                """,
                (
                    document.source_type,
                    document.uri,
                    document.title,
                    document.content,
                    document.version,
                    _dump_json(document.metadata),
                    now,
                    document.document_id,
                ),
            )
            self._connection.execute(
                "UPDATE chunks SET stale = 0 WHERE document_id = ?",
                (document.document_id,),
            )
            self._connection.execute(
                "UPDATE document_aliases SET stale = 0 "
                "WHERE canonical_document_id = ?",
                (document.document_id,),
            )
            chunk_rows = self._connection.execute(
                "SELECT chunk_id, text, metadata_json FROM chunks WHERE document_id = ?",
                (document.document_id,),
            ).fetchall()
            for chunk_row in chunk_rows:
                chunk_metadata = _load_json(chunk_row["metadata_json"], {})
                chunk_metadata["document_version"] = document.version
                self._connection.execute(
                    "UPDATE chunks SET source_uri = ?, title = ?, metadata_json = ? "
                    "WHERE chunk_id = ?",
                    (
                        document.uri,
                        document.title,
                        _dump_json(chunk_metadata),
                        chunk_row["chunk_id"],
                    ),
                )
                self._connection.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    (chunk_row["chunk_id"],),
                )
                self._connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, title, text) VALUES (?, ?, ?)",
                    (chunk_row["chunk_id"], document.title, chunk_row["text"]),
                )
            return "unchanged", 0, False

        result = "indexed"
        stale_revision = False
        if existing is not None:
            result = "updated"
            stale_revision = True
            self._archive_revision(existing)
            self._connection.execute(
                "UPDATE document_aliases SET stale = 1 "
                "WHERE canonical_document_id = ?",
                (document.document_id,),
            )
            self._delete_chunks(document.document_id)
            self._connection.execute(
                """
                UPDATE documents
                SET source_type = ?, uri = ?, title = ?, content = ?,
                    content_hash = ?, version = ?, metadata_json = ?,
                    stale = 0, indexed_at = ?
                WHERE document_id = ?
                """,
                (
                    document.source_type,
                    document.uri,
                    document.title,
                    document.content,
                    content_hash,
                    document.version,
                    _dump_json(document.metadata),
                    now,
                    document.document_id,
                ),
            )
        else:
            self._connection.execute(
                """
                INSERT INTO documents(
                    document_id, namespace, source_type, uri, title, content, content_hash,
                    version, metadata_json, stale, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    document.document_id,
                    namespace,
                    document.source_type,
                    document.uri,
                    document.title,
                    document.content,
                    content_hash,
                    document.version,
                    _dump_json(document.metadata),
                    now,
                ),
            )
        chunks = self.chunker.chunk_document(document)
        for chunk in chunks:
            embedding = tuple(float(value) for value in self.embedder.embed(chunk.text))
            self._connection.execute(
                """
                INSERT INTO chunks(
                    chunk_id, document_id, namespace, text, content_hash, source_uri,
                    title, start_line, end_line, page, score, metadata_json,
                    embedding_json, stale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    namespace,
                    chunk.text,
                    chunk.content_hash or sha256_text(chunk.text),
                    chunk.source_uri,
                    chunk.title,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.page,
                    float(chunk.score),
                    _dump_json(chunk.metadata),
                    _dump_json(embedding),
                ),
            )
            self._connection.execute(
                "INSERT INTO chunks_fts(chunk_id, title, text) VALUES (?, ?, ?)",
                (chunk.chunk_id, chunk.title, chunk.text),
            )
        self._connection.execute(
            "DELETE FROM document_aliases WHERE alias_document_id = ?",
            (document.document_id,),
        )
        return result, len(chunks), stale_revision

    def _archive_revision(self, row: sqlite3.Row) -> None:
        self._connection.execute(
            """
            INSERT INTO document_revisions(
                document_id, namespace, source_type, uri, title, content, content_hash,
                version, metadata_json, stale, superseded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                row["document_id"],
                row["namespace"],
                row["source_type"],
                row["uri"],
                row["title"],
                row["content"],
                row["content_hash"],
                row["version"],
                row["metadata_json"],
                _now(),
            ),
        )

    def _delete_chunks(self, document_id: str) -> None:
        rows = self._connection.execute(
            "SELECT chunk_id FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        for row in rows:
            self._connection.execute(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                (row["chunk_id"],),
            )
        self._connection.execute(
            "DELETE FROM chunks WHERE document_id = ?",
            (document_id,),
        )

    def _mark_document_stale(self, document_id: str) -> None:
        self._connection.execute(
            "UPDATE documents SET stale = 1 WHERE document_id = ?",
            (document_id,),
        )
        self._connection.execute(
            "UPDATE chunks SET stale = 1 WHERE document_id = ?",
            (document_id,),
        )
        self._connection.execute(
            "UPDATE document_aliases SET stale = 1 "
            "WHERE canonical_document_id = ?",
            (document_id,),
        )

    def _lexical_search(
        self,
        query: str,
        limit: int,
        *,
        include_stale: bool,
        namespace: str,
    ) -> list[str]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        rows = self._connection.execute(
            """
            SELECT chunks_fts.chunk_id AS chunk_id,
                   bm25(chunks_fts, 0.0, 1.0, 1.0) AS rank_score
            FROM chunks_fts
            JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
            JOIN documents ON documents.document_id = chunks.document_id
            WHERE chunks_fts MATCH ?
              AND documents.namespace = ?
              AND (? = 1 OR (chunks.stale = 0 AND documents.stale = 0))
            ORDER BY rank_score ASC, chunks_fts.chunk_id ASC
            LIMIT ?
            """,
            (fts_query, namespace, int(include_stale), limit),
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def _vector_search(
        self,
        query: str,
        limit: int,
        *,
        include_stale: bool,
        namespace: str,
    ) -> list[tuple[str, float]]:
        query_vector = tuple(float(value) for value in self.embedder.embed(query))
        rows = self._connection.execute(
            """
            SELECT chunks.chunk_id, chunks.embedding_json
            FROM chunks
            JOIN documents ON documents.document_id = chunks.document_id
            WHERE documents.namespace = ?
              AND (? = 1 OR (chunks.stale = 0 AND documents.stale = 0))
            """,
            (namespace, int(include_stale)),
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for row in rows:
            embedding = _load_json(row["embedding_json"], [])
            try:
                similarity = cosine_similarity(query_vector, embedding)
            except ValueError:
                continue
            if similarity > 0.0:
                scored.append((str(row["chunk_id"]), similarity))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    def _chunk_row(
        self,
        chunk_id: str,
        *,
        include_stale: bool,
        namespace: str,
    ) -> sqlite3.Row | None:
        return self._connection.execute(
            """
            SELECT chunks.*
            FROM chunks
            JOIN documents ON documents.document_id = chunks.document_id
            WHERE chunks.chunk_id = ?
              AND documents.namespace = ?
              AND (? = 1 OR (chunks.stale = 0 AND documents.stale = 0))
            """,
            (chunk_id, namespace, int(include_stale)),
        ).fetchone()

    def _ensure_namespace_columns(self) -> None:
        for table in ("documents", "document_aliases", "document_revisions", "chunks"):
            columns = {
                str(row["name"])
                for row in self._connection.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            if "namespace" not in columns:
                self._connection.execute(
                    f"ALTER TABLE {table} "
                    "ADD COLUMN namespace TEXT NOT NULL DEFAULT ''"
                )

    def _chunk_from_row(
        self,
        row: sqlite3.Row,
        *,
        score: float,
        retrieval_metadata: dict[str, Any],
    ) -> EvidenceChunk:
        metadata = _load_json(row["metadata_json"], {})
        metadata["retrieval"] = retrieval_metadata
        return EvidenceChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            text=str(row["text"]),
            content_hash=str(row["content_hash"]),
            source_uri=str(row["source_uri"]),
            title=str(row["title"]),
            start_line=row["start_line"],
            end_line=row["end_line"],
            page=row["page"],
            score=float(score),
            metadata=metadata,
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> SourceDocument:
        metadata = _load_json(row["metadata_json"], {})
        metadata["_index_stale"] = bool(row["stale"])
        return SourceDocument(
            document_id=str(
                metadata.get("_tracepilot_original_document_id", row["document_id"])
            ),
            source_type=str(row["source_type"]),
            uri=str(row["uri"]),
            title=str(row["title"]),
            content=str(row["content"]),
            content_hash=str(
                metadata.get("_tracepilot_original_content_hash", row["content_hash"])
            ),
            version=str(row["version"]),
            metadata=metadata,
        )


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
    unique = tuple(dict.fromkeys(token for token in tokens if token))
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique)


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_json(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scoped_document_id(document_id: str, namespace: str) -> str:
    if not namespace:
        return document_id
    digest = hashlib.sha256(
        f"{namespace}\0{document_id}".encode("utf-8")
    ).hexdigest()[:28]
    return f"doc-scope-{digest}"


def _scope_document(document: SourceDocument, namespace: str) -> SourceDocument:
    if not namespace:
        return document
    original_hash = document.content_hash or sha256_text(document.content)
    scoped_hash = hashlib.sha256(
        f"{namespace}\0{original_hash}".encode("utf-8")
    ).hexdigest()
    return replace(
        document,
        document_id=_scoped_document_id(document.document_id, namespace),
        content_hash=scoped_hash,
        metadata={
            **document.metadata,
            "_tracepilot_namespace": namespace,
            "_tracepilot_original_document_id": document.document_id,
            "_tracepilot_original_content_hash": original_hash,
        },
    )
