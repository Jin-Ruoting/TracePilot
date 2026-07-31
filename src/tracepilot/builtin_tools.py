from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from .models import SourceDocument
from .rag import Ingestor, RagIndex
from .serialization import to_jsonable
from .tooling import EffectClass, ToolContract, ToolRegistry


class WebSearchProvider(Protocol):
    def search(self, query: str, top_k: int = 10) -> Iterable[SourceDocument]:
        ...


class DisabledWebSearchProvider:
    """Explicit offline provider used unless a live provider is configured."""

    def search(self, query: str, top_k: int = 10) -> Iterable[SourceDocument]:
        del query, top_k
        return ()


def build_default_registry(
    *,
    workspace_root: str | Path,
    index: RagIndex,
    workspace_id: str = "",
    web_provider: WebSearchProvider | None = None,
) -> ToolRegistry:
    root = Path(workspace_root).expanduser().resolve()
    active_namespace = workspace_id
    ingestor = Ingestor()
    provider = web_provider or DisabledWebSearchProvider()
    registry = ToolRegistry()

    def index_workspace(workspace_path: str, workspace_id: str) -> dict[str, Any]:
        nonlocal active_namespace
        if active_namespace and workspace_id and workspace_id != active_namespace:
            raise ValueError("workspace_id does not match the active tool registry")
        active_namespace = active_namespace or workspace_id
        target = _resolve_path(root, workspace_path, allow_root=True)
        documents = ingestor.ingest_path(target)
        report = index.sync_documents(
            documents,
            source_type="local",
            namespace=active_namespace,
        )
        return {
            "report": asdict(report),
            "documents": len(documents),
            "source_versions": {
                document.uri: document.version or document.content_hash
                for document in documents
            },
        }

    def search_index(query: str, top_k: int, failed_steps: list[str] | None = None) -> list[dict[str, Any]]:
        del failed_steps
        return [
            to_jsonable(chunk)
            for chunk in index.search(
                query,
                top_k=top_k,
                namespace=active_namespace,
            )
        ]

    def web_search(query: str, top_k: int) -> list[dict[str, Any]]:
        live_documents = list(provider.search(query, top_k=top_k))
        if live_documents:
            index.index_documents(live_documents, namespace=active_namespace)
        chunks = index.search(
            query,
            top_k=min(top_k, 8),
            namespace=active_namespace,
        )
        return [
            to_jsonable(chunk)
            for chunk in chunks
            if not chunk.source_uri.startswith("file:")
        ]

    def fetch_url(url: str) -> dict[str, Any]:
        documents = ingestor.ingest_url(url)
        report = index.index_documents(documents, namespace=active_namespace)
        return {
            "report": asdict(report),
            "documents": [to_jsonable(document) for document in documents],
        }

    def read_file(path: str, start_line: int = 1, end_line: int = 240) -> dict[str, Any]:
        target = _resolve_path(root, path)
        if not target.is_file():
            raise FileNotFoundError(target)
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid line range")
        raw = target.read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError("file exceeds the 2 MB read limit")
        text = raw.decode("utf-8")
        lines = text.splitlines()
        selected = lines[start_line - 1 : end_line]
        return {
            "path": str(target),
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "content": "\n".join(selected),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def search_code(
        query: str,
        workspace_path: str,
        max_results: int = 40,
    ) -> list[dict[str, Any]]:
        target = _resolve_path(root, workspace_path, allow_root=True)
        terms = _query_terms(query)
        if not terms:
            return []
        results: list[dict[str, Any]] = []
        for path in _iter_text_files(target):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                lowered = line.casefold()
                matched = [term for term in terms if term in lowered]
                if not matched:
                    continue
                results.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "text": line.strip()[:500],
                        "matched_terms": matched,
                    }
                )
                if len(results) >= max(1, min(max_results, 200)):
                    return results
        return results

    def inspect_git(workspace_path: str) -> dict[str, Any]:
        target = _resolve_path(root, workspace_path, allow_root=True)
        probe = _run_read_command(
            ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
            timeout=10,
        )
        if probe["returncode"] != 0:
            return {"is_repository": False, "workspace": str(target)}
        revision = _run_read_command(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            timeout=10,
        )
        status = _run_read_command(
            ["git", "-C", str(target), "status", "--short", "--branch"],
            timeout=10,
        )
        return {
            "is_repository": True,
            "root": probe["stdout"].strip(),
            "revision": revision["stdout"].strip(),
            "status": status["stdout"].splitlines(),
        }

    def propose_patch(
        goal: str,
        constraints: list[str],
        workspace_path: str,
    ) -> dict[str, Any]:
        target = _resolve_path(root, workspace_path, allow_root=True)
        return {
            "kind": "action_plan",
            "goal": goal,
            "workspace": str(target),
            "constraints": list(constraints),
            "proposed_changes": [],
            "execution_status": "not_requested",
            "note": (
                "This artifact contains no mutation. Supply an explicit command "
                "or structured file changes to request approval."
            ),
        }

    def apply_patch(
        workspace_path: str,
        changes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        target_root = _resolve_path(root, workspace_path, allow_root=True)
        validated: list[tuple[Path, str, str, bool]] = []
        seen_targets: set[Path] = set()
        for change in changes:
            if not isinstance(change, Mapping):
                raise ValueError("each change must be an object")
            relative_path = str(change.get("path", ""))
            if not relative_path:
                raise ValueError("change path is required")
            target = _resolve_path(target_root, relative_path)
            if target in seen_targets:
                raise ValueError(f"duplicate change path: {relative_path}")
            seen_targets.add(target)
            expected_value = change.get("expected_sha256")
            if not isinstance(expected_value, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}",
                expected_value,
            ):
                raise ValueError(
                    f"expected_sha256 is required for {relative_path}"
                )
            expected = expected_value.casefold()
            existed = target.exists()
            current = target.read_bytes() if existed else b""
            current_hash = hashlib.sha256(current).hexdigest()
            if expected != current_hash:
                raise ValueError(f"content hash mismatch for {relative_path}")
            content = change.get("content")
            if not isinstance(content, str):
                raise ValueError(f"change content must be text for {relative_path}")
            validated.append((target, content, current_hash, existed))
        for target, _content, previous_hash, existed in validated:
            if target.exists() != existed:
                raise ValueError(f"file existence changed before write: {target}")
            current = target.read_bytes() if target.exists() else b""
            if hashlib.sha256(current).hexdigest() != previous_hash:
                raise ValueError(f"content changed before write: {target}")
        written: list[dict[str, str]] = []
        for target, content, previous_hash, _existed in validated:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(
                {
                    "path": str(target),
                    "previous_sha256": previous_hash,
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
        return {"written": written}

    def run_command(
        workspace_path: str,
        argv: list[str],
        environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        target = _resolve_path(root, workspace_path, allow_root=True)
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must contain at least one non-empty string")
        env = os.environ.copy()
        for key, value in (environment or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid environment variable: {key}")
            env[key] = str(value)
        process = subprocess.Popen(
            argv,
            cwd=target,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired as exc:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - non-POSIX fallback
                process.kill()
            stdout, stderr = process.communicate()
            detail = (stderr or stdout or "").strip()
            suffix = f": {detail[-2_000:]}" if detail else ""
            raise RuntimeError(f"command exceeded 120s timeout{suffix}") from exc
        result = {
            "argv": argv,
            "returncode": process.returncode,
            "stdout": stdout[-20_000:],
            "stderr": stderr[-20_000:],
        }
        if process.returncode != 0:
            detail = result["stderr"].strip() or result["stdout"].strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"command exited with status {process.returncode}{suffix}"
            )
        return result

    _register(
        registry,
        "index_workspace",
        index_workspace,
        EffectClass.INTERNAL_WRITE,
        "Parse and index a workspace snapshot in the internal knowledge index.",
        permissions=("workspace_read", "internal_state"),
    )
    _register(
        registry,
        "search_index",
        search_index,
        EffectClass.READ,
        "Run hybrid BM25 and vector retrieval over the knowledge index.",
        permissions=("internal_state",),
    )
    _register(
        registry,
        "web_search",
        web_search,
        EffectClass.READ,
        "Search configured external sources and return evidence chunks.",
        permissions=("network", "internal_state"),
    )
    _register(
        registry,
        "fetch_url",
        fetch_url,
        EffectClass.READ,
        "Fetch, parse, and index an explicitly supplied URL.",
        permissions=("network", "internal_state"),
    )
    _register(
        registry,
        "read_file",
        read_file,
        EffectClass.READ,
        "Read a bounded line range from a file inside the workspace.",
        permissions=("workspace_read",),
    )
    _register(
        registry,
        "search_code",
        search_code,
        EffectClass.READ,
        "Search source files inside the workspace without invoking a shell.",
        permissions=("workspace_read",),
    )
    _register(
        registry,
        "inspect_git",
        inspect_git,
        EffectClass.READ,
        "Read the current Git revision and worktree status.",
        permissions=("workspace_read",),
    )
    _register(
        registry,
        "propose_patch",
        propose_patch,
        EffectClass.INTERNAL_WRITE,
        "Create a reviewable action artifact without modifying user files.",
        permissions=("internal_state",),
    )
    _register(
        registry,
        "apply_patch",
        apply_patch,
        EffectClass.MUTATION,
        "Apply structured, hash-checked file changes inside the workspace.",
        permissions=("workspace_write",),
        idempotent=False,
    )
    _register(
        registry,
        "run_command",
        run_command,
        EffectClass.MUTATION,
        "Run an exact argv vector inside the workspace after approval.",
        permissions=("subprocess",),
        idempotent=False,
        timeout_seconds=130,
    )
    return registry


def _register(
    registry: ToolRegistry,
    name: str,
    fn: Any,
    effect_class: EffectClass,
    description: str,
    *,
    permissions: tuple[str, ...],
    idempotent: bool = True,
    timeout_seconds: float = 30,
) -> None:
    registry.register(
        ToolContract(
            name=name,
            description=description,
            effect_class=effect_class,
            timeout_seconds=timeout_seconds,
            idempotent=idempotent,
            permissions=permissions,
        ),
        fn,
    )


def _resolve_path(root: Path, candidate: str | Path, *, allow_root: bool = False) -> Path:
    value = Path(candidate).expanduser()
    resolved = value.resolve() if value.is_absolute() else (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes workspace: {candidate}") from exc
    if not allow_root and resolved == root:
        raise PermissionError("operation requires a path below the workspace root")
    return resolved


def _iter_text_files(root: Path) -> Iterable[Path]:
    excluded = {".git", ".venv", "__pycache__", "node_modules", "dist", "build"}
    allowed = {
        ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html", ".java",
        ".js", ".json", ".md", ".py", ".rb", ".rs", ".sh", ".sql", ".toml",
        ".ts", ".tsx", ".txt", ".yaml", ".yml",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in allowed:
            continue
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts[:-1]):
            continue
        if path.stat().st_size <= 2_000_000:
            yield path


def _query_terms(query: str) -> tuple[str, ...]:
    stop = {
        "and", "are", "for", "how", "into", "should", "the", "this", "what",
        "when", "where", "which", "with", "your",
    }
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", query.casefold())
    return tuple(dict.fromkeys(term for term in terms if term not in stop))[:12]


def _run_read_command(argv: Sequence[str], *, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


__all__ = [
    "DisabledWebSearchProvider",
    "WebSearchProvider",
    "build_default_registry",
]
