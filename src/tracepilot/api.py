from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .agent import TERMINAL_STATUSES, TracePilotAgent, workspace_id_for_path
from .memory import MemoryStore
from .models import AgentRun, EvidenceChunk, EvidenceClaim, TaskContext, TaskRequest
from .reporting import render_markdown_report, report_payload
from .serialization import run_from_dict, to_jsonable
from .workbench import render_workbench


CATALOG_SCOPE = "__tracepilot_catalog__"


def _citation_view(
    claim: EvidenceClaim,
    chunks: Mapping[str, EvidenceChunk],
) -> dict[str, Any]:
    cited = next(
        (chunks[item] for item in claim.evidence_ids if item in chunks),
        None,
    )
    if cited is None:
        return {"source_id": "", "source_uri": "", "location": ""}
    if cited.page is not None:
        location = f"page {cited.page}"
    elif cited.start_line is not None and cited.end_line is not None:
        location = f"lines {cited.start_line}-{cited.end_line}"
    else:
        location = ""
    return {
        "source_id": cited.chunk_id,
        "source_uri": cited.source_uri,
        "source_title": cited.title,
        "location": location,
    }


class TaskService:
    def __init__(
        self,
        db_path: str | Path,
        *,
        frozen_sources: str | Path | None = None,
        default_workspace_id: str = "",
    ) -> None:
        self.db_path = str(db_path)
        self.frozen_sources = str(frozen_sources) if frozen_sources else None
        self.default_workspace_id = default_workspace_id
        self.store = MemoryStore(self.db_path)
        saved = self.store.get_working_memory(CATALOG_SCOPE, "workspace_ids") or []
        self._workspace_ids = set(str(item) for item in saved)
        if default_workspace_id:
            self._workspace_ids.add(default_workspace_id)
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        self.store.close()

    def submit(self, payload: Mapping[str, Any]) -> AgentRun:
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            raise ValueError("goal is required")
        workspace = Path(str(payload.get("workspace_path", "."))).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace_path is not a directory: {workspace}")
        workspace_id = str(payload.get("workspace_id", "")).strip()
        if not workspace_id or workspace_id == "default":
            workspace_id = self.default_workspace_id or workspace_id_for_path(workspace)
        self._bind_workspace(workspace_id, workspace)
        request = TaskRequest(
            goal=goal,
            workspace_id=workspace_id,
            workspace_path=str(workspace),
            source_urls=tuple(str(item) for item in payload.get("source_urls", [])),
            constraints=tuple(str(item) for item in payload.get("constraints", [])),
            execution_policy=str(
                payload.get("execution_policy", "approval_required")
            ),
            metadata=dict(payload.get("metadata", {})),
        )
        placeholder = AgentRun(
            task_id=request.task_id,
            request=request,
            context=TaskContext(
                task_id=request.task_id,
                workspace_id=workspace_id,
                workspace_path=str(workspace),
            ),
        )
        self._remember_workspace(workspace_id)
        self.store.save_task(workspace_id, placeholder, task_id=request.task_id)
        frozen = payload.get("frozen_sources", self.frozen_sources)
        thread = threading.Thread(
            target=self._run_background,
            args=(request, frozen),
            name=f"tracepilot-{request.task_id[:12]}",
            daemon=True,
        )
        with self._lock:
            self._threads[request.task_id] = thread
        thread.start()
        return placeholder

    def wait(self, task_id: str, timeout: float = 10) -> AgentRun | None:
        with self._lock:
            thread = self._threads.get(task_id)
        if thread is not None:
            thread.join(timeout=timeout)
        return self.find_task(task_id)

    def list_tasks(self, workspace_id: str = "") -> list[AgentRun]:
        workspace_ids = [workspace_id] if workspace_id else sorted(self._workspace_ids)
        runs: list[AgentRun] = []
        for identifier in workspace_ids:
            runs.extend(
                run_from_dict(entry.payload)
                for entry in self.store.list_tasks(identifier)
            )
        return sorted(runs, key=lambda run: run.updated_at, reverse=True)

    def find_task(self, task_id: str, workspace_id: str = "") -> AgentRun | None:
        workspace_ids = [workspace_id] if workspace_id else sorted(self._workspace_ids)
        for identifier in workspace_ids:
            entry = self.store.get_task(identifier, task_id)
            if entry is not None:
                return run_from_dict(entry.payload)
        return None

    def resume(self, task_id: str, workspace_id: str = "") -> AgentRun:
        run = self._require_task(task_id, workspace_id)
        with TracePilotAgent(self.db_path) as agent:
            return agent.resume(run.request.workspace_id, task_id)

    def decide(
        self,
        task_id: str,
        approval_id: str,
        payload: Mapping[str, Any],
        workspace_id: str = "",
    ) -> AgentRun:
        run = self._require_task(task_id, workspace_id)
        with TracePilotAgent(self.db_path) as agent:
            return agent.decide_approval(
                run.request.workspace_id,
                task_id,
                approval_id,
                decision=str(payload.get("decision", "approved")),
                arguments=payload.get("arguments"),
                comment=str(payload.get("comment", "")),
                decided_by=str(payload.get("decided_by", "workbench")),
            )

    def promote(self, task_id: str, workspace_id: str = "") -> Any:
        run = self._require_task(task_id, workspace_id)
        with TracePilotAgent(self.db_path) as agent:
            return agent.promote_decision(run.request.workspace_id, task_id)

    def memories(self, workspace_id: str, *, include_all: bool = False) -> list[Any]:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        return list(
            self.store.list_decision_memories(
                workspace_id,
                status=None if include_all else "active",
                include_deleted=include_all,
            )
        )

    def delete_memory(self, memory_id: str, workspace_id: str = "") -> Any:
        identifier = workspace_id or self._workspace_for_memory(memory_id)
        return self.store.delete_decision_memory(identifier, memory_id)

    def supersede_memory(
        self,
        memory_id: str,
        payload: Mapping[str, Any],
        workspace_id: str = "",
    ) -> Any:
        identifier = workspace_id or self._workspace_for_memory(memory_id)
        source_task_id = str(payload.get("task_id", "")).strip()
        source_run = (
            self._require_task(source_task_id, identifier)
            if source_task_id
            else None
        )
        if source_run is not None and source_run.status.value != "completed":
            raise ValueError(
                "memory supersede source task must be completed"
            )
        replacement = {
            "workspace_id": identifier,
            "kind": "decision",
            "content": payload.get("content", payload),
            "source_run_id": source_run.run_id if source_run else "",
            "source_task_id": source_run.task_id if source_run else "",
            "source_versions": (
                source_run.context.source_versions if source_run else {}
            ),
        }
        return self.store.supersede_decision_memory(
            identifier,
            memory_id,
            replacement,
            source=(
                source_run.context.source_versions
                if source_run is not None
                else None
            ),
            workspace_state=(
                {
                    "workspace_revision": source_run.context.workspace_revision,
                    "workspace_hash": source_run.context.workspace_hash,
                    "source_versions": source_run.context.source_versions,
                }
                if source_run is not None
                else None
            ),
        )

    def events(
        self,
        task_id: str,
        workspace_id: str = "",
    ) -> Iterable[dict[str, Any]]:
        run = self._require_task(task_id, workspace_id)
        entries = self.store.list_events(
            run.request.workspace_id,
            task_id=task_id,
        )
        return [to_jsonable(entry.payload) for entry in entries]

    def task_view(self, run: AgentRun) -> dict[str, Any]:
        payload = to_jsonable(run)
        payload["goal"] = run.request.goal
        payload["workspace_id"] = run.request.workspace_id
        payload["workspace_path"] = run.request.workspace_path
        payload["status"] = run.status.value
        payload["plan"]["steps"] = [
            {
                **to_jsonable(step),
                "status": step.metadata.get("status", "planned"),
                "error": step.metadata.get("error", ""),
            }
            for step in run.plan.steps
        ]
        payload["tool_results"] = [
            {**to_jsonable(result), "ok": result.success}
            for result in run.tool_results
        ]
        chunks = {
            item.chunk_id: item
            for item in run.evidence
            if isinstance(item, EvidenceChunk)
        }
        claims = [
            item for item in run.evidence if isinstance(item, EvidenceClaim)
        ]
        payload["evidence_chunks"] = [to_jsonable(item) for item in chunks.values()]
        payload["evidence"] = [
            {
                **to_jsonable(claim),
                **_citation_view(claim, chunks),
            }
            for claim in claims
        ]
        payload["memories"] = [
            to_jsonable(entry)
            for entry in self.store.list_decision_memories(
                run.request.workspace_id,
                status=None,
                include_deleted=False,
            )
        ]
        return payload

    def _run_background(self, request: TaskRequest, frozen_sources: Any) -> None:
        try:
            with TracePilotAgent(self.db_path) as agent:
                agent.run(request, frozen_sources=frozen_sources)
        finally:
            with self._lock:
                self._threads.pop(request.task_id, None)

    def _remember_workspace(self, workspace_id: str) -> None:
        self._workspace_ids.add(workspace_id)
        self.store.put_working_memory(
            CATALOG_SCOPE,
            "workspace_ids",
            sorted(self._workspace_ids),
        )

    def _bind_workspace(self, workspace_id: str, workspace: Path) -> None:
        key = "workspace_binding"
        resolved = str(workspace)
        existing = self.store.get_working_memory(workspace_id, key)
        if existing is not None:
            bound_path = str(existing.get("workspace_path", ""))
            if bound_path and bound_path != resolved:
                raise ValueError(
                    f"workspace_id {workspace_id!r} is already bound to "
                    "a different workspace path"
                )
            return
        self.store.put_working_memory(
            workspace_id,
            key,
            {
                "workspace_id": workspace_id,
                "workspace_path": resolved,
            },
        )

    def _require_task(self, task_id: str, workspace_id: str = "") -> AgentRun:
        run = self.find_task(task_id, workspace_id)
        if run is None:
            raise KeyError(f"unknown task: {task_id}")
        return run

    def _workspace_for_memory(self, memory_id: str) -> str:
        for workspace_id in sorted(self._workspace_ids):
            if self.store.get_decision_memory(workspace_id, memory_id) is not None:
                return workspace_id
        raise KeyError(f"unknown memory: {memory_id}")


def create_app(
    db_path: str | Path = ".cache/tracepilot/workbench.db",
    *,
    frozen_sources: str | Path | None = None,
    default_workspace_id: str = "",
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "FastAPI support requires the 'api' extra: pip install -e '.[api]'"
        ) from exc

    app = FastAPI(
        title="TracePilot API",
        version="0.2.0",
        description="Local technical research and approval-gated execution API.",
    )
    service = TaskService(
        db_path,
        frozen_sources=frozen_sources,
        default_workspace_id=default_workspace_id,
    )
    app.state.tracepilot = service

    @app.get("/", response_class=HTMLResponse)
    def workbench() -> str:
        return render_workbench()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "schema_version": 2}

    @app.get("/v1/tasks")
    def list_tasks(workspace_id: str = "") -> dict[str, Any]:
        return {
            "tasks": [
                {
                    "task_id": run.task_id,
                    "goal": run.request.goal,
                    "workspace_id": run.request.workspace_id,
                    "workspace_path": run.request.workspace_path,
                    "status": run.status.value,
                    "updated_at": run.updated_at,
                }
                for run in service.list_tasks(workspace_id)
            ]
        }

    @app.post("/v1/tasks", status_code=202)
    def create_task(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.task_view(service.submit(payload))
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/tasks/{task_id}")
    def get_task(task_id: str, workspace_id: str = "") -> dict[str, Any]:
        run = service.find_task(task_id, workspace_id)
        if run is None:
            raise HTTPException(status_code=404, detail="task not found")
        return service.task_view(run)

    @app.post("/v1/tasks/{task_id}/resume")
    def resume_task(task_id: str, workspace_id: str = "") -> dict[str, Any]:
        try:
            return service.task_view(service.resume(task_id, workspace_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/tasks/{task_id}/approvals/{approval_id}")
    def decide_approval(
        task_id: str,
        approval_id: str,
        payload: dict[str, Any],
        workspace_id: str = "",
    ) -> dict[str, Any]:
        try:
            run = service.decide(
                task_id,
                approval_id,
                payload,
                workspace_id,
            )
            return service.task_view(run)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/tasks/{task_id}/events")
    def task_events(task_id: str, workspace_id: str = "") -> Any:
        try:
            run = service._require_task(task_id, workspace_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        def stream() -> Iterable[str]:
            sent: set[str] = set()
            idle_after_terminal = 0
            while True:
                events = list(service.events(task_id, run.request.workspace_id))
                emitted = False
                for event in events:
                    event_id = str(event.get("event_id", ""))
                    if event_id in sent:
                        continue
                    sent.add(event_id)
                    emitted = True
                    yield (
                        f"id: {event_id}\n"
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    )
                current = service.find_task(task_id, run.request.workspace_id)
                terminal = current is not None and current.status in TERMINAL_STATUSES
                if terminal and not emitted:
                    idle_after_terminal += 1
                else:
                    idle_after_terminal = 0
                if idle_after_terminal >= 2:
                    break
                if not emitted:
                    yield ": keepalive\n\n"
                time.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/tasks/{task_id}/artifacts")
    def task_artifacts(task_id: str, workspace_id: str = "") -> dict[str, Any]:
        run = service.find_task(task_id, workspace_id)
        if run is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "report_markdown": render_markdown_report(run),
            "report": report_payload(run),
        }

    @app.get("/v1/memory")
    def list_memory(
        workspace_id: str = Query(min_length=1),
        include_all: bool = False,
    ) -> dict[str, Any]:
        return {
            "memories": [
                to_jsonable(entry)
                for entry in service.memories(
                    workspace_id,
                    include_all=include_all,
                )
            ]
        }

    @app.post("/v1/memory/promote")
    def promote_memory(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            entry = service.promote(
                str(payload.get("task_id", "")),
                str(payload.get("workspace_id", "")),
            )
            return to_jsonable(entry)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/memory/{memory_id}/supersede")
    def supersede_memory(
        memory_id: str,
        payload: dict[str, Any],
        workspace_id: str = "",
    ) -> dict[str, Any]:
        try:
            return to_jsonable(
                service.supersede_memory(memory_id, payload, workspace_id)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/v1/memory/{memory_id}")
    def delete_memory(memory_id: str, workspace_id: str = "") -> dict[str, Any]:
        try:
            return to_jsonable(service.delete_memory(memory_id, workspace_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.exception_handler(Exception)
    async def unhandled_error(_request: Any, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}"},
        )

    return app


def serve(
    *,
    db_path: str | Path,
    host: str,
    port: int,
    frozen_sources: str | Path | None = None,
    default_workspace_id: str = "",
) -> None:
    try:
        import uvicorn
        app = create_app(
            db_path,
            frozen_sources=frozen_sources,
            default_workspace_id=default_workspace_id,
        )
    except (ImportError, RuntimeError):  # pragma: no cover - environment dependent
        from .preview_server import serve_preview

        serve_preview(
            db_path=db_path,
            host=host,
            port=port,
            frozen_sources=frozen_sources,
            default_workspace_id=default_workspace_id,
        )
        return
    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the TracePilot workbench.")
    parser.add_argument("--db", default=".cache/tracepilot/workbench.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--frozen-sources")
    parser.add_argument("--workspace-id", default="")
    args = parser.parse_args(argv)
    serve(
        db_path=args.db,
        host=args.host,
        port=args.port,
        frozen_sources=args.frozen_sources,
        default_workspace_id=args.workspace_id,
    )


if __name__ == "__main__":
    main()
