from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .api import TaskService
from .reporting import render_markdown_report, report_payload
from .serialization import to_jsonable
from .workbench import render_workbench


def serve_preview(
    *,
    db_path: str | Path,
    host: str,
    port: int,
    frozen_sources: str | Path | None = None,
    default_workspace_id: str = "",
) -> None:
    service = TaskService(
        db_path,
        frozen_sources=frozen_sources,
        default_workspace_id=default_workspace_id,
    )

    class Handler(_PreviewHandler):
        task_service = service

    server = ThreadingHTTPServer((host, port), Handler)
    print(
        f"FastAPI dependencies are unavailable; serving the compatible local "
        f"preview at http://{host}:{port}"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


class _PreviewHandler(BaseHTTPRequestHandler):
    task_service: TaskService
    server_version = "TracePilotPreview/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        workspace_id = query.get("workspace_id", [""])[0]
        path = parsed.path
        if path == "/":
            self._html(render_workbench())
            return
        if path == "/health":
            self._json({"ok": True, "schema_version": 2, "adapter": "stdlib-preview"})
            return
        if path == "/v1/tasks":
            self._json(
                {
                    "tasks": [
                        {
                            "task_id": run.task_id,
                            "goal": run.request.goal,
                            "workspace_id": run.request.workspace_id,
                            "workspace_path": run.request.workspace_path,
                            "status": run.status.value,
                            "updated_at": run.updated_at,
                        }
                        for run in self.task_service.list_tasks(workspace_id)
                    ]
                }
            )
            return
        task_match = re.fullmatch(r"/v1/tasks/([^/]+)", path)
        if task_match:
            run = self.task_service.find_task(task_match.group(1), workspace_id)
            if run is None:
                self._error(HTTPStatus.NOT_FOUND, "task not found")
            else:
                self._json(self.task_service.task_view(run))
            return
        event_match = re.fullmatch(r"/v1/tasks/([^/]+)/events", path)
        if event_match:
            self._events(event_match.group(1), workspace_id)
            return
        artifact_match = re.fullmatch(r"/v1/tasks/([^/]+)/artifacts", path)
        if artifact_match:
            run = self.task_service.find_task(artifact_match.group(1), workspace_id)
            if run is None:
                self._error(HTTPStatus.NOT_FOUND, "task not found")
            else:
                self._json(
                    {
                        "report_markdown": render_markdown_report(run),
                        "report": report_payload(run),
                    }
                )
            return
        if path == "/v1/memory":
            if not workspace_id:
                self._error(HTTPStatus.BAD_REQUEST, "workspace_id is required")
                return
            include_all = query.get("include_all", ["false"])[0].casefold() == "true"
            self._json(
                {
                    "memories": [
                        to_jsonable(item)
                        for item in self.task_service.memories(
                            workspace_id,
                            include_all=include_all,
                        )
                    ]
                }
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "route not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        workspace_id = query.get("workspace_id", [""])[0]
        payload = self._body()
        path = parsed.path
        try:
            if path == "/v1/tasks":
                self._json(
                    self.task_service.task_view(self.task_service.submit(payload)),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            resume_match = re.fullmatch(r"/v1/tasks/([^/]+)/resume", path)
            if resume_match:
                self._json(
                    self.task_service.task_view(
                        self.task_service.resume(resume_match.group(1), workspace_id)
                    )
                )
                return
            approval_match = re.fullmatch(
                r"/v1/tasks/([^/]+)/approvals/([^/]+)",
                path,
            )
            if approval_match:
                self._json(
                    self.task_service.task_view(
                        self.task_service.decide(
                            approval_match.group(1),
                            approval_match.group(2),
                            payload,
                            workspace_id,
                        )
                    )
                )
                return
            if path == "/v1/memory/promote":
                self._json(
                    to_jsonable(
                        self.task_service.promote(
                            str(payload.get("task_id", "")),
                            str(payload.get("workspace_id", "")),
                        )
                    )
                )
                return
            supersede_match = re.fullmatch(
                r"/v1/memory/([^/]+)/supersede",
                path,
            )
            if supersede_match:
                self._json(
                    to_jsonable(
                        self.task_service.supersede_memory(
                            supersede_match.group(1),
                            payload,
                            workspace_id,
                        )
                    )
                )
                return
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except (ValueError, RuntimeError, OSError) as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "route not found")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        workspace_id = parse_qs(parsed.query).get("workspace_id", [""])[0]
        match = re.fullmatch(r"/v1/memory/([^/]+)", parsed.path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "route not found")
            return
        try:
            self._json(
                to_jsonable(
                    self.task_service.delete_memory(match.group(1), workspace_id)
                )
            )
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[preview] {format % args}")

    def _events(self, task_id: str, workspace_id: str) -> None:
        try:
            events = self.task_service.events(task_id, workspace_id)
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        body = "".join(
            f"id: {event.get('event_id', '')}\n"
            f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            for event in events
        )
        if not body:
            body = ": keepalive\n\n"
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _json(
        self,
        payload: Any,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, detail: str) -> None:
        self._json({"detail": detail}, status=status)


__all__ = ["serve_preview"]
