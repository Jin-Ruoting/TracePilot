from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from .agent import TracePilotAgent, workspace_id_for_path
from .memory import MemoryStore
from .models import TaskRequest
from .rag import Ingestor, RagIndex
from .reporting import render_markdown_report, write_markdown_report
from .serialization import (
    run_from_dict,
    run_to_dict,
    run_to_public_dict,
    to_jsonable,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracepilot",
        description="Evidence-driven technical research and approval-gated execution.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    ingest = subparsers.add_parser("ingest", help="Index local and frozen sources.")
    ingest.add_argument("workspace", nargs="?", default=".")
    ingest.add_argument("--frozen-sources")
    ingest.add_argument("--source-url", action="append", default=[])
    ingest.add_argument("--db", default=".cache/tracepilot/tracepilot.db")
    ingest.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="Run a technical research task.")
    run.add_argument("goal")
    run.add_argument("--workspace", default=".")
    run.add_argument("--workspace-id")
    run.add_argument("--frozen-sources")
    run.add_argument("--source-url", action="append", default=[])
    run.add_argument("--constraint", action="append", default=[])
    run.add_argument("--db", default=".cache/tracepilot/tracepilot.db")
    run.add_argument("--json-out")
    run.add_argument(
        "--public-output",
        action="store_true",
        help="Sanitize local paths and common secrets in --json-out.",
    )
    run.add_argument("--report-out")
    run.add_argument("--read-only", action="store_true")
    run.add_argument(
        "--command",
        dest="requested_command",
        help="Exact command to propose; it is parsed into argv and still requires approval.",
    )
    run.add_argument(
        "--patch-spec",
        help="JSON file containing a list of structured path/content changes.",
    )

    resume = subparsers.add_parser("resume", help="Resume a persisted task.")
    resume.add_argument("task_id")
    _add_state_scope(resume)
    resume.add_argument("--json-out")
    resume.add_argument(
        "--public-output",
        action="store_true",
        help="Sanitize local paths and common secrets in --json-out.",
    )

    approve = subparsers.add_parser("approve", help="Decide a mutation approval.")
    approve.add_argument("task_id")
    approve.add_argument("approval_id")
    approve.add_argument(
        "--decision",
        choices=("approved", "modified", "rejected"),
        default="approved",
    )
    approve.add_argument("--arguments-json")
    approve.add_argument("--comment", default="")
    approve.add_argument("--decided-by", default="human")
    approve.add_argument("--json-out")
    approve.add_argument(
        "--public-output",
        action="store_true",
        help="Sanitize local paths and common secrets in --json-out.",
    )
    _add_state_scope(approve)

    memory = subparsers.add_parser("memory", help="Manage confirmed decision memory.")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)
    memory_list = memory_subparsers.add_parser("list")
    _add_state_scope(memory_list)
    memory_list.add_argument("--all", action="store_true")
    memory_list.add_argument("--json", action="store_true")

    memory_promote = memory_subparsers.add_parser("promote")
    memory_promote.add_argument("task_id")
    _add_state_scope(memory_promote)

    memory_supersede = memory_subparsers.add_parser("supersede")
    memory_supersede.add_argument("memory_id")
    memory_supersede.add_argument("--content", required=True)
    memory_supersede.add_argument(
        "--from-task",
        help=(
            "Completed task whose current evidence and workspace snapshot "
            "support the replacement."
        ),
    )
    _add_state_scope(memory_supersede)

    memory_delete = memory_subparsers.add_parser("delete")
    memory_delete.add_argument("memory_id")
    _add_state_scope(memory_delete)

    serve = subparsers.add_parser("serve", help="Start the FastAPI workbench.")
    serve.add_argument("--db", default=".cache/tracepilot/workbench.db")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--workspace-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "ingest":
        return _ingest(args)
    if args.action == "run":
        return _run(args)
    if args.action == "resume":
        return _resume(args)
    if args.action == "approve":
        return _approve(args)
    if args.action == "memory":
        return _memory(args)
    if args.action == "serve":
        from .api import serve

        serve(
            db_path=args.db,
            host=args.host,
            port=args.port,
            default_workspace_id=args.workspace_id,
        )
        return 0
    raise AssertionError(f"unhandled command: {args.action}")


def _ingest(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    workspace_id = workspace_id_for_path(workspace)
    ingestor = Ingestor()
    with RagIndex(args.db) as index:
        local = ingestor.ingest_path(workspace)
        local_report = index.sync_documents(
            local,
            source_type="local",
            namespace=workspace_id,
        )
        additional = []
        if args.frozen_sources:
            additional.extend(ingestor.ingest_frozen(args.frozen_sources))
        if args.source_url:
            additional.extend(ingestor.ingest_url(args.source_url))
        additional_report = index.index_documents(
            additional,
            namespace=workspace_id,
        )
        payload = {
            "workspace": str(workspace),
            "workspace_id": workspace_id,
            "local": to_jsonable(local_report),
            "additional": to_jsonable(additional_report),
            "stats": index.stats(namespace=workspace_id),
        }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"Indexed {payload['stats']['active_documents']} active documents "
            f"and {payload['stats']['active_chunks']} chunks."
        )
        print(f"Workspace id: {payload['workspace_id']}")
    return 0


def _run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    metadata: dict[str, Any] = {}
    if args.requested_command:
        metadata["requested_mutation"] = {
            "tool_name": "run_command",
            "arguments": {
                "workspace_path": str(workspace),
                "argv": shlex.split(args.requested_command),
            },
        }
    if args.patch_spec:
        changes = json.loads(Path(args.patch_spec).read_text(encoding="utf-8"))
        if not isinstance(changes, list):
            raise ValueError("patch spec must contain a JSON list")
        metadata["requested_mutation"] = {
            "tool_name": "apply_patch",
            "arguments": {
                "workspace_path": str(workspace),
                "changes": changes,
            },
        }
    request = TaskRequest(
        goal=args.goal,
        workspace_id=args.workspace_id or workspace_id_for_path(workspace),
        workspace_path=str(workspace),
        source_urls=tuple(args.source_url),
        constraints=tuple(args.constraint),
        execution_policy="read_only" if args.read_only else "approval_required",
        metadata=metadata,
    )
    with TracePilotAgent(args.db) as agent:
        run = agent.run(request, frozen_sources=args.frozen_sources)
    _write_outputs(
        run,
        args.json_out,
        args.report_out,
        public_output=args.public_output,
    )
    _print_run(run)
    return 1 if run.status.value == "failed" else 0


def _resume(args: argparse.Namespace) -> int:
    with TracePilotAgent(args.db) as agent:
        run = agent.resume(args.workspace_id, args.task_id)
    _write_outputs(run, args.json_out, None, public_output=args.public_output)
    _print_run(run)
    return 1 if run.status.value == "failed" else 0


def _approve(args: argparse.Namespace) -> int:
    arguments = None
    if args.arguments_json:
        arguments = json.loads(Path(args.arguments_json).read_text(encoding="utf-8"))
        if not isinstance(arguments, dict):
            raise ValueError("approval arguments must be a JSON object")
    with TracePilotAgent(args.db) as agent:
        run = agent.decide_approval(
            args.workspace_id,
            args.task_id,
            args.approval_id,
            decision=args.decision,
            arguments=arguments,
            comment=args.comment,
            decided_by=args.decided_by,
        )
    _write_outputs(run, args.json_out, None, public_output=args.public_output)
    _print_run(run)
    return 1 if run.status.value == "failed" else 0


def _memory(args: argparse.Namespace) -> int:
    if args.memory_command == "promote":
        with TracePilotAgent(args.db) as agent:
            entry = agent.promote_decision(args.workspace_id, args.task_id)
        print(json.dumps(to_jsonable(entry), indent=2, ensure_ascii=False))
        return 0
    with MemoryStore(args.db) as store:
        if args.memory_command == "list":
            entries = store.list_decision_memories(
                args.workspace_id,
                status=None if args.all else "active",
                include_deleted=args.all,
            )
            payload = [to_jsonable(entry) for entry in entries]
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                for entry in entries:
                    print(f"{entry.memory_id}\t{entry.status}\t{entry.kind}")
            return 0
        if args.memory_command == "supersede":
            source_run = None
            if args.from_task:
                task_entry = store.get_task(
                    args.workspace_id,
                    args.from_task,
                )
                if task_entry is None:
                    raise KeyError(f"unknown task: {args.from_task}")
                source_run = run_from_dict(task_entry.payload)
                if source_run.status.value != "completed":
                    raise ValueError(
                        "memory supersede source task must be completed"
                    )
            replacement = {
                "workspace_id": args.workspace_id,
                "kind": "decision",
                "content": {"summary": args.content},
                "source_run_id": source_run.run_id if source_run else "",
                "source_task_id": source_run.task_id if source_run else "",
                "source_versions": (
                    source_run.context.source_versions if source_run else {}
                ),
            }
            entry = store.supersede_decision_memory(
                args.workspace_id,
                args.memory_id,
                replacement,
                source=(
                    source_run.context.source_versions
                    if source_run is not None
                    else None
                ),
                workspace_state=(
                    {
                        "workspace_revision": (
                            source_run.context.workspace_revision
                        ),
                        "workspace_hash": source_run.context.workspace_hash,
                        "source_versions": source_run.context.source_versions,
                    }
                    if source_run is not None
                    else None
                ),
            )
            print(json.dumps(to_jsonable(entry), indent=2, ensure_ascii=False))
            return 0
        if args.memory_command == "delete":
            entry = store.delete_decision_memory(
                args.workspace_id,
                args.memory_id,
            )
            print(json.dumps(to_jsonable(entry), indent=2, ensure_ascii=False))
            return 0
    raise AssertionError(f"unhandled memory command: {args.memory_command}")


def _add_state_scope(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--db", default=".cache/tracepilot/tracepilot.db")


def _write_outputs(
    run: Any,
    json_out: str | None,
    report_out: str | None,
    *,
    public_output: bool = False,
) -> None:
    if json_out:
        output = Path(json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            run_to_public_dict(run, project_root=Path.cwd())
            if public_output
            else run_to_dict(run)
        )
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if report_out:
        write_markdown_report(run, report_out)


def _print_run(run: Any) -> None:
    print(f"Task: {run.task_id}")
    print(f"Workspace id: {run.request.workspace_id}")
    print(f"Status: {run.status.value}")
    if run.summary:
        print(render_markdown_report(run))
    if run.approvals:
        pending = [item for item in run.approvals if item.status.value in {"pending", "modified"}]
        for approval in pending:
            print(
                f"Approval required: {approval.approval_id} "
                f"for {approval.tool_name} ({approval.arguments_hash})"
            )
    if run.error:
        print(f"Error: {run.error}")


if __name__ == "__main__":
    raise SystemExit(main())
