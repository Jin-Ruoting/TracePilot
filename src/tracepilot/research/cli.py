from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .models import ResearchSpec
from .reporting import write_report


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("research", help="Run reproducible code-first research projects.")
    commands = parser.add_subparsers(dest="research_action", required=True)
    run = commands.add_parser("run", help="Freeze a spec and run isolated experiments.")
    run.add_argument("spec")
    run.add_argument("--execute", action="store_true", help="Authorize this spec's isolated code and bounded compute.")
    run.add_argument("--project-id")
    run.add_argument("--python", dest="experiment_python")
    run.add_argument("--max-concurrent", type=int)
    run.add_argument("--output-root", default=os.environ.get("TRACEPILOT_STATE_ROOT", "outputs/research"))
    run.add_argument("--artifacts-root", default=os.environ.get("TRACEPILOT_ARTIFACTS_ROOT"))
    run.add_argument("--router-config", help="Kev routing configuration; defaults to TRACEPILOT_ROUTER_CONFIG.")
    _waiting(run)
    for name in ("resume", "status", "report", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("project_id")
        command.add_argument("--output-root", default=os.environ.get("TRACEPILOT_STATE_ROOT", "outputs/research"))
        if name == "resume":
            command.add_argument("--router-config", help="Kev routing configuration; defaults to TRACEPILOT_ROUTER_CONFIG.")
            _waiting(command)
    context = commands.add_parser("context", help="Build evidence and code context with the existing offline RAG.")
    context.add_argument("workspace")
    context.add_argument("--goal", required=True)
    context.add_argument("--evidence", action="append", default=[])
    context.add_argument("--file", action="append")
    context.add_argument("--out", required=True)
    route = commands.add_parser("route", help="Evaluate a routing event; no worker or experiment is executed.")
    route.add_argument("event")
    route.add_argument("--router-config")
    route.add_argument("--endpoint")
    route.add_argument("--model")
    route.add_argument("--api-key-env")
    route.add_argument("--confidence-threshold", type=float)
    route.add_argument("--state-version", type=int, default=0)
    route.add_argument("--out", required=True)
    for name in ("propose", "propose-patch"):
        command = commands.add_parser(name, help="Call a configured model; save a reviewable proposal without execution.")
        command.add_argument("context")
        command.add_argument("--endpoint", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--api-key-env", default="OPENAI_API_KEY")
        command.add_argument("--out", required=True)


def _waiting(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=600)
    parser.add_argument("--poll-interval", type=float, default=0.25)


def handle(args: argparse.Namespace) -> int:
    if args.research_action in {"context", "route"}:
        if args.research_action == "context":
            from .context import build_context
            result = build_context(args.workspace, args.goal, evidence_paths=args.evidence, files=args.file)
        else:
            from ..routing import configured_router
            result = configured_router(args.router_config, endpoint=args.endpoint, model=args.model,
                                       api_key_env=args.api_key_env, threshold=args.confidence_threshold).route(
                json.loads(Path(args.event).read_text()), state_version=args.state_version).to_dict()
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(str(output.resolve()))
        return 0
    if args.research_action in {"propose", "propose-patch"}:
        from ..workers.research import ResearchWorker
        from ..workers.gateway import CompatibleJSONGateway
        context = json.loads(Path(args.context).read_text())
        worker = ResearchWorker(CompatibleJSONGateway(args.endpoint, args.model, api_key_env=args.api_key_env))
        result = worker.propose(context) if args.research_action == "propose" else worker.propose_patch(context)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(str(output.resolve()))
        return 0
    from ..coordinator import ResearchCoordinator
    from ..routing import configured_router
    router = configured_router(args.router_config) if args.research_action in {"run", "resume"} else None
    coordinator = ResearchCoordinator(args.output_root, artifacts_root=getattr(args, "artifacts_root", None),
                                      router=router)
    try:
        if args.research_action == "run":
            spec = ResearchSpec.load(args.spec)
            if args.project_id:
                spec = replace(spec, project_id=args.project_id)
            if args.max_concurrent is not None:
                spec = replace(spec, budget=replace(spec.budget, max_concurrent=args.max_concurrent))
            config = {"kind": "local", "python": args.experiment_python or sys.executable}
            coordinator.create(spec, backend=config, allow_execution=args.execute)
            args.project_id = spec.project_id
        if args.research_action in {"run", "resume"}:
            if args.poll_interval <= 0 or args.wait_timeout <= 0:
                raise ValueError("poll interval and wait timeout must be positive")
            value = (coordinator.wait(args.project_id, timeout=args.wait_timeout, interval=args.poll_interval)
                     if args.wait else coordinator.tick(args.project_id))
            if value["status"] in {"completed", "completed_with_failures", "budget_exhausted", "needs_attention"}:
                value["reports"] = write_report(coordinator, args.project_id)
            print(json.dumps(value, indent=2, ensure_ascii=False))
            return 0 if value["status"] in {"active", "completed"} and not value.get("wait_timed_out") else 2
        if args.research_action == "report":
            value = write_report(coordinator, args.project_id)
        elif args.research_action == "cancel":
            value = coordinator.cancel(args.project_id)
        else:
            value = coordinator.status(args.project_id)
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return 0
    finally:
        coordinator.close()
