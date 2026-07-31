from __future__ import annotations

import argparse
import compileall
import json
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .benchmark import render_markdown as render_benchmark_markdown
from .benchmark import report_to_dict as benchmark_to_dict
from .benchmark import run_benchmark
from .builtin_tools import build_default_registry
from .execution import ApprovalRequiredError, ExecutionRuntime
from .rag import RagIndex
from .serialization import checkpoint_from_dict
from .workbench import render_workbench, write_workbench


@dataclass(frozen=True)
class GateCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class QualityGateReport:
    passed: bool
    checks: tuple[GateCheck, ...]


def run_quality_gate(
    root: str | Path | None = None,
    *,
    include_tests: bool = True,
    artifact_dir: str | Path | None = None,
) -> QualityGateReport:
    project_root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    artifacts = Path(artifact_dir or project_root / "traces")
    artifacts.mkdir(parents=True, exist_ok=True)
    checks: list[GateCheck] = []

    checks.append(
        _capture(
            "compile",
            lambda: _compile(project_root),
        )
    )
    if include_tests:
        checks.append(
            _capture(
                "tests",
                lambda: _tests(project_root),
            )
        )
    checks.append(_capture("tool_contracts_and_policy", _tool_contracts))
    checks.append(_capture("checkpoint_compatibility", _checkpoint_compatibility))
    checks.append(
        _capture(
            "generic_benchmark",
            lambda: _benchmark(project_root, artifacts),
        )
    )
    checks.append(
        _capture(
            "workbench",
            lambda: _workbench(artifacts),
        )
    )
    checks.append(_capture("fastapi_adapter", _fastapi_adapter))
    return QualityGateReport(
        passed=all(check.status in {"pass", "skip"} for check in checks),
        checks=tuple(checks),
    )


def render_markdown(report: QualityGateReport) -> str:
    lines = [
        "# TracePilot Quality Gate",
        "",
        f"- Overall: `{'pass' if report.passed else 'fail'}`",
        f"- Checks: `{len(report.checks)}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        detail = check.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check.name} | {check.status} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TracePilot quality gate.")
    parser.add_argument("--out", default="traces/quality-gate-report.md")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    report = run_quality_gate(root, include_tests=not args.skip_tests)
    rendered = (
        json.dumps(asdict(report), indent=2, ensure_ascii=False)
        if args.json
        else render_markdown(report)
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


def _capture(name: str, fn: Callable[[], str]) -> GateCheck:
    try:
        return GateCheck(name=name, status="pass", detail=fn())
    except _SkipCheck as exc:
        return GateCheck(name=name, status="skip", detail=str(exc))
    except Exception as exc:
        return GateCheck(
            name=name,
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
        )


def _compile(root: Path) -> str:
    if not compileall.compile_dir(root / "src", quiet=1):
        raise RuntimeError("compileall reported failures")
    return "Python sources compile successfully"


def _tests(root: Path) -> str:
    env = os.environ.copy()
    source_path = str(root / "src")
    env["PYTHONPATH"] = (
        source_path
        if not env.get("PYTHONPATH")
        else f"{source_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stdout + result.stderr)[-4000:])
    summary = next(
        (
            line.strip()
            for line in reversed((result.stdout + result.stderr).splitlines())
            if line.startswith("Ran ")
        ),
        "test suite passed",
    )
    return summary


def _tool_contracts() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        (workspace / "notes.md").write_text("technical context\n", encoding="utf-8")
        with RagIndex(Path(tmp) / "index.db") as index:
            registry = build_default_registry(
                workspace_root=workspace,
                index=index,
            )
            for contract in registry.contracts:
                contract.validate()
            effects = {
                contract.name: contract.effect_class.value
                for contract in registry.contracts
            }
            if effects.get("apply_patch") != "mutation":
                raise RuntimeError("apply_patch is not classified as mutation")
            if effects.get("run_command") != "mutation":
                raise RuntimeError("run_command is not classified as mutation")
            runtime = ExecutionRuntime(registry, workspace)
            try:
                runtime.execute(
                    "run_command",
                    {
                        "workspace_path": str(workspace),
                        "argv": [sys.executable, "-c", "print('blocked')"],
                    },
                )
            except ApprovalRequiredError:
                pass
            else:
                raise RuntimeError("mutation executed without approval")
    return f"{len(effects)} contracts validated; unapproved mutation blocked"


def _checkpoint_compatibility() -> str:
    run = checkpoint_from_dict(
        {
            "schema_version": 1,
            "state": {
                "question": {"text": "legacy question", "domain": "bioml"},
                "project_name": "legacy",
                "next_node": "done",
                "trace": [],
            },
        }
    )
    if run.metadata.get("migrated_from") != 1:
        raise RuntimeError("v1 checkpoint was not marked as migrated")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from bioml_agent.agent_loop import BioMLResearchAgent

    if BioMLResearchAgent is None:
        raise RuntimeError("legacy BioML agent import failed")
    return "v1 checkpoint mapped to schema v2; legacy import remains available"


def _benchmark(root: Path, artifacts: Path) -> str:
    report = run_benchmark(
        root / "benchmarks" / "general_cases.jsonl",
        root / "benchmarks" / "general_sources.jsonl",
    )
    (artifacts / "general-benchmark.json").write_text(
        json.dumps(benchmark_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifacts / "general-benchmark.md").write_text(
        render_benchmark_markdown(report),
        encoding="utf-8",
    )
    if not report.passed:
        raise RuntimeError(
            f"full TracePilot passed {report.passed_cases}/{report.total_cases}"
        )
    return f"full TracePilot passed {report.passed_cases}/{report.total_cases} frozen cases"


def _workbench(artifacts: Path) -> str:
    html = render_workbench()
    required = (
        'data-tab="evidence"',
        'data-tab="memory"',
        'data-tab="execution"',
        "EventSource",
        "data-approval",
        "Source chunks",
    )
    missing = [marker for marker in required if marker not in html]
    if missing:
        raise RuntimeError(f"workbench is missing: {', '.join(missing)}")
    write_workbench(artifacts / "workbench.html")
    return "task, evidence, memory, approval, execution, and SSE views rendered"


def _fastapi_adapter() -> str:
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:
        raise _SkipCheck("optional FastAPI dependency is not installed") from exc
    from .api import create_app

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Path(tmp) / "api.db")
        paths = {route.path for route in app.routes}
        required = {
            "/v1/tasks",
            "/v1/tasks/{task_id}",
            "/v1/tasks/{task_id}/events",
            "/v1/tasks/{task_id}/approvals/{approval_id}",
            "/v1/tasks/{task_id}/artifacts",
            "/v1/memory",
        }
        missing = sorted(required - paths)
        if missing:
            raise RuntimeError(f"API routes are missing: {', '.join(missing)}")
    return f"FastAPI adapter exposes {len(required)} required route families"


class _SkipCheck(RuntimeError):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
