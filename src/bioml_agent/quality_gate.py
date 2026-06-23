from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


PROMOTION_QUESTION = "Should we try retrieval-aware conditioning for promoter expression prediction?"


@dataclass(frozen=True)
class CheckSpec:
    name: str
    command: tuple[str, ...]
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def __call__(self, command: tuple[str, ...], cwd: Path) -> CommandResult:
        ...


@dataclass(frozen=True)
class QualityCheckResult:
    name: str
    command: tuple[str, ...]
    passed: bool
    returncode: int
    stdout_tail: str
    stderr_tail: str
    artifacts: tuple[str, ...]
    missing_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    checks: tuple[QualityCheckResult, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


def default_check_specs(python: str | None = None) -> tuple[CheckSpec, ...]:
    py = python or sys.executable
    return (
        CheckSpec(
            name="compile",
            command=(py, "-m", "compileall", "-q", "src"),
        ),
        CheckSpec(
            name="unit_tests",
            command=(py, "-m", "unittest", "discover", "-s", "tests"),
        ),
        CheckSpec(
            name="tool_contracts",
            command=(py, "-m", "bioml_agent.tool_contracts", "--validate"),
        ),
        CheckSpec(
            name="eval_coverage_markdown",
            command=(py, "-m", "bioml_agent.eval_report", "--out", "traces/eval-coverage-report.md"),
            artifacts=("traces/eval-coverage-report.md",),
        ),
        CheckSpec(
            name="eval_coverage_json",
            command=(py, "-m", "bioml_agent.eval_report", "--json", "--out", "traces/eval-coverage-report.json"),
            artifacts=("traces/eval-coverage-report.json",),
        ),
        CheckSpec(
            name="promotion_artifacts",
            command=(
                py,
                "-m",
                "bioml_agent.cli",
                PROMOTION_QUESTION,
                "--experiment-log",
                "examples/epicetra_experiment_log.json",
                "--eval",
                "--audit",
                "--trace-out",
                "traces/promotion-run.json",
                "--report-out",
                "traces/promotion-decision-packet.md",
                "--report-html-out",
                "traces/promotion-decision-review.html",
            ),
            artifacts=(
                "traces/promotion-run.json",
                "traces/promotion-decision-packet.md",
                "traces/promotion-decision-review.html",
            ),
        ),
        CheckSpec(
            name="review_workbench_html",
            command=(
                py,
                "-m",
                "bioml_agent.review_workbench",
                "traces/promotion-run.json",
                "--out",
                "traces/review-workbench.html",
            ),
            artifacts=("traces/review-workbench.html",),
        ),
        CheckSpec(
            name="review_workbench_json",
            command=(
                py,
                "-m",
                "bioml_agent.review_workbench",
                "traces/promotion-run.json",
                "--json",
                "--out",
                "traces/review-workbench.json",
            ),
            artifacts=("traces/review-workbench.json",),
        ),
        CheckSpec(
            name="portfolio_demo",
            command=(py, "-m", "bioml_agent.demo", "--json", "--out", "traces/portfolio-demo.json"),
            artifacts=("traces/portfolio-demo.json",),
        ),
        CheckSpec(
            name="reference_catalog_markdown",
            command=(py, "-m", "bioml_agent.reference_catalog", "--out", "traces/reference-catalog-report.md"),
            artifacts=("traces/reference-catalog-report.md",),
        ),
        CheckSpec(
            name="reference_catalog_json",
            command=(
                py,
                "-m",
                "bioml_agent.reference_catalog",
                "--json",
                "--out",
                "traces/reference-catalog-report.json",
            ),
            artifacts=("traces/reference-catalog-report.json",),
        ),
        CheckSpec(
            name="service_interface_markdown",
            command=(py, "-m", "bioml_agent.service_interface", "--out", "traces/service-interface-report.md"),
            artifacts=("traces/service-interface-report.md",),
        ),
        CheckSpec(
            name="service_interface_json",
            command=(
                py,
                "-m",
                "bioml_agent.service_interface",
                "--json",
                "--out",
                "traces/service-interface-report.json",
            ),
            artifacts=("traces/service-interface-report.json",),
        ),
        CheckSpec(
            name="service_runtime_markdown",
            command=(py, "-m", "bioml_agent.service_runtime", "--out", "traces/service-runtime-demo.md"),
            artifacts=("traces/service-runtime-demo.md",),
        ),
        CheckSpec(
            name="service_runtime_json",
            command=(
                py,
                "-m",
                "bioml_agent.service_runtime",
                "--json",
                "--out",
                "traces/service-runtime-demo.json",
            ),
            artifacts=("traces/service-runtime-demo.json",),
        ),
        CheckSpec(
            name="service_api_markdown",
            command=(py, "-m", "bioml_agent.service_api", "--out", "traces/service-api-readiness.md"),
            artifacts=("traces/service-api-readiness.md",),
        ),
        CheckSpec(
            name="service_api_json",
            command=(
                py,
                "-m",
                "bioml_agent.service_api",
                "--json",
                "--out",
                "traces/service-api-readiness.json",
            ),
            artifacts=("traces/service-api-readiness.json",),
        ),
        CheckSpec(
            name="service_mcp_markdown",
            command=(py, "-m", "bioml_agent.service_mcp", "--out", "traces/service-mcp-readiness.md"),
            artifacts=("traces/service-mcp-readiness.md",),
        ),
        CheckSpec(
            name="service_mcp_json",
            command=(
                py,
                "-m",
                "bioml_agent.service_mcp",
                "--json",
                "--out",
                "traces/service-mcp-readiness.json",
            ),
            artifacts=("traces/service-mcp-readiness.json",),
        ),
        CheckSpec(
            name="portfolio_index",
            command=(py, "-m", "bioml_agent.portfolio_index", "build", "--out", "traces/index.html"),
            artifacts=(
                "traces/index.html",
                "traces/reference-catalog-report.md",
                "traces/review-workbench.html",
                "traces/service-interface-report.md",
                "traces/service-runtime-demo.md",
                "traces/service-api-readiness.md",
                "traces/service-mcp-readiness.md",
            ),
        ),
    )


def run_quality_gate(
    specs: tuple[CheckSpec, ...] | None = None,
    cwd: str | Path = ".",
    runner: CommandRunner | None = None,
    stop_on_failure: bool = False,
) -> QualityGateResult:
    root = Path(cwd)
    command_runner = runner or _run_subprocess
    results: list[QualityCheckResult] = []
    for spec in specs or default_check_specs():
        command_result = command_runner(spec.command, root)
        missing_artifacts = _missing_artifacts(root, spec.artifacts)
        passed = command_result.returncode == 0 and not missing_artifacts
        results.append(
            QualityCheckResult(
                name=spec.name,
                command=spec.command,
                passed=passed,
                returncode=command_result.returncode,
                stdout_tail=_tail(command_result.stdout),
                stderr_tail=_tail(command_result.stderr),
                artifacts=spec.artifacts,
                missing_artifacts=missing_artifacts,
            )
        )
        if stop_on_failure and not passed:
            break
    return QualityGateResult(
        passed=all(check.passed for check in results),
        checks=tuple(results),
    )


def quality_gate_to_dict(result: QualityGateResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "passed_count": result.passed_count,
        "total": result.total,
        "checks": [
            {
                "name": check.name,
                "command": list(check.command),
                "passed": check.passed,
                "returncode": check.returncode,
                "stdout_tail": check.stdout_tail,
                "stderr_tail": check.stderr_tail,
                "artifacts": list(check.artifacts),
                "missing_artifacts": list(check.missing_artifacts),
            }
            for check in result.checks
        ],
    }


def format_quality_gate_result(result: QualityGateResult) -> str:
    lines = [
        "# TracePilot Quality Gate",
        f"Passed: {result.passed}",
        f"Checks: {result.passed_count}/{result.total}",
    ]
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.extend(
            [
                "",
                f"## {status} {check.name}",
                f"Command: `{' '.join(check.command)}`",
                f"Return code: `{check.returncode}`",
            ]
        )
        if check.artifacts:
            lines.append(f"Artifacts: {_format_artifacts(check.artifacts)}")
        if check.missing_artifacts:
            lines.append(f"Missing artifacts: {_format_artifacts(check.missing_artifacts)}")
        if check.stdout_tail:
            lines.extend(["", "Stdout tail:", "```text", check.stdout_tail, "```"])
        if check.stderr_tail:
            lines.extend(["", "Stderr tail:", "```text", check.stderr_tail, "```"])
    return "\n".join(lines)


def write_quality_gate_result(result: QualityGateResult, path: str | Path, json_output: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(quality_gate_to_dict(result), indent=2, ensure_ascii=False)
        if json_output
        else format_quality_gate_result(result)
    )
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local TracePilot quality gate.")
    parser.add_argument("--cwd", default=".", help="Project root to run checks from.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failing check.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--out", help="Write quality gate report to this path.")
    args = parser.parse_args()

    result = run_quality_gate(cwd=args.cwd, stop_on_failure=args.fail_fast)
    text = (
        json.dumps(quality_gate_to_dict(result), indent=2, ensure_ascii=False)
        if args.json
        else format_quality_gate_result(result)
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        if not args.json:
            output_path.with_suffix(".json").write_text(
                json.dumps(quality_gate_to_dict(result), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        _refresh_portfolio_index_after_report(cwd=Path(args.cwd))
    else:
        print(text)
    raise SystemExit(0 if result.passed else 1)


def _run_subprocess(command: tuple[str, ...], cwd: Path) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _missing_artifacts(root: Path, artifacts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in artifacts if not (root / path).is_file())


def _tail(text: str, line_count: int = 18) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-line_count:])


def _format_artifacts(paths: tuple[str, ...]) -> str:
    return ", ".join(f"`{path}`" for path in paths)


def _refresh_portfolio_index_after_report(cwd: Path) -> None:
    try:
        from .portfolio_index import write_portfolio_index

        write_portfolio_index(trace_dir=cwd / "traces", out=cwd / "traces" / "index.html")
    except Exception:
        # The dedicated portfolio_index check already reports generation failures.
        return


if __name__ == "__main__":
    main()
