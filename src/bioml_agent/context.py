from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .fixtures import DEFAULT_CONTEXT
from .schemas import ExperimentContext


class ExperimentContextProvider(Protocol):
    def read(self, project_name: str) -> ExperimentContext:
        ...


class FixtureExperimentContextProvider:
    def __init__(self, context: ExperimentContext = DEFAULT_CONTEXT) -> None:
        self.context = context

    def read(self, project_name: str) -> ExperimentContext:
        if project_name != self.context.project_name:
            raise ValueError(f"unknown project context: {project_name}")
        return self.context


class FileExperimentContextProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._contexts = _load_contexts(self.path)

    def read(self, project_name: str) -> ExperimentContext:
        try:
            return self._contexts[project_name]
        except KeyError as exc:
            known = ", ".join(sorted(self._contexts)) or "(none)"
            raise ValueError(f"unknown project context: {project_name}; available: {known}") from exc


@dataclass(frozen=True)
class ExperimentRunRecord:
    run_id: str
    description: str
    status: str
    split: str
    metrics: dict[str, float]


class ExperimentLogContextProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._contexts = _load_experiment_log_contexts(self.path)

    def read(self, project_name: str) -> ExperimentContext:
        try:
            return self._contexts[project_name]
        except KeyError as exc:
            known = ", ".join(sorted(self._contexts)) or "(none)"
            raise ValueError(f"unknown project context: {project_name}; available: {known}") from exc


def build_context_provider(
    context_file: str | Path | None = None,
    experiment_log: str | Path | None = None,
) -> ExperimentContextProvider:
    if context_file and experiment_log:
        raise ValueError("use either context_file or experiment_log, not both")
    if experiment_log:
        return ExperimentLogContextProvider(experiment_log)
    if context_file:
        return FileExperimentContextProvider(context_file)
    return FixtureExperimentContextProvider()


def _load_contexts(path: Path) -> dict[str, ExperimentContext]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment context JSON at {path}: {exc}") from exc

    contexts: dict[str, ExperimentContext] = {}
    for index, record in enumerate(_context_records(raw), start=1):
        context = _context_from_record(record, path=path, index=index)
        if context.project_name in contexts:
            raise ValueError(f"duplicate project context {context.project_name!r} in {path}")
        contexts[context.project_name] = context
    if not contexts:
        raise ValueError(f"no experiment contexts found in {path}")
    return contexts


def _context_records(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and "contexts" in raw:
        records = raw["contexts"]
    elif isinstance(raw, dict):
        records = [raw]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("experiment context JSON must be an object, a list, or an object with contexts")

    if not isinstance(records, list):
        raise ValueError("experiment context field 'contexts' must be a list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each experiment context record must be an object")
    return records


def _context_from_record(raw: dict[str, Any], path: Path, index: int) -> ExperimentContext:
    return ExperimentContext(
        project_name=_required_str(raw, "project_name", path=path, index=index),
        trusted_split=_required_str(raw, "trusted_split", path=path, index=index),
        diagnostic_split=_required_str(raw, "diagnostic_split", path=path, index=index),
        baseline=_required_str(raw, "baseline", path=path, index=index),
        metrics=_required_str_tuple(raw, "metrics", path=path, index=index),
        constraints=_required_str_tuple(raw, "constraints", path=path, index=index),
    )


def _required_str(raw: dict[str, Any], field: str, path: Path, index: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"context #{index} in {path} requires non-empty string field {field!r}")
    return value


def _required_str_tuple(raw: dict[str, Any], field: str, path: Path, index: int) -> tuple[str, ...]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"context #{index} in {path} requires non-empty list field {field!r}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"context #{index} in {path} field {field!r} must contain only strings")
        items.append(item)
    return tuple(items)


def _load_experiment_log_contexts(path: Path) -> dict[str, ExperimentContext]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment log JSON at {path}: {exc}") from exc

    contexts: dict[str, ExperimentContext] = {}
    for index, project in enumerate(_experiment_log_projects(raw), start=1):
        context = _context_from_experiment_project(project, path=path, index=index)
        if context.project_name in contexts:
            raise ValueError(f"duplicate experiment log project {context.project_name!r} in {path}")
        contexts[context.project_name] = context
    if not contexts:
        raise ValueError(f"no experiment log projects found in {path}")
    return contexts


def _experiment_log_projects(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and "projects" in raw:
        projects = raw["projects"]
    elif isinstance(raw, dict):
        projects = [raw]
    elif isinstance(raw, list):
        projects = raw
    else:
        raise ValueError("experiment log JSON must be an object, a list, or an object with projects")
    if not isinstance(projects, list):
        raise ValueError("experiment log field 'projects' must be a list")
    for project in projects:
        if not isinstance(project, dict):
            raise ValueError("each experiment log project must be an object")
    return projects


def _context_from_experiment_project(raw: dict[str, Any], path: Path, index: int) -> ExperimentContext:
    project_name = _required_str(raw, "project_name", path=path, index=index)
    trusted_split = _required_str(raw, "trusted_split", path=path, index=index)
    diagnostic_split = _required_str(raw, "diagnostic_split", path=path, index=index)
    runs = _run_records(raw.get("runs"), path=path, project_name=project_name)
    selection_metric = str(raw.get("selection_metric", "Pearson"))
    baseline = _select_baseline_run(
        runs=runs,
        trusted_split=trusted_split,
        selection_metric=selection_metric,
        path=path,
        project_name=project_name,
    )
    metrics = (
        _required_str_tuple(raw, "metrics", path=path, index=index)
        if "metrics" in raw
        else _metrics_from_run(baseline, path=path, project_name=project_name)
    )
    constraints = (
        _required_str_tuple(raw, "constraints", path=path, index=index)
        if "constraints" in raw
        else _default_constraints(trusted_split=trusted_split, diagnostic_split=diagnostic_split)
    )
    return ExperimentContext(
        project_name=project_name,
        trusted_split=trusted_split,
        diagnostic_split=diagnostic_split,
        baseline=_baseline_text(baseline, trusted_split=trusted_split, metrics=metrics),
        metrics=metrics,
        constraints=constraints,
    )


def _run_records(raw: Any, path: Path, project_name: str) -> tuple[ExperimentRunRecord, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"experiment log project {project_name!r} in {path} requires non-empty runs list")
    records: list[ExperimentRunRecord] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"run #{index} for project {project_name!r} in {path} must be an object")
        run_id = _required_run_str(item, "run_id", path=path, project_name=project_name, index=index)
        description = str(item.get("description") or run_id)
        status = _required_run_str(item, "status", path=path, project_name=project_name, index=index)
        split = _required_run_str(item, "split", path=path, project_name=project_name, index=index)
        records.append(
            ExperimentRunRecord(
                run_id=run_id,
                description=description,
                status=status,
                split=split,
                metrics=_run_metrics(item.get("metrics"), path=path, project_name=project_name, index=index),
            )
        )
    return tuple(records)


def _required_run_str(raw: dict[str, Any], field: str, path: Path, project_name: str, index: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"run #{index} for project {project_name!r} in {path} requires string field {field!r}")
    return value


def _run_metrics(raw: Any, path: Path, project_name: str, index: int) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"run #{index} for project {project_name!r} in {path} requires non-empty metrics object")
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"run #{index} for project {project_name!r} in {path} has an invalid metric name")
        if not isinstance(value, int | float):
            raise ValueError(f"run #{index} for project {project_name!r} in {path} metric {key!r} must be numeric")
        metrics[key] = float(value)
    return metrics


def _select_baseline_run(
    runs: tuple[ExperimentRunRecord, ...],
    trusted_split: str,
    selection_metric: str,
    path: Path,
    project_name: str,
) -> ExperimentRunRecord:
    eligible = [
        run
        for run in runs
        if run.split == trusted_split and run.status in {"trusted_baseline", "promoted", "baseline"}
    ]
    if not eligible:
        raise ValueError(
            f"experiment log project {project_name!r} in {path} has no trusted baseline run on {trusted_split!r}"
        )
    with_metric = [run for run in eligible if selection_metric in run.metrics]
    if with_metric:
        return max(with_metric, key=lambda run: run.metrics[selection_metric])
    return eligible[0]


def _metrics_from_run(run: ExperimentRunRecord, path: Path, project_name: str) -> tuple[str, ...]:
    if not run.metrics:
        raise ValueError(f"baseline run {run.run_id!r} for project {project_name!r} in {path} has no metrics")
    return tuple(run.metrics)


def _default_constraints(trusted_split: str, diagnostic_split: str) -> tuple[str, ...]:
    return (
        f"main claims must use {trusted_split}",
        f"{diagnostic_split} is diagnostic only",
        "new modules need ablation against the trusted baseline",
        "promotion requires metric improvement without worse leakage risk",
    )


def _baseline_text(run: ExperimentRunRecord, trusted_split: str, metrics: tuple[str, ...]) -> str:
    metric_parts = []
    for metric in metrics:
        if metric in run.metrics:
            metric_parts.append(f"{metric}={run.metrics[metric]:.4g}")
    metric_text = ", ".join(metric_parts)
    if metric_text:
        return f"{run.description} ({run.run_id}; {metric_text} on {trusted_split})"
    return f"{run.description} ({run.run_id}; {trusted_split})"
