from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


@dataclass(frozen=True)
class ExperimentBudget:
    max_runs: int = 30
    max_concurrent: int = 2
    run_timeout_seconds: float = 180
    total_run_seconds: float = 5400
    threads_per_run: int = 1

    def __post_init__(self) -> None:
        for name in ("max_runs", "max_concurrent", "threads_per_run"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_concurrent > self.max_runs:
            raise ValueError("max_concurrent exceeds max_runs")
        for name in ("run_timeout_seconds", "total_run_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.total_run_seconds < self.run_timeout_seconds:
            raise ValueError("total_run_seconds cannot be smaller than one run timeout")


@dataclass(frozen=True)
class Variant:
    variant_id: str
    changes: tuple[dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        identifier(self.variant_id)
        seen: set[str] = set()
        for change in self.changes:
            if set(change) != {"path", "expected_sha256", "content"}:
                raise ValueError("a change requires path, expected_sha256 and content")
            path = Path(change["path"])
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("patch paths must be relative and contained")
            if change["path"] in seen:
                raise ValueError("duplicate patch target")
            seen.add(change["path"])
            if not re.fullmatch(r"[a-f0-9]{64}", change["expected_sha256"]):
                raise ValueError("patch must name the expected source SHA-256")
            if not isinstance(change["content"], str):
                raise ValueError("patch content must be text")


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    mechanism: str
    falsification: str
    evidence_refs: tuple[str, ...] = ()
    origin: str = "human"

    def __post_init__(self) -> None:
        identifier(self.hypothesis_id)
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.statement, self.mechanism, self.falsification)):
            raise ValueError("hypothesis must include statement, mechanism and falsification")
        if self.origin not in {"human", "predefined_benchmark", "model_proposal"}:
            raise ValueError("unknown hypothesis origin")


@dataclass(frozen=True)
class ResearchSpec:
    project_id: str
    goal: str
    workspace: str
    command: tuple[str, ...]
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    variants: tuple[Variant, ...]
    hypothesis: Hypothesis
    baseline: str = "baseline"
    metric: str = "accuracy"
    direction: str = "maximize"
    split_seed: int = 20261102
    partition: str = "validation"
    minimum_effect: float = 0.0
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)
    schema_version: int = 1
    required_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifier(self.project_id)
        if self.schema_version != 1:
            raise ValueError("unsupported research schema version")
        if not self.goal.strip() or not self.command:
            raise ValueError("goal and an argv command are required")
        if not all(isinstance(s, str) and s for s in self.command):
            raise ValueError("command must contain nonempty strings")
        if not any("{output}" in part for part in self.command):
            raise ValueError("command must include the {output} metrics path")
        if not self.datasets or len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets must be nonempty and unique")
        for dataset in self.datasets:
            identifier(dataset)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int) or self.split_seed < 0:
            raise ValueError("split_seed must be a nonnegative integer")
        identifier(self.metric)
        if any(not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", name) for name in self.required_modules):
            raise ValueError("required_modules must contain Python module names")
        ids = [v.variant_id for v in self.variants]
        if len(ids) < 2 or len(set(ids)) != len(ids) or self.baseline not in ids:
            raise ValueError("variants require a baseline and at least one unique candidate")
        if next(v for v in self.variants if v.variant_id == self.baseline).changes:
            raise ValueError("baseline must preserve the source workspace")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be maximize or minimize")
        if self.partition != "validation":
            raise ValueError("selection runs use validation only; test evaluation is a separate protocol")
        if not math.isfinite(self.minimum_effect) or self.minimum_effect < 0:
            raise ValueError("minimum_effect must be finite and nonnegative")
        if self.run_count > self.budget.max_runs:
            raise ValueError(f"{self.run_count} planned runs exceed budget {self.budget.max_runs}")

    @property
    def run_count(self) -> int:
        return len(self.datasets) * len(self.seeds) * len(self.variants)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, base_dir: Path | None = None) -> ResearchSpec:
        data = dict(value)
        workspace = Path(str(data["workspace"])).expanduser()
        if not workspace.is_absolute():
            workspace = (base_dir or Path.cwd()) / workspace
        data["workspace"] = str(workspace.resolve())
        for key in ("command", "datasets", "seeds"):
            data[key] = tuple(data[key])
        data["required_modules"] = tuple(data.get("required_modules", ()))
        data["variants"] = tuple(Variant(v["variant_id"], tuple(v.get("changes", ())))
                                 for v in data["variants"])
        hypothesis = dict(data["hypothesis"])
        hypothesis["evidence_refs"] = tuple(hypothesis.get("evidence_refs", ()))
        data["hypothesis"] = Hypothesis(**hypothesis)
        data["budget"] = ExperimentBudget(**data.get("budget", {}))
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> ResearchSpec:
        target = Path(path).resolve()
        return cls.from_dict(json.loads(target.read_text()), base_dir=target.parent)
