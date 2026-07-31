from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .models import PlanStep, TaskContext, TaskPlan, TaskRequest

MAX_PLAN_STEPS = 20
MAX_PLANNING_ROUNDS = 3
MAX_PARALLEL_READS = 4
_EFFECT_CLASSES = frozenset({"read", "internal_write", "mutation"})


class PlanningError(RuntimeError):
    pass


class PlanValidationError(PlanningError, ValueError):
    pass


class PlanningLimitError(PlanningError):
    pass


@dataclass(frozen=True)
class PlanningConfig:
    max_steps: int = MAX_PLAN_STEPS
    max_rounds: int = MAX_PLANNING_ROUNDS
    max_parallel_reads: int = MAX_PARALLEL_READS

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= MAX_PLAN_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_PLAN_STEPS}")
        if not 1 <= self.max_rounds <= MAX_PLANNING_ROUNDS:
            raise ValueError(f"max_rounds must be between 1 and {MAX_PLANNING_ROUNDS}")
        if not 1 <= self.max_parallel_reads <= MAX_PARALLEL_READS:
            raise ValueError(
                f"max_parallel_reads must be between 1 and {MAX_PARALLEL_READS}"
            )

    @property
    def max_concurrency(self) -> int:
        return self.max_parallel_reads

    @property
    def max_replans(self) -> int:
        return max(0, self.max_rounds - 1)


class DeterministicPlanner:
    """Build a bounded single-agent DAG from a task and its local context.

    The planner deliberately uses stable step identifiers and ordering. An LLM
    can enrich the query or rationale outside this class without changing the
    execution and recovery invariants enforced here.
    """

    def __init__(
        self,
        config: PlanningConfig | None = None,
        *,
        available_tools: Iterable[str] | Any | None = None,
        tool_effects: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or PlanningConfig()
        self._available_tools, inferred_effects = _normalize_available_tools(available_tools)
        self._tool_effects = {**inferred_effects, **dict(tool_effects or {})}

    def plan(self, request: TaskRequest, context: TaskContext) -> TaskPlan:
        if not request.goal.strip():
            raise PlanValidationError("task goal must not be empty")
        plan = self._build_plan(request, context, revision=1)
        validate_plan(plan, config=self.config)
        return plan

    def replan(
        self,
        request: TaskRequest,
        context: TaskContext,
        previous_plan: TaskPlan,
        *,
        failed_steps: Iterable[str] = (),
        evidence_gaps: Iterable[str] = (),
    ) -> TaskPlan:
        revision = previous_plan.revision + 1
        if revision > self.config.max_rounds:
            raise PlanningLimitError(
                f"planning round limit exceeded: {revision} > {self.config.max_rounds}"
            )

        failed = tuple(dict.fromkeys(failed_steps))
        gaps = tuple(
            item.strip() for item in evidence_gaps if isinstance(item, str) and item.strip()
        )
        plan = self._build_plan(
            request,
            context,
            revision=revision,
            failed_steps=failed,
            evidence_gaps=gaps,
        )
        validate_plan(plan, config=self.config)
        return plan

    def _build_plan(
        self,
        request: TaskRequest,
        context: TaskContext,
        *,
        revision: int,
        failed_steps: tuple[str, ...] = (),
        evidence_gaps: tuple[str, ...] = (),
    ) -> TaskPlan:
        steps: list[PlanStep] = []

        def add(
            step_id: str,
            description: str,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            depends_on: tuple[str, ...] = (),
            required_evidence: bool = False,
            effect_class: str | None = None,
        ) -> None:
            if len(steps) >= self.config.max_steps or not self._has_tool(tool_name):
                return
            known_ids = {item.step_id for item in steps}
            dependencies = tuple(item for item in depends_on if item in known_ids)
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=description,
                    tool_name=tool_name,
                    arguments=arguments,
                    depends_on=dependencies,
                    effect_class=effect_class or self._effect_for(tool_name),
                    required_evidence=required_evidence,
                )
            )

        add(
            "index_workspace",
            "Index the current workspace snapshot.",
            "index_workspace",
            {
                "workspace_path": request.workspace_path,
                "workspace_id": request.workspace_id,
            },
            effect_class="internal_write",
        )
        add(
            "search_index",
            "Retrieve relevant local documentation and source code.",
            "search_index",
            {"query": request.goal, "top_k": 20},
            depends_on=("index_workspace",),
            required_evidence=True,
        )
        add(
            "search_code",
            "Search the workspace for implementation context.",
            "search_code",
            {"query": request.goal, "workspace_path": request.workspace_path},
            required_evidence=True,
        )
        add(
            "inspect_git",
            "Inspect the workspace revision and changed files.",
            "inspect_git",
            {"workspace_path": request.workspace_path},
            required_evidence=True,
        )
        add(
            "web_search",
            "Search current external technical sources.",
            "web_search",
            {"query": request.goal, "top_k": 10},
            required_evidence=True,
        )

        for index, url in enumerate(request.source_urls, start=1):
            add(
                f"fetch_url_{index}",
                f"Fetch pinned source {index}.",
                "fetch_url",
                {"url": url},
                required_evidence=True,
            )

        all_gaps = list(evidence_gaps)
        stored_gaps = context.metadata.get("evidence_gaps", ())
        if isinstance(stored_gaps, str):
            all_gaps.append(stored_gaps)
        elif isinstance(stored_gaps, Iterable):
            all_gaps.extend(str(item) for item in stored_gaps)
        for index, gap in enumerate(dict.fromkeys(item.strip() for item in all_gaps if item.strip()), start=1):
            add(
                f"supplemental_search_{revision}_{index}",
                f"Close evidence gap: {gap}",
                "web_search",
                {"query": f"{request.goal} {gap}", "top_k": 10},
                required_evidence=True,
            )

        if failed_steps:
            add(
                f"recovery_search_{revision}",
                "Recover from failed plan steps using the local hybrid index.",
                "search_index",
                {
                    "query": request.goal,
                    "top_k": 20,
                    "failed_steps": list(failed_steps),
                },
                depends_on=("index_workspace",),
                required_evidence=True,
            )

        memory_hints = tuple(
            hint
            for hint in (_memory_hint(memory.content) for memory in context.memories)
            if hint
        )
        if memory_hints:
            add(
                "verify_decision_memory",
                "Revalidate confirmed prior decisions against current evidence.",
                "search_index",
                {
                    "query": " ".join(
                        (request.goal, *memory_hints[:3])
                    )[:2_000],
                    "top_k": 20,
                },
                depends_on=("index_workspace",),
                required_evidence=True,
            )

        evidence_steps = tuple(item.step_id for item in steps if item.required_evidence)
        if request.execution_policy != "read_only":
            add(
                "propose_patch",
                "Draft a reviewable action artifact without changing user files.",
                "propose_patch",
                {
                    "goal": request.goal,
                    "constraints": list(request.constraints),
                    "workspace_path": request.workspace_path,
                },
                depends_on=evidence_steps,
                effect_class="internal_write",
            )
        requested_mutation = request.metadata.get("requested_mutation")
        if requested_mutation and request.execution_policy != "read_only":
            if not isinstance(requested_mutation, Mapping):
                raise PlanValidationError(
                    "requested_mutation metadata must be an object"
                )
            mutation_tool = str(requested_mutation.get("tool_name", ""))
            if mutation_tool not in {"apply_patch", "run_command"}:
                raise PlanValidationError(
                    "requested_mutation tool_name must be apply_patch or run_command"
                )
            if not any(item.step_id == "propose_patch" for item in steps):
                raise PlanValidationError(
                    "requested mutation requires the propose_patch action-plan step"
                )
            mutation_arguments = requested_mutation.get("arguments", {})
            if not isinstance(mutation_arguments, Mapping):
                raise PlanValidationError(
                    "requested_mutation arguments must be an object"
                )
            add(
                "requested_mutation",
                f"Execute approved action with {mutation_tool}.",
                mutation_tool,
                dict(mutation_arguments),
                depends_on=("propose_patch",),
                effect_class="mutation",
            )

        if not steps and self._available_tools:
            name = sorted(self._available_tools)[0]
            add(
                "research",
                "Collect task evidence.",
                name,
                {"query": request.goal},
                required_evidence=True,
            )
        if not steps:
            raise PlanValidationError("no available tools can satisfy the task")

        return TaskPlan(
            task_id=request.task_id,
            steps=tuple(steps),
            revision=revision,
            rationale=(
                "Collect local and external evidence in parallel, then create a "
                "reviewable action proposal."
            ),
            max_concurrency=self.config.max_parallel_reads,
            metadata={
                "planning_round": revision,
                "failed_steps": list(failed_steps),
                "evidence_gaps": list(evidence_gaps),
                "single_agent": True,
                "recalled_memory_ids": [
                    memory.memory_id for memory in context.memories
                ],
            },
        )

    def _has_tool(self, name: str) -> bool:
        return self._available_tools is None or name in self._available_tools

    def _effect_for(self, name: str) -> str:
        return self._tool_effects.get(name, "read")


def validate_plan(
    plan: TaskPlan,
    *,
    config: PlanningConfig | None = None,
) -> None:
    config = config or PlanningConfig()
    errors: list[str] = []
    if not plan.task_id:
        errors.append("plan task_id must not be empty")
    if not plan.steps:
        errors.append("plan must contain at least one step")
    if len(plan.steps) > config.max_steps:
        errors.append(f"plan exceeds step limit: {len(plan.steps)} > {config.max_steps}")
    if not 1 <= plan.revision <= config.max_rounds:
        errors.append(
            f"plan revision must be between 1 and {config.max_rounds}: {plan.revision}"
        )
    if not 1 <= plan.max_concurrency <= config.max_parallel_reads:
        errors.append(
            "plan max_concurrency exceeds configured parallel read limit: "
            f"{plan.max_concurrency} > {config.max_parallel_reads}"
        )

    ids = [step.step_id for step in plan.steps]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    errors.extend(f"duplicate step id: {item}" for item in duplicates)
    known = set(ids)
    for step in plan.steps:
        if not step.step_id:
            errors.append("plan step id must not be empty")
        if not step.tool_name:
            errors.append(f"{step.step_id or '<unknown>'}: tool_name must not be empty")
        if step.effect_class not in _EFFECT_CLASSES:
            errors.append(
                f"{step.step_id}: unsupported effect_class '{step.effect_class}'"
            )
        for dependency in step.depends_on:
            if dependency == step.step_id:
                errors.append(f"{step.step_id}: step cannot depend on itself")
            elif dependency not in known:
                errors.append(f"{step.step_id}: unknown dependency '{dependency}'")

    if not duplicates and not _is_acyclic(plan.steps):
        errors.append("plan dependency graph contains a cycle")
    if errors:
        raise PlanValidationError("; ".join(errors))


def ready_steps(
    plan: TaskPlan,
    completed: Iterable[str] = (),
    running: Iterable[str] = (),
    *,
    limit: int | None = None,
) -> tuple[PlanStep, ...]:
    completed_set = set(completed)
    running_set = set(running)
    capacity = plan.max_concurrency - len(running_set)
    if limit is not None:
        capacity = min(capacity, max(0, limit))
    if capacity <= 0:
        return ()

    candidates = [
        step
        for step in plan.steps
        if step.step_id not in completed_set
        and step.step_id not in running_set
        and set(step.depends_on).issubset(completed_set)
    ]
    if not candidates:
        return ()
    serialized = next(
        (step for step in candidates if step.effect_class != "read"),
        None,
    )
    if serialized is not None:
        return (serialized,) if not running_set else ()
    return tuple(candidates[:capacity])


def topological_batches(plan: TaskPlan) -> tuple[tuple[PlanStep, ...], ...]:
    validate_plan(plan)
    completed: set[str] = set()
    batches: list[tuple[PlanStep, ...]] = []
    while len(completed) < len(plan.steps):
        batch = ready_steps(plan, completed)
        if not batch:
            raise PlanValidationError("plan cannot make progress")
        batches.append(batch)
        completed.update(item.step_id for item in batch)
    return tuple(batches)


def _is_acyclic(steps: tuple[PlanStep, ...]) -> bool:
    dependencies = {step.step_id: set(step.depends_on) for step in steps}
    ready = [step_id for step_id, deps in dependencies.items() if not deps]
    visited = 0
    while ready:
        step_id = ready.pop()
        visited += 1
        for candidate_id, candidate_deps in dependencies.items():
            if step_id not in candidate_deps:
                continue
            candidate_deps.remove(step_id)
            if not candidate_deps:
                ready.append(candidate_id)
    return visited == len(steps)


def _normalize_available_tools(
    tools: Iterable[str] | Any | None,
) -> tuple[set[str] | None, dict[str, str]]:
    if tools is None:
        return None, {}
    names_value = getattr(tools, "names", None)
    if names_value is not None:
        names = set(names_value() if callable(names_value) else names_value)
        effects: dict[str, str] = {}
        contract_getter = getattr(tools, "contract", None)
        if callable(contract_getter):
            for name in names:
                contract = contract_getter(name)
                effect = getattr(contract, "effect_class", "read")
                effects[name] = getattr(effect, "value", effect)
        return names, effects
    return set(tools), {}


def _memory_hint(content: Any) -> str:
    if isinstance(content, Mapping):
        for key in ("summary", "recommendation", "decision", "goal"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
        return ""
    return str(content).strip()[:500] if content is not None else ""


PlanningEngine = DeterministicPlanner


__all__ = [
    "DeterministicPlanner",
    "MAX_PARALLEL_READS",
    "MAX_PLANNING_ROUNDS",
    "MAX_PLAN_STEPS",
    "PlanValidationError",
    "PlanningConfig",
    "PlanningEngine",
    "PlanningError",
    "PlanningLimitError",
    "ready_steps",
    "topological_batches",
    "validate_plan",
]
