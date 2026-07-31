from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .builtin_tools import WebSearchProvider, build_default_registry
from .execution import ExecutionRuntime
from .llm import LLMGateway
from .memory import MemoryStore, workspace_fingerprint
from .models import (
    AgentRun,
    ApprovalRequest,
    EvidenceChunk,
    EvidenceClaim,
    MemoryRecord,
    PlanStep,
    TaskContext,
    TaskPlan,
    TaskRequest,
    TaskStatus,
    ToolResult,
    TraceEvent,
    utc_now,
)
from .planning import DeterministicPlanner
from .rag import (
    Ingestor,
    IndexingReport,
    RagIndex,
    check_citation_coverage,
    check_faithfulness,
)
from .serialization import (
    checkpoint_from_dict,
    checkpoint_to_dict,
    run_from_dict,
    to_jsonable,
)


TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}


class TracePilotAgent:
    """Evidence-driven single-agent runtime for technical research and actions."""

    def __init__(
        self,
        db_path: str | Path = ".cache/tracepilot/tracepilot.db",
        *,
        llm: LLMGateway | None = None,
        web_provider: WebSearchProvider | None = None,
        planner: DeterministicPlanner | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.memory = MemoryStore(self.db_path)
        self.index = RagIndex(self.db_path)
        self.llm = llm
        self.web_provider = web_provider
        self._planner = planner

    def close(self) -> None:
        self.index.close()
        self.memory.close()

    def __enter__(self) -> "TracePilotAgent":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def run(
        self,
        request: TaskRequest | str,
        *,
        workspace_path: str | Path | None = None,
        workspace_id: str | None = None,
        frozen_sources: str | Path | Iterable[Mapping[str, Any]] | None = None,
    ) -> AgentRun:
        parsed = self._coerce_request(
            request,
            workspace_path=workspace_path,
            workspace_id=workspace_id,
        )
        workspace = Path(parsed.workspace_path)
        if not workspace.exists():
            raise FileNotFoundError(workspace)
        if not workspace.is_dir():
            raise NotADirectoryError(workspace)
        self._bind_workspace(parsed)

        context = TaskContext(
            task_id=parsed.task_id,
            workspace_id=parsed.workspace_id,
            workspace_path=parsed.workspace_path,
        )
        run = AgentRun(
            task_id=parsed.task_id,
            request=parsed,
            context=context,
            status=TaskStatus.CREATED,
            metadata={"schema_version": 2, "single_agent": True},
        )
        self._event(run, "task_created", status=run.status.value)
        self._persist(run)

        try:
            self._index_sources(run, frozen_sources=frozen_sources)
            self._load_decision_memory(run)
            run.transition(TaskStatus.PLANNING)
            self._event(run, "status_changed", status=run.status.value)

            registry = build_default_registry(
                workspace_root=workspace,
                index=self.index,
                workspace_id=parsed.workspace_id,
                web_provider=self.web_provider,
            )
            planner = self._planner or DeterministicPlanner(available_tools=registry)
            run.plan = planner.plan(parsed, run.context)
            self._persist(run)

            run.transition(TaskStatus.RESEARCHING)
            self._event(run, "status_changed", status=run.status.value)
            outcome = self._execute_research_plan(run, registry, planner)
            if outcome in {"awaiting_approval", "needs_revision"}:
                return run

            self._finalize_research(run)
            return run
        except Exception as exc:
            self._fail(run, exc)
            return run

    def resume(self, workspace_id: str, task_id: str) -> AgentRun:
        entry = self.memory.get_task(workspace_id, task_id)
        if entry is None:
            raise KeyError(f"unknown task: {task_id}")
        run = run_from_dict(entry.payload)
        latest = self.memory.latest_checkpoint(workspace_id, task_id)
        if latest is not None:
            checkpoint_run = checkpoint_from_dict(latest.payload)
            if checkpoint_run.updated_at >= run.updated_at:
                run = checkpoint_run
        if run.status in TERMINAL_STATUSES or run.status == TaskStatus.AWAITING_APPROVAL:
            return run
        if run.status == TaskStatus.EXECUTING:
            claimed = next(
                (
                    approval
                    for approval in run.approvals
                    if approval.metadata.get("execution_state") == "claimed"
                ),
                None,
            )
            if claimed is not None:
                raise RuntimeError(
                    "the task stopped after claiming an approved mutation; "
                    "automatic resume is blocked because execution may already "
                    "have produced side effects"
                )
            raise RuntimeError(
                "an executing mutation cannot be resumed automatically"
            )
        if run.status == TaskStatus.NEEDS_REVISION:
            run.transition(TaskStatus.PLANNING)
        elif run.status in {TaskStatus.CREATED, TaskStatus.INDEXING}:
            return self.run(run.request)
        workspace = Path(run.request.workspace_path)
        registry = build_default_registry(
            workspace_root=workspace,
            index=self.index,
            workspace_id=run.request.workspace_id,
            web_provider=self.web_provider,
        )
        planner = self._planner or DeterministicPlanner(available_tools=registry)
        if run.status == TaskStatus.PLANNING:
            run.plan = planner.replan(
                run.request,
                run.context,
                run.plan,
                evidence_gaps=("resume after interrupted or incomplete execution",),
            )
            run.transition(TaskStatus.RESEARCHING)
        outcome = self._execute_research_plan(run, registry, planner)
        if outcome not in {"awaiting_approval", "needs_revision"}:
            self._finalize_research(run)
        return run

    def decide_approval(
        self,
        workspace_id: str,
        task_id: str,
        approval_id: str,
        *,
        decision: str,
        arguments: Mapping[str, Any] | None = None,
        comment: str = "",
        decided_by: str = "human",
    ) -> AgentRun:
        entry = self.memory.get_task(workspace_id, task_id)
        if entry is None:
            raise KeyError(f"unknown task: {task_id}")
        run = run_from_dict(entry.payload)
        approval = next(
            (item for item in run.approvals if item.approval_id == approval_id),
            None,
        )
        if approval is None:
            raise KeyError(f"unknown approval: {approval_id}")

        already_executed = any(
            result.metadata.get("approval_id") == approval_id
            for result in run.tool_results
        )
        if already_executed:
            return run
        if (
            approval.status.value == "approved"
            and approval.metadata.get("execution_state") == "claimed"
        ):
            raise RuntimeError(
                "the approved mutation has an uncertain prior execution state; "
                "TracePilot will not retry it automatically"
            )

        registry = build_default_registry(
            workspace_root=run.request.workspace_path,
            index=self.index,
            workspace_id=run.request.workspace_id,
            web_provider=self.web_provider,
        )
        runtime = ExecutionRuntime(registry, run.request.workspace_path)
        runtime.restore_approval(approval)
        offset = len(runtime.trace)

        normalized = decision.casefold()
        if normalized in {"reject", "rejected"}:
            updated = runtime.reject(
                approval,
                reason=comment,
                rejected_by=decided_by,
            )
            run.approvals = _replace_approval(run.approvals, updated)
            run.transition(TaskStatus.NEEDS_REVISION)
            self._update_step(run, approval.step_id, "rejected", error=comment)
            self._take_runtime_trace(run, runtime, offset)
            self._persist(run)
            return run
        if normalized in {"modify", "modified"}:
            if arguments is None:
                raise ValueError("modified approval requires arguments")
            updated = runtime.modify(
                approval,
                arguments,
                modified_by=decided_by,
            )
            run.approvals = _replace_approval(run.approvals, updated)
            self._take_runtime_trace(run, runtime, offset)
            self._persist(run)
            return run
        if normalized not in {"approve", "approved"}:
            raise ValueError("decision must be approved, modified, or rejected")

        approved = runtime.approve(
            approval,
            approved_by=decided_by,
            arguments=arguments,
        )
        approved = replace(
            approved,
            metadata={**approved.metadata, "execution_state": "claimed"},
        )
        run.approvals = _replace_approval(run.approvals, approved)
        run.transition(TaskStatus.EXECUTING)
        self._update_step(run, approved.step_id, "running")
        self._event(
            run,
            "mutation_execution_claimed",
            status="executing",
            metadata={
                "approval_id": approved.approval_id,
                "arguments_hash": approved.arguments_hash,
            },
        )
        self._persist(
            run,
            checkpoint_step=f"{approved.step_id}_execution_claimed",
        )
        result = runtime.execute(
            approved.tool_name,
            approved.arguments,
            task_id=run.task_id,
            run_id=run.run_id,
            step_id=approved.step_id,
            approval=approved,
            idempotency_key=approved.approval_id,
            state=run.context.state,
        )
        run.tool_results.append(result)
        approved = replace(
            approved,
            metadata={
                **approved.metadata,
                "execution_state": "completed" if result.success else "failed",
            },
        )
        run.approvals = _replace_approval(run.approvals, approved)
        self._take_runtime_trace(run, runtime, offset)
        self._update_step(
            run,
            approved.step_id,
            "completed" if result.success else "failed",
            error=result.error,
        )
        if not result.success:
            self._fail(run, RuntimeError(result.error or "mutation failed"))
            return run

        run.transition(TaskStatus.VERIFYING)
        verification = runtime.execute(
            "inspect_git",
            {"workspace_path": run.request.workspace_path},
            task_id=run.task_id,
            run_id=run.run_id,
            step_id=f"{approved.step_id}_verification",
            idempotency_key=f"{approved.approval_id}:verify",
        )
        run.tool_results.append(verification)
        self._take_runtime_trace(run, runtime, len(runtime.trace) - 1)
        run.metadata["verification"] = to_jsonable(verification)
        if verification.success:
            try:
                post_execution_index = self._refresh_workspace_snapshot(run)
            except Exception as exc:
                self._fail(
                    run,
                    RuntimeError(
                        "post-execution workspace snapshot failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
                return run
            run.metadata["post_execution_index"] = to_jsonable(
                post_execution_index
            )
            execution_text = (
                f"Approved tool {approved.tool_name} completed successfully, "
                "and the post-execution workspace snapshot completed."
            )
            execution_chunk = EvidenceChunk(
                document_id=f"execution:{result.result_id}",
                source_uri=f"trace://{run.run_id}/{result.result_id}",
                title="Approved execution result",
                text=execution_text,
                metadata={
                    "tool_result_id": result.result_id,
                    "verification_result_id": verification.result_id,
                },
            )
            execution_claim = EvidenceClaim(
                text=execution_text,
                evidence_ids=(execution_chunk.chunk_id,),
                supported=True,
                confidence=1.0,
            )
            run.evidence.extend((execution_chunk, execution_claim))
            run.transition(TaskStatus.COMPLETED)
            run.summary = (
                f"{run.summary} {execution_text}"
            ).strip()
            self._record_evaluation(run)
            self._finish(run)
        else:
            self._fail(run, RuntimeError(verification.error or "verification failed"))
        return run

    def promote_decision(self, workspace_id: str, task_id: str) -> Any:
        entry = self.memory.get_task(workspace_id, task_id)
        if entry is None:
            raise KeyError(f"unknown task: {task_id}")
        run = run_from_dict(entry.payload)
        if run.status != TaskStatus.COMPLETED:
            raise ValueError("only a completed run can be promoted to decision memory")
        record = {
            "workspace_id": workspace_id,
            "kind": "decision",
            "content": {
                "goal": run.request.goal,
                "summary": run.summary,
                "claims": [
                    to_jsonable(item)
                    for item in run.evidence
                    if isinstance(item, EvidenceClaim)
                ],
            },
            "source_run_id": run.run_id,
            "source_task_id": run.task_id,
            "workspace_revision": run.context.workspace_revision,
            "source_versions": run.context.source_versions,
            "confirmed": True,
        }
        return self.memory.promote_decision_memory(
            workspace_id,
            record,
            source=run.context.source_versions,
            workspace_state=self._workspace_state(run.context),
            idempotency_key=f"promote:{run.run_id}",
        )

    def get_task(self, workspace_id: str, task_id: str) -> AgentRun | None:
        entry = self.memory.get_task(workspace_id, task_id)
        return None if entry is None else run_from_dict(entry.payload)

    def list_tasks(self, workspace_id: str) -> list[AgentRun]:
        return [run_from_dict(entry.payload) for entry in self.memory.list_tasks(workspace_id)]

    def _index_sources(
        self,
        run: AgentRun,
        *,
        frozen_sources: str | Path | Iterable[Mapping[str, Any]] | None,
    ) -> None:
        run.transition(TaskStatus.INDEXING)
        self._event(run, "status_changed", status=run.status.value)
        ingestor = Ingestor()
        local_documents = ingestor.ingest_path(run.request.workspace_path)
        documents = list(local_documents)
        local_report = self.index.sync_documents(
            local_documents,
            source_type="local",
            namespace=run.request.workspace_id,
        )
        reports = [local_report]
        if frozen_sources is not None:
            frozen_documents = ingestor.ingest_frozen(frozen_sources)
            documents.extend(frozen_documents)
            reports.append(
                self.index.sync_documents(
                    frozen_documents,
                    source_type="frozen",
                    namespace=run.request.workspace_id,
                )
            )
        url_documents: list[Any] = []
        for url in run.request.source_urls:
            try:
                url_documents.extend(ingestor.ingest_url(url))
            except Exception as exc:
                self._event(
                    run,
                    "source_ingest_failed",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    metadata={"url": url},
                )
        if run.request.source_urls:
            documents.extend(url_documents)
            reports.append(
                self.index.sync_documents(
                    url_documents,
                    source_type="url",
                    namespace=run.request.workspace_id,
                )
            )
        report = _merge_indexing_reports(reports)
        run.context.source_versions = {
            document.uri: document.version or document.content_hash
            for document in documents
        }
        run.context.workspace_revision = _git_revision(Path(run.request.workspace_path))
        run.context.workspace_hash = _document_set_hash(local_documents)
        run.context.metadata["indexing_report"] = to_jsonable(report)
        run.context.evidence = self.index.search(
            run.request.goal,
            top_k=8,
            namespace=run.request.workspace_id,
        )
        self._event(
            run,
            "sources_indexed",
            status="completed",
            outputs=to_jsonable(report),
        )
        self._persist(run, checkpoint_step="sources_indexed")

    def _refresh_workspace_snapshot(self, run: AgentRun) -> IndexingReport:
        documents = Ingestor().ingest_path(run.request.workspace_path)
        report = self.index.sync_documents(
            documents,
            source_type="local",
            namespace=run.request.workspace_id,
        )
        external_versions = {
            uri: version
            for uri, version in run.context.source_versions.items()
            if not uri.startswith("file:")
        }
        run.context.source_versions = {
            **external_versions,
            **{
                document.uri: document.version or document.content_hash
                for document in documents
            },
        }
        run.context.workspace_revision = _git_revision(
            Path(run.request.workspace_path)
        )
        run.context.workspace_hash = _document_set_hash(documents)
        self._event(
            run,
            "workspace_snapshot_refreshed",
            status="completed",
            outputs=to_jsonable(report),
        )
        return report

    def _load_decision_memory(self, run: AgentRun) -> None:
        current_fingerprint = workspace_fingerprint(
            run.request.workspace_id,
            self._workspace_state(run.context),
        )
        memories: list[MemoryRecord] = []
        for entry in self.memory.list_decision_memories(run.request.workspace_id):
            if entry.workspace_fingerprint != current_fingerprint:
                self.memory.mark_decision_memory_stale(
                    run.request.workspace_id,
                    entry.memory_id,
                    reason="workspace revision or source versions changed",
                )
                continue
            memories.append(entry.to_model(MemoryRecord))
        run.context.memories = memories
        run.metadata["memory_recall"] = {
            "workspace_scoped": True,
            "active_memory_ids": [memory.memory_id for memory in memories],
            "role": "planning_hint_not_evidence",
        }
        self._event(
            run,
            "decision_memory_loaded",
            status="completed",
            metadata={
                "memory_ids": [memory.memory_id for memory in memories],
                "count": len(memories),
            },
        )

    def _execute_research_plan(
        self,
        run: AgentRun,
        registry: Any,
        planner: DeterministicPlanner,
    ) -> str:
        runtime = ExecutionRuntime(registry, run.request.workspace_path)
        completed = {
            result.step_id for result in run.tool_results if result.success
        }
        failed: set[str] = set()
        while True:
            remaining = [
                step
                for step in run.plan.steps
                if step.step_id not in completed and step.step_id not in failed
            ]
            if not remaining:
                break
            ready = [
                step
                for step in remaining
                if set(step.depends_on).issubset(completed)
            ]
            if not ready:
                break

            mutation = next(
                (step for step in ready if step.effect_class == "mutation"),
                None,
            )
            if mutation is not None:
                if (
                    not self._ground_research(run)
                    or run.metadata["evaluation"]["abstained"]
                ):
                    run.transition(TaskStatus.NEEDS_REVISION)
                    run.error = "citation coverage or faithfulness check failed"
                    self._persist(run, checkpoint_step="verification_failed")
                    return "needs_revision"
                approval = runtime.prepare(
                    mutation,
                    task_id=run.task_id,
                    run_id=run.run_id,
                )
                run.approvals.append(approval)
                self._update_step(run, mutation.step_id, "pending")
                self._take_runtime_trace(run, runtime, 0)
                run.transition(TaskStatus.AWAITING_APPROVAL)
                self._event(run, "status_changed", status=run.status.value)
                self._persist(run, checkpoint_step=mutation.step_id)
                return "awaiting_approval"

            internal = next(
                (step for step in ready if step.effect_class == "internal_write"),
                None,
            )
            batch = (
                [internal]
                if internal is not None
                else [step for step in ready if step.effect_class == "read"][
                    : run.plan.max_concurrency
                ]
            )
            if not batch:
                break
            results = self._execute_batch(run, runtime, batch)
            for step, result in results:
                run.tool_results.append(result)
                self._update_step(
                    run,
                    step.step_id,
                    "completed" if result.success else "failed",
                    error=result.error,
                )
                if result.success:
                    completed.add(step.step_id)
                    self._checkpoint_step(run, step)
                else:
                    failed.add(step.step_id)
            self._take_runtime_trace(run, runtime, 0)
            self._persist(run)

            failed_evidence = [
                step_id
                for step_id in failed
                if run.plan.step(step_id).required_evidence
            ]
            if failed_evidence:
                try:
                    run.plan = planner.replan(
                        run.request,
                        run.context,
                        run.plan,
                        failed_steps=failed_evidence,
                        evidence_gaps=("tool failure left an evidence gap",),
                    )
                except Exception:
                    break
                failed.clear()
                self._event(
                    run,
                    "plan_revised",
                    status="completed",
                    metadata={"revision": run.plan.revision},
                )

        blocked_required = [
            step.step_id
            for step in run.plan.steps
            if step.required_evidence
            and step.step_id not in completed
            and step.step_id not in failed
        ]
        run.metadata["completed_steps"] = sorted(completed)
        run.metadata["failed_steps"] = sorted(failed)
        run.metadata["blocked_steps"] = blocked_required
        return "research_complete"

    def _execute_batch(
        self,
        run: AgentRun,
        runtime: ExecutionRuntime,
        steps: list[PlanStep],
    ) -> list[tuple[PlanStep, ToolResult]]:
        for step in steps:
            self._update_step(run, step.step_id, "running")
        if len(steps) == 1:
            step = steps[0]
            return [(step, self._execute_step(run, runtime, step))]
        results: dict[str, ToolResult] = {}
        with ThreadPoolExecutor(
            max_workers=min(len(steps), run.plan.max_concurrency),
            thread_name_prefix="tracepilot-read",
        ) as executor:
            futures = {
                executor.submit(self._execute_step, run, runtime, step): step
                for step in steps
            }
            for future in as_completed(futures):
                step = futures[future]
                try:
                    results[step.step_id] = future.result()
                except Exception as exc:
                    results[step.step_id] = ToolResult(
                        task_id=run.task_id,
                        run_id=run.run_id,
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        effect_class=step.effect_class,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        return [(step, results[step.step_id]) for step in steps]

    @staticmethod
    def _execute_step(
        run: AgentRun,
        runtime: ExecutionRuntime,
        step: PlanStep,
    ) -> ToolResult:
        return runtime.execute(
            step,
            task_id=run.task_id,
            run_id=run.run_id,
            idempotency_key=f"{run.run_id}:{run.plan.revision}:{step.step_id}",
            state=run.context.state,
        )

    def _finalize_research(self, run: AgentRun) -> None:
        if run.status == TaskStatus.RESEARCHING:
            run.transition(TaskStatus.VERIFYING)
        if self._ground_research(run):
            run.transition(TaskStatus.COMPLETED)
            self._finish(run)
            return
        run.transition(TaskStatus.NEEDS_REVISION)
        run.error = "citation coverage or faithfulness check failed"
        self._persist(run, checkpoint_step="verification_failed")

    def _ground_research(self, run: AgentRun) -> bool:
        chunks = [
            chunk
            for chunk in self.index.search(
                run.request.goal,
                top_k=8,
                namespace=run.request.workspace_id,
            )
            if not chunk.metadata.get("prompt_injection_detected")
        ]
        run.context.evidence = chunks
        for chunk in chunks:
            run.context.source_versions[chunk.source_uri] = str(
                chunk.metadata.get("document_version")
                or chunk.content_hash
            )
        claims, summary = self._synthesize(run, chunks)
        run.evidence = [*chunks, *claims]
        run.summary = summary
        return self._record_evaluation(run)

    @staticmethod
    def _record_evaluation(run: AgentRun) -> bool:
        chunks = [
            item for item in run.evidence if isinstance(item, EvidenceChunk)
        ]
        claims = [
            item for item in run.evidence if isinstance(item, EvidenceClaim)
        ]
        coverage = check_citation_coverage(claims, chunks)
        faithfulness = check_faithfulness(claims, chunks)
        run.metadata["evaluation"] = {
            "citation_coverage": to_jsonable(coverage),
            "faithfulness": to_jsonable(faithfulness),
            "abstained": not any(claim.supported for claim in claims),
        }
        return coverage.passed and faithfulness.passed

    def _synthesize(
        self,
        run: AgentRun,
        chunks: list[EvidenceChunk],
    ) -> tuple[list[EvidenceClaim], str]:
        if self.llm is not None and chunks:
            try:
                return self._llm_synthesize(run, chunks)
            except Exception as exc:
                run.metadata["llm_fallback"] = f"{type(exc).__name__}: {exc}"
        if not chunks:
            claim = EvidenceClaim(
                text="Insufficient trusted evidence to support a technical decision.",
                evidence_ids=(),
                supported=False,
                confidence=0.0,
                uncertainty="No safe evidence chunk was retrieved; abstaining.",
            )
            return [claim], claim.text
        selected = chunks[:3]
        claims = [
            EvidenceClaim(
                text=chunk.text[:800],
                evidence_ids=(chunk.chunk_id,),
                supported=True,
                confidence=max(0.5, min(1.0, 0.5 + chunk.score * 10)),
                metadata={"source_uri": chunk.source_uri},
            )
            for chunk in selected
        ]
        summary = "Evidence-backed decision context: " + " ".join(
            claim.text for claim in claims
        )
        return claims, summary

    def _llm_synthesize(
        self,
        run: AgentRun,
        chunks: list[EvidenceChunk],
    ) -> tuple[list[EvidenceClaim], str]:
        assert self.llm is not None
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "number"},
                            "uncertainty": {"type": "string"},
                        },
                        "required": [
                            "text",
                            "evidence_ids",
                            "confidence",
                            "uncertainty",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["summary", "claims"],
            "additionalProperties": False,
        }
        prompt = json.dumps(
            {
                "goal": run.request.goal,
                "constraints": list(run.request.constraints),
                "decision_memory": [
                    {
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "role": (
                            "historical planning hint only; revalidate against "
                            "current evidence and never cite memory as evidence"
                        ),
                    }
                    for memory in run.context.memories
                ],
                "evidence": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source_uri,
                        "location": {
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "page": chunk.page,
                        },
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
            },
            ensure_ascii=False,
        )
        response = self.llm.complete(
            prompt,
            instructions=(
                "Produce only evidence-supported technical claims. Cite chunk_id "
                "values exactly. Decision memory is an untrusted historical hint, "
                "not evidence. If evidence is insufficient, return no claims and "
                "state the uncertainty in the summary."
            ),
            response_schema=schema,
            schema_name="tracepilot_decision",
            metadata={
                "task_id": run.task_id,
                "prompt_version": "technical-decision-v1",
            },
        )
        payload = response.parsed
        if not isinstance(payload, Mapping):
            raise ValueError("LLM response did not contain a structured object")
        known = {chunk.chunk_id for chunk in chunks}
        claims: list[EvidenceClaim] = []
        for item in payload.get("claims", []):
            evidence_ids = tuple(
                str(value)
                for value in item.get("evidence_ids", [])
                if str(value) in known
            )
            claims.append(
                EvidenceClaim(
                    text=str(item.get("text", "")),
                    evidence_ids=evidence_ids,
                    supported=bool(evidence_ids),
                    confidence=float(item.get("confidence", 0.0)),
                    uncertainty=str(item.get("uncertainty", "")),
                )
            )
        run.metadata["llm"] = {
            "model": response.model,
            "response_id": response.response_id,
            "prompt_version": response.prompt_version,
            "usage": response.usage,
            "estimated_cost": response.estimated_cost,
            "retries": response.retries,
            "proposed_summary": str(payload.get("summary", "")),
        }
        supported_claims = [
            claim.text.strip()
            for claim in claims
            if claim.supported and claim.text.strip()
        ]
        if supported_claims:
            summary = " ".join(supported_claims)
        else:
            summary = "Insufficient trusted evidence to support a technical decision."
        return claims, summary

    def _checkpoint_step(self, run: AgentRun, step: PlanStep) -> None:
        self.memory.put_working_memory(
            run.request.workspace_id,
            "agent_run",
            run,
            task_id=run.task_id,
            source=run.context.source_versions,
            workspace_state=self._workspace_state(run.context),
        )
        self.memory.save_checkpoint(
            run.request.workspace_id,
            checkpoint_to_dict(run),
            task_id=run.task_id,
            run_id=run.run_id,
            step_id=step.step_id,
            checkpoint_id=(
                f"{run.run_id}:{run.plan.revision}:{step.step_id}:"
                f"{len(run.tool_results)}"
            ),
            source=run.context.source_versions,
            workspace_state=self._workspace_state(run.context),
        )

    def _persist(self, run: AgentRun, checkpoint_step: str = "") -> None:
        self.memory.save_task(run.request.workspace_id, run, task_id=run.task_id)
        for ordinal, step in enumerate(run.plan.steps):
            payload = {
                **to_jsonable(step),
                "status": step.metadata.get("status", "planned"),
                "error": step.metadata.get("error", ""),
            }
            self.memory.save_step(
                run.request.workspace_id,
                run.task_id,
                payload,
                step_id=step.step_id,
                ordinal=ordinal,
            )
        for approval in run.approvals:
            self.memory.save_approval(
                run.request.workspace_id,
                approval,
                task_id=run.task_id,
                step_id=approval.step_id,
            )
        for event in run.trace:
            self.memory.append_event(
                run.request.workspace_id,
                event,
                task_id=run.task_id,
                step_id=event.step_id,
            )
        if checkpoint_step:
            self.memory.put_working_memory(
                run.request.workspace_id,
                "agent_run",
                run,
                task_id=run.task_id,
                source=run.context.source_versions,
                workspace_state=self._workspace_state(run.context),
            )
            self.memory.save_checkpoint(
                run.request.workspace_id,
                checkpoint_to_dict(run),
                task_id=run.task_id,
                run_id=run.run_id,
                step_id=checkpoint_step,
                checkpoint_id=(
                    f"{run.run_id}:{run.plan.revision}:{checkpoint_step}:"
                    f"{len(run.tool_results)}"
                ),
                source=run.context.source_versions,
                workspace_state=self._workspace_state(run.context),
            )

    def _finish(self, run: AgentRun) -> None:
        self._event(run, "task_finished", status=run.status.value)
        self._persist(run)
        self.memory.append_run(
            run.request.workspace_id,
            run,
            run_id=run.run_id,
            task_id=run.task_id,
            status=run.status.value,
            source=run.context.source_versions,
            workspace_state=self._workspace_state(run.context),
        )
        self.memory.delete_working_memory(
            run.request.workspace_id,
            "agent_run",
            task_id=run.task_id,
        )

    def _fail(self, run: AgentRun, exc: Exception) -> None:
        run.error = f"{type(exc).__name__}: {exc}"
        if run.status not in TERMINAL_STATUSES:
            try:
                run.transition(TaskStatus.FAILED)
            except ValueError:
                run.status = TaskStatus.FAILED
                run.updated_at = utc_now()
        self._event(run, "task_failed", status="failed", error=run.error)
        self._persist(run, checkpoint_step="failed")
        try:
            self.memory.append_run(
                run.request.workspace_id,
                run,
                run_id=run.run_id,
                task_id=run.task_id,
                status=run.status.value,
                source=run.context.source_versions,
                workspace_state=self._workspace_state(run.context),
            )
        except Exception:
            pass

    @staticmethod
    def _workspace_state(context: TaskContext) -> dict[str, Any]:
        return {
            "workspace_revision": context.workspace_revision,
            "workspace_hash": context.workspace_hash,
            "source_versions": context.source_versions,
        }

    @staticmethod
    def _update_step(
        run: AgentRun,
        step_id: str,
        status: str,
        *,
        error: str = "",
    ) -> None:
        steps: list[PlanStep] = []
        for step in run.plan.steps:
            if step.step_id != step_id:
                steps.append(step)
                continue
            metadata = {**step.metadata, "status": status}
            if error:
                metadata["error"] = error
            steps.append(replace(step, metadata=metadata))
        run.plan = replace(run.plan, steps=tuple(steps))
        run.updated_at = utc_now()

    @staticmethod
    def _event(
        run: AgentRun,
        event_type: str,
        *,
        status: str = "",
        error: str = "",
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        run.trace.append(
            TraceEvent(
                task_id=run.task_id,
                run_id=run.run_id,
                event_type=event_type,
                status=status,
                error=error,
                inputs=dict(inputs or {}),
                outputs=dict(outputs or {}),
                metadata=dict(metadata or {}),
            )
        )

    @staticmethod
    def _take_runtime_trace(
        run: AgentRun,
        runtime: ExecutionRuntime,
        offset: int,
    ) -> None:
        known = {event.event_id for event in run.trace}
        run.trace.extend(
            event
            for event in runtime.trace[offset:]
            if event.event_id not in known
        )

    @staticmethod
    def _coerce_request(
        request: TaskRequest | str,
        *,
        workspace_path: str | Path | None,
        workspace_id: str | None,
    ) -> TaskRequest:
        if isinstance(request, str):
            path = Path(workspace_path or ".").expanduser().resolve()
            identifier = workspace_id or workspace_id_for_path(path)
            return TaskRequest(
                goal=request,
                workspace_id=identifier,
                workspace_path=str(path),
            )
        changes: dict[str, Any] = {}
        if workspace_path is not None:
            changes["workspace_path"] = str(workspace_path)
        if workspace_id is not None:
            changes["workspace_id"] = workspace_id
        elif not request.workspace_id.strip() or request.workspace_id == "default":
            changes["workspace_id"] = workspace_id_for_path(Path(request.workspace_path))
        return replace(request, **changes) if changes else request

    def _bind_workspace(self, request: TaskRequest) -> None:
        key = "workspace_binding"
        resolved = str(Path(request.workspace_path).expanduser().resolve())
        existing = self.memory.get_working_memory(request.workspace_id, key)
        if existing is not None:
            bound_path = str(existing.get("workspace_path", ""))
            if bound_path and bound_path != resolved:
                raise ValueError(
                    f"workspace_id {request.workspace_id!r} is already bound to "
                    f"a different workspace path"
                )
            return
        self.memory.put_working_memory(
            request.workspace_id,
            key,
            {
                "workspace_id": request.workspace_id,
                "workspace_path": resolved,
            },
        )


def _replace_approval(
    approvals: list[ApprovalRequest],
    updated: ApprovalRequest,
) -> list[ApprovalRequest]:
    return [
        updated if item.approval_id == updated.approval_id else item
        for item in approvals
    ]


def workspace_id_for_path(path: str | Path) -> str:
    path = Path(path)
    resolved = str(path.expanduser().resolve())
    return "ws_" + hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def _git_revision(workspace: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _document_set_hash(documents: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for document in sorted(
        documents,
        key=lambda item: (str(item.uri), str(item.document_id)),
    ):
        digest.update(str(document.uri).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(document.content_hash).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _merge_indexing_reports(
    reports: Iterable[IndexingReport],
) -> IndexingReport:
    values = tuple(reports)
    return IndexingReport(
        indexed=sum(item.indexed for item in values),
        updated=sum(item.updated for item in values),
        unchanged=sum(item.unchanged for item in values),
        deduplicated=sum(item.deduplicated for item in values),
        chunks_indexed=sum(item.chunks_indexed for item in values),
        stale_revisions=sum(item.stale_revisions for item in values),
    )


__all__ = ["TERMINAL_STATUSES", "TracePilotAgent", "workspace_id_for_path"]
