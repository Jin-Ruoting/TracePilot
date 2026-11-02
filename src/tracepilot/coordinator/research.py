from __future__ import annotations

import json
import fcntl
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..experiments.runner import LocalRunner
from ..experiments.preflight import preflight
from ..research.models import ResearchSpec, canonical, digest
from ..research.store import ResearchStore, TERMINAL
from ..routing import CascadeRouter
from ..workers.coding import CodingWorker, file_manifest, snapshot_workspace
from ..workers.review import ReviewWorker, validate_metrics


class ResearchCoordinator:
    """Model-independent control plane with an explicit experiment execution grant."""

    def __init__(self, output_root: str | Path, *, artifacts_root: str | Path | None = None,
                 router: CascadeRouter | None = None) -> None:
        self.root = Path(output_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = ResearchStore(self.root / "research.sqlite3")
        self.artifacts_root = Path(artifacts_root).expanduser().resolve() if artifacts_root else None
        self.router = router or CascadeRouter()
        self.runners: dict[str, Any] = {}

    def close(self) -> None:
        self.store.close()

    def create(self, spec: ResearchSpec, *, backend: dict[str, Any] | None = None,
               allow_execution: bool = False) -> dict[str, Any]:
        if not allow_execution:
            raise PermissionError("creating an executable project requires an explicit experiment grant")
        config = dict(backend or {"kind": "local", "python": sys.executable})
        if config.get("kind") != "local":
            raise ValueError("execute on the local host; use the GitHub-synced remote launcher for SSH hosts")
        config.setdefault("python", sys.executable)
        config["cuda_visible_devices"] = os.environ.get("TRACEPILOT_GPU_UUID", "")
        config["python"] = str(Path(shutil.which(config["python"]) or config["python"]).expanduser().absolute())
        environment = {"imports_checked": False}
        if spec.required_modules:
            environment = preflight(config["python"], list(spec.required_modules))
            environment["imports_checked"] = True
        project = self.root / spec.project_id
        source = project / "source"
        manifest = snapshot_workspace(Path(spec.workspace), source)
        definition = {
            "spec": spec.to_dict(), "backend": config, "source_manifest": manifest,
            "artifacts_root": str(self.artifacts_root) if self.artifacts_root else None,
            "source_hash": digest(manifest), "environment": environment, "grant": {
                "scope": "isolated experiment copies", "budget": asdict(spec.budget),
                "network_and_process_isolation": "host-managed; not an OS sandbox",
                "cuda_visible_devices": config["cuda_visible_devices"],
            },
        }
        fresh = self.store.create_project(spec.project_id, definition)
        (project / "definition.json").write_text(canonical(definition) + "\n")
        worker = CodingWorker()
        variants = project / "variants"
        variants.mkdir(exist_ok=True)
        for variant in spec.variants:
            target = variants / variant.variant_id
            provenance = variants / (variant.variant_id + ".provenance.json")
            if target.exists():
                if not provenance.exists():
                    raise ValueError("incomplete variant preparation; create a new project revision")
                record = json.loads(provenance.read_text())
                if file_manifest(target) != record["files"]:
                    raise ValueError("prepared variant was modified")
            else:
                record = worker.prepare(source, target, variant)
            for dataset in spec.datasets:
                for seed in spec.seeds:
                    job = {
                        "project_id": spec.project_id, "variant": variant.variant_id,
                        "hypothesis_id": spec.hypothesis.hypothesis_id, "dataset": dataset, "seed": seed,
                        "split_seed": spec.split_seed, "partition": spec.partition, "metric": spec.metric,
                        "command": list(spec.command), "python": config["python"],
                        "timeout_seconds": spec.budget.run_timeout_seconds, "threads": spec.budget.threads_per_run,
                        "cuda_visible_devices": config["cuda_visible_devices"],
                        "code_hash": record["code_hash"], "code_manifest": record["files"],
                        "definition_digest": digest(definition),
                    }
                    job["job_id"] = "run_" + digest(job)[:24]
                    if self.artifacts_root:
                        job["artifact_directory"] = str(self.artifacts_root / spec.project_id / "runs" / job["job_id"])
                    self.store.enqueue(spec.project_id, job)
        if fresh:
            self._route(spec.project_id, {"kind": "project_created", "hypothesis_origin": spec.hypothesis.origin})
            self._route(spec.project_id, {"kind": "experiment_spec_ready"})
        return self.status(spec.project_id)

    def _runner(self, project_id: str) -> Any:
        if project_id not in self.runners:
            self.runners[project_id] = LocalRunner()
        return self.runners[project_id]

    def _directory(self, job: dict[str, Any]) -> Path:
        return self.root / job["project_id"] / "runs" / job["job_id"]

    def report_directory(self, project_id: str) -> Path:
        root = self.store.project(project_id)["definition"].get("artifacts_root")
        return (Path(root) if root else self.root) / project_id

    def _route(self, project_id: str, event: dict[str, Any]) -> None:
        events = self.store.events(project_id)
        decision = self.router.route(event, state_version=events[-1]["seq"] if events else 0)
        self.store.record_decision(project_id, dict(event=event, decision=decision.to_dict()))

    def tick(self, project_id: str) -> dict[str, Any]:
        definition = self.store.project(project_id)["definition"]
        spec = ResearchSpec.from_dict(definition["spec"])
        runner = self._runner(project_id)
        for row in self.store.jobs(project_id):
            if row["status"] not in {"starting", "running"}:
                continue
            job, directory = row["definition"], self._directory(row["definition"])
            try:
                if self.store.cancellation_requested(row["id"]):
                    self._prepare_job(job, directory)
                    runner.cancel(job, directory)
                if row["status"] == "starting":
                    self._prepare_job(job, directory)
                    runner.submit(job, directory)
                outcome = runner.poll(job, directory)
                if outcome["status"] == "running":
                    self.store.transition(row["id"], "running")
                elif outcome["status"] in TERMINAL:
                    if outcome["status"] == "succeeded":
                        try:
                            outcome["metrics"] = validate_metrics(job, outcome, directory)
                        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                            outcome["status"], outcome["error"] = "invalid", str(exc)
                    if outcome["status"] == "lost":
                        outcome["elapsed_seconds"] = job["timeout_seconds"]
                    self.store.transition(row["id"], outcome["status"], outcome)
                    self._route(project_id, {"kind": "job_" + outcome["status"], "job_id": row["id"]})
            except (ConnectionError, TimeoutError, OSError) as exc:
                self.store.add_event(project_id, "transport_unavailable",
                                     {"error_type": type(exc).__name__, "detail": str(exc)}, row["id"])
        rows = self.store.jobs(project_id)
        if any(row["status"] == "lost" for row in rows):
            return self.status(project_id)
        for row in rows:
            if row["status"] != "queued":
                continue
            if self.store.claim(row["id"], spec.budget.max_concurrent, spec.budget.total_run_seconds):
                job, directory = row["definition"], self._directory(row["definition"])
                try:
                    self._prepare_job(job, directory)
                    runner.submit(job, directory)
                except (ConnectionError, TimeoutError, OSError) as exc:
                    self.store.add_event(project_id, "transport_unavailable",
                                         {"error_type": type(exc).__name__, "detail": str(exc)}, row["id"])
                except Exception as exc:
                    self.store.transition(row["id"], "failed", {
                        "status": "failed", "elapsed_seconds": 0, "error": str(exc), "exit_code": None})
        return self.status(project_id)

    def _prepare_job(self, job: dict[str, Any], directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "preparation.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            target = directory / "work"
            if not target.exists():
                temporary = directory / "preparing"
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(self.root / job["project_id"] / "variants" / job["variant"], temporary)
                temporary.rename(target)
            definition_path = directory / "job.json"
            if definition_path.exists() and json.loads(definition_path.read_text()) != job:
                raise ValueError("persisted job parameters were modified")
            if not definition_path.exists():
                from ..experiments.supervisor import atomic_json
                atomic_json(definition_path, job)

    def status(self, project_id: str) -> dict[str, Any]:
        project = self.store.project(project_id)
        jobs = self.store.jobs(project_id)
        counts: dict[str, int] = {}
        for row in jobs:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        done = bool(jobs) and all(row["status"] in TERMINAL for row in jobs)
        state = "completed" if done and counts.get("succeeded") == len(jobs) else (
            "completed_with_failures" if done else "active")
        if not done and counts.get("lost"):
            state = "needs_attention"
        if not done and counts.get("queued") and not (counts.get("starting") or counts.get("running")):
            budget = project["definition"]["spec"]["budget"]
            used = sum((j["result"] or {}).get("elapsed_seconds", 0) for j in jobs)
            if state != "needs_attention" and used + budget["run_timeout_seconds"] > budget["total_run_seconds"]:
                state = "budget_exhausted"
        self.store.set_project_status(project_id, state)
        return {"project_id": project_id, "status": state, "counts": counts, "total": len(jobs),
                "directory": str(self.root / project_id)}

    def wait(self, project_id: str, *, timeout: float = 600, interval: float = 0.25) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            status = self.tick(project_id)
            if status["status"] in {"completed", "completed_with_failures", "budget_exhausted", "needs_attention"}:
                return status
            if time.monotonic() >= deadline:
                return dict(status, wait_timed_out=True)
            time.sleep(interval)

    def cancel(self, project_id: str) -> dict[str, Any]:
        runner = self._runner(project_id)
        for row in self.store.jobs(project_id):
            if row["status"] == "queued":
                self.store.cancel_queued(row["id"])
            elif row["status"] in {"starting", "running"}:
                self.store.request_cancel(row["id"])
                directory = self._directory(row["definition"])
                self._prepare_job(row["definition"], directory)
                try:
                    runner.cancel(row["definition"], directory)
                except (ConnectionError, TimeoutError, OSError) as exc:
                    self.store.add_event(project_id, "cancellation_pending_transport",
                                         {"error_type": type(exc).__name__, "detail": str(exc)}, row["id"])
        return self.status(project_id)

    def review(self, project_id: str) -> dict[str, Any]:
        project = self.store.project(project_id)
        spec = ResearchSpec.from_dict(project["definition"]["spec"])
        report = ReviewWorker().review(spec, self.store.jobs(project_id))
        routes = [row["payload"]["decision"] for row in self.store.decisions(project_id)
                  if "decision" in row["payload"]]
        report.update(backend=project["definition"]["backend"], source_hash=project["definition"]["source_hash"],
                      budget=project["definition"]["spec"]["budget"],
                      event_count=len(self.store.events(project_id)),
                      model_usage={"generation_calls": 0, "note": "Supplied experiment spec; no generative model called."},
                      routing={"total": len(routes),
                               "rule": sum(row["source"] == "rule" for row in routes),
                               "decision_model": sum(row["source"] == "decision_model" for row in routes),
                               "fallback": sum(row["source"] == "fallback" for row in routes)})
        self.store.record_decision(project_id, {"kind": "pilot_review", "report": report})
        return report
