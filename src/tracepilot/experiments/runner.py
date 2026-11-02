from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..research.models import canonical
from . import supervisor


def render_command(command: list[str], values: dict[str, Any]) -> list[str]:
    result = []
    for item in command:
        for key, value in values.items():
            item = item.replace("{" + key + "}", str(value))
        result.append(item)
    return result


def request_for(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"], "timeout_seconds": job["timeout_seconds"],
        "code_manifest": job.get("code_manifest", {}),
        "environment": {
            "OMP_NUM_THREADS": str(job["threads"]), "OPENBLAS_NUM_THREADS": str(job["threads"]),
            "MKL_NUM_THREADS": str(job["threads"]), "NUMEXPR_NUM_THREADS": str(job["threads"]),
            "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": job.get("cuda_visible_devices", ""),
        },
        "command": job["command"],
    }


def values_for(job: dict[str, Any]) -> dict[str, Any]:
    return {name: job[name] for name in
            ("python", "dataset", "seed", "split_seed", "partition", "variant")}


class LocalRunner:
    def __init__(self) -> None:
        self.children: dict[str, subprocess.Popen[Any]] = {}

    def submit(self, job: dict[str, Any], directory: Path) -> None:
        request = request_for(job)
        artifacts = Path(job["artifact_directory"]) if job.get("artifact_directory") else directory
        artifacts.mkdir(parents=True, exist_ok=True)
        metrics_directory = artifacts if job.get("artifact_directory") else directory / "work"
        values = dict(values_for(job), workspace=str(directory / "work"),
                      output=str(metrics_directory / "metrics.json"))
        request["argv"] = render_command(request.pop("command"), values)
        request.update(workspace=values["workspace"], metrics_path=values["output"])
        request["artifact_directory"] = str(artifacts)
        path = directory / "request.json"
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError("local request differs from the frozen job")
        if not path.exists():
            supervisor.atomic_json(path, request)
        if supervisor.inspect_job(directory)["status"] != "not_started":
            return
        previous = self.children.get(job["job_id"])
        if previous is not None and previous.poll() is None:
            return
        with (directory / "launcher.log").open("ab") as log:
            self.children[job["job_id"]] = subprocess.Popen(
                [sys.executable, str(Path(supervisor.__file__).resolve()), str(path)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            )

    def poll(self, job: dict[str, Any], directory: Path) -> dict[str, Any]:
        state = supervisor.inspect_job(directory)
        child = self.children.get(job["job_id"])
        if child is not None and child.poll() is not None:
            self.children.pop(job["job_id"], None)
            if state["status"] == "not_started":
                return {"status": "failed", "elapsed_seconds": 0, "exit_code": child.returncode,
                        "error": (directory / "launcher.log").read_text()[-4000:]}
        if state["status"] == "not_started" and (directory / "request.json").exists():
            if time.time() - (directory / "request.json").stat().st_mtime > 30:
                return {"status": "lost", "elapsed_seconds": 0, "exit_code": None,
                        "error": "supervisor did not acknowledge startup"}
        return state

    def cancel(self, job: dict[str, Any], directory: Path) -> None:
        (directory / "cancel.request").touch()
