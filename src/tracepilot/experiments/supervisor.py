"""Standalone POSIX supervisor; also copied to SSH hosts (Python 3.8+).

The file lock is held for the complete child lifetime. Repeated submissions
cannot start a second child. Cancellation uses a job-local request file, not
a persisted PID that could have been reused by an unrelated process.
"""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inspect_job(root):
    root = Path(root)
    if (root / "status.json").exists():
        return json.loads((root / "status.json").read_text())
    running = False
    with (root / "execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            running = True
    if running:
        return {"status": "running"}
    # The supervisor may have published completion between our first check and lock acquisition.
    if (root / "status.json").exists():
        return json.loads((root / "status.json").read_text())
    if (root / "started.json").exists():
        return {"status": "lost", "exit_code": None, "elapsed_seconds": 0,
                "error": "supervisor disappeared; execution outcome is unknown; no automatic retry"}
    return {"status": "not_started"}


def kill_group(child, sig):
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass


def execute(request_path):
    root = Path(request_path).resolve().parent
    with (root / "execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (root / "status.json").exists() or (root / "started.json").exists():
            return
        request = json.loads(Path(request_path).read_text())
        artifacts = Path(request.get("artifact_directory", root))
        artifacts.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        interrupted = []
        signal.signal(signal.SIGTERM, lambda *_: interrupted.append(True))
        signal.signal(signal.SIGINT, lambda *_: interrupted.append(True))
        atomic_json(root / "started.json", {
            "job_id": request["job_id"], "pid": os.getpid(), "started_at_unix": time.time(),
            "request_sha256": hashlib.sha256(Path(request_path).read_bytes()).hexdigest(),
        })
        child = None
        state = "failed"
        error = ""
        exit_code = None
        try:
            for relative, expected in request.get("code_manifest", {}).items():
                source = Path(request["workspace"]) / relative
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError("input code differs from the frozen manifest: " + relative)
            with (artifacts / "stdout.log").open("wb") as stdout, (artifacts / "stderr.log").open("wb") as stderr:
                if (root / "cancel.request").exists():
                    state = "cancelled"
                else:
                    environment = os.environ.copy()
                    environment.update(request.get("environment", {}))
                    child = subprocess.Popen(request["argv"], cwd=request["workspace"], env=environment,
                                             stdout=stdout, stderr=stderr, start_new_session=True)
                    last_heartbeat = 0
                    while child.poll() is None:
                        elapsed = time.monotonic() - started
                        if interrupted or (root / "cancel.request").exists():
                            state = "cancelled"
                            break
                        if elapsed >= request["timeout_seconds"]:
                            state = "timed_out"
                            break
                        if elapsed - last_heartbeat >= 1:
                            atomic_json(root / "heartbeat.json", {"elapsed_seconds": elapsed, "at_unix": time.time()})
                            last_heartbeat = elapsed
                        time.sleep(0.05)
                    if child.poll() is None:
                        kill_group(child, signal.SIGTERM)
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            kill_group(child, signal.SIGKILL)
                            child.wait()
                    # A child can exit before its own subprocesses. Bound the whole process group.
                    kill_group(child, signal.SIGKILL)
                    exit_code = child.returncode
                    if state not in {"cancelled", "timed_out"}:
                        state = "succeeded" if exit_code == 0 else "failed"
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            state = "failed"
            if child is not None:
                kill_group(child, signal.SIGKILL)
                child.wait()
        result = {
            "job_id": request["job_id"], "status": state, "exit_code": exit_code,
            "elapsed_seconds": time.monotonic() - started, "error": error,
            "host": platform.node(), "supervisor_python": sys.version.split()[0],
            "finished_at_unix": time.time(),
        }
        for relative, expected in request.get("code_manifest", {}).items():
            source = Path(request["workspace"]) / relative
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                result["status"] = "invalid"
                result["error"] = "experiment modified frozen input code: " + relative
        metrics = Path(request["metrics_path"])
        try:
            metrics.resolve().relative_to((artifacts if request.get("artifact_directory") else Path(request["workspace"])).resolve())
            if metrics.is_file() and metrics.stat().st_size <= 1_000_000:
                result["metrics_sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
        except ValueError:
            pass
        atomic_json(root / "status.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    execute(args.request)


if __name__ == "__main__":
    main()
