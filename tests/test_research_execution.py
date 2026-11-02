from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tracepilot.coordinator import ResearchCoordinator
from tracepilot.experiments import supervisor
from tracepilot.research.models import ExperimentBudget, Hypothesis, ResearchSpec, Variant
from tracepilot.research.reporting import write_report
from tracepilot.workers.coding import file_manifest


TRAIN = r'''
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from behavior import SCORE
p = argparse.ArgumentParser()
for name in ("dataset", "seed", "split-seed", "output", "mode"):
    p.add_argument("--" + name, required=True)
a = p.parse_args()
with Path("executions.txt").open("a") as f:
    f.write("once\n")
if a.mode == "wait":
    time.sleep(30)
elif a.mode == "child":
    child = subprocess.Popen([sys.executable, "-c",
        "import signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n Path('child.heartbeat').write_text(str(time.monotonic()))\n time.sleep(0.02)\n"])
    Path("child.pid").write_text(str(child.pid))
    time.sleep(30)
elif a.mode == "fail":
    print("intentional fixture failure", file=sys.stderr)
    sys.exit(7)
elif a.mode == "mutate":
    Path("behavior.py").write_text("SCORE = 999\n")
elif a.mode == "missing":
    sys.exit(0)
elif a.mode == "brief":
    time.sleep(0.2)
value = {
    "schema_version": 1, "dataset": a.dataset, "seed": int(a.seed),
    "split_seed": int(a.split_seed), "partition": "validation", "split_hash": "a" * 64,
    "metrics": {"accuracy": SCORE + int(a.seed) / 1000},
    "protocol": {"fixture": True, "budget": 1, "test_evaluated": False},
}
if a.mode == "nan":
    value["metrics"]["accuracy"] = float("nan")
if a.mode == "wrong_seed":
    value["seed"] += 1
Path(a.output).write_text(json.dumps(value))
print("fixture completed")
'''




class ResearchExecutionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tracepilot research ")
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "train.py").write_text(TRAIN)
        (self.source / "behavior.py").write_text("SCORE = 0.2\n")
        self.coordinators = []

    def tearDown(self):
        for coordinator in self.coordinators:
            try:
                for row in coordinator.store.db.execute("SELECT id FROM research_projects").fetchall():
                    coordinator.cancel(row["id"])
                    coordinator.wait(row["id"], timeout=4, interval=0.03)
                for runner in coordinator.runners.values():
                    for child in getattr(runner, "children", {}).values():
                        child.wait(timeout=3)
                coordinator.close()
            except Exception:
                pass
        self.temp.cleanup()

    def coordinator(self):
        coordinator = ResearchCoordinator(self.root / "out")
        self.coordinators.append(coordinator)
        return coordinator

    def spec(self, mode="ok", timeout=3, seeds=(1, 2), concurrent=2, total=30, project_id="test"):
        patch_data = {"path": "behavior.py", "expected_sha256": file_manifest(self.source)["behavior.py"],
                      "content": "SCORE = 0.6\n"}
        return ResearchSpec(
            project_id=project_id, goal="Exercise actual child processes with synthetic test fixtures",
            workspace=str(self.source),
            command=("{python}", "train.py", "--dataset", "{dataset}", "--seed", "{seed}",
                     "--split-seed", "{split_seed}", "--output", "{output}", "--mode", mode),
            datasets=("fixture",), seeds=seeds,
            variants=(Variant("baseline"), Variant("candidate", (patch_data,))),
            hypothesis=Hypothesis("fixture", "A fixture comparison", "Controlled constants", "Delta below zero",
                                  origin="predefined_benchmark"),
            budget=ExperimentBudget(max_runs=2 * len(seeds), max_concurrent=concurrent,
                                    run_timeout_seconds=timeout, total_run_seconds=total),
        )

    def start(self, **kwargs):
        c = self.coordinator()
        spec = self.spec(**kwargs)
        c.create(spec, allow_execution=True)
        return c, spec

    def test_execution_requires_explicit_grant(self):
        with self.assertRaises(PermissionError):
            self.coordinator().create(self.spec())

    def test_code_and_bulk_artifacts_stay_in_separate_roots_after_resume(self):
        c = ResearchCoordinator(self.root / "out", artifacts_root=self.root / "bulk")
        self.coordinators.append(c)
        spec = self.spec(seeds=(1,))
        c.create(spec, allow_execution=True)
        c.close()
        self.coordinators.remove(c)
        resumed = self.coordinator()
        self.assertEqual(resumed.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"succeeded": 2})
        paths = write_report(resumed, spec.project_id)
        self.assertTrue(Path(paths["json"]).is_relative_to(self.root / "bulk"))
        for row in resumed.store.jobs(spec.project_id):
            code = resumed._directory(row["definition"]) / "work"
            artifacts = Path(row["definition"]["artifact_directory"])
            self.assertTrue((code / "train.py").exists())
            self.assertFalse((code / "metrics.json").exists())
            self.assertTrue((artifacts / "metrics.json").exists())
            self.assertTrue((artifacts / "stdout.log").exists())
            self.assertFalse((artifacts / "train.py").exists())

    def test_broken_import_preflight_does_not_enqueue_any_jobs(self):
        c = self.coordinator()
        spec = replace(self.spec(), required_modules=("tracepilot_missing_fixture_dependency",))
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            c.create(spec, allow_execution=True)
        self.assertEqual(c.store.jobs(spec.project_id), [])
        self.assertFalse((c.root / spec.project_id).exists())

    def test_relative_python_path_survives_isolated_working_directory(self):
        c = self.coordinator()
        spec = self.spec(seeds=(1,))
        c.create(spec, backend={"kind": "local", "python": os.path.relpath(sys.executable)}, allow_execution=True)
        self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"succeeded": 2})

    def test_real_commands_isolated_report_and_idempotent_resume(self):
        c, spec = self.start()
        self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"succeeded": 4})
        before = c.store.jobs(spec.project_id)
        self.assertEqual((self.source / "behavior.py").read_text(), "SCORE = 0.2\n")
        report_paths = write_report(c, spec.project_id)
        report = json.loads(Path(report_paths["json"]).read_text())
        self.assertEqual(report["successful_runs"], 4)
        self.assertAlmostEqual(report["comparisons"][0]["oriented_delta_mean"], 0.4)
        self.assertEqual(report["comparisons"][0]["verdict"], "promising")
        c.create(spec, allow_execution=True)
        c.tick(spec.project_id)
        self.assertEqual(c.store.jobs(spec.project_id), before)
        for row in before:
            work = c._directory(row["definition"]) / "work"
            self.assertEqual((work / "executions.txt").read_text(), "once\n")
        self.assertTrue(Path(report_paths["html"]).is_file())

    def test_controller_restart_does_not_repeat_completed_side_effects(self):
        c, spec = self.start(mode="brief", concurrent=1)
        c.tick(spec.project_id)
        original_runner = c.runners[spec.project_id]
        c.close()
        self.coordinators.remove(c)
        resumed = self.coordinator()
        self.assertEqual(resumed.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"succeeded": 4})
        for child in original_runner.children.values():
            child.wait(timeout=3)
        for row in resumed.store.jobs(spec.project_id):
            path = resumed._directory(row["definition"]) / "work" / "executions.txt"
            self.assertEqual(path.read_text(), "once\n")

    def test_duplicate_supervisors_execute_once(self):
        c, spec = self.start(mode="brief", seeds=(1,), concurrent=1)
        c.tick(spec.project_id)
        job = c.store.jobs(spec.project_id)[0]
        request = c._directory(job["definition"]) / "request.json"
        twins = [subprocess.Popen([sys.executable, supervisor.__file__, str(request)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(3)]
        for child in twins:
            self.assertEqual(child.wait(timeout=5), 0)
        c.wait(spec.project_id, timeout=10, interval=0.03)
        self.assertEqual((request.parent / "work" / "executions.txt").read_text(), "once\n")

    def test_failure_logs_and_exit_code_are_preserved(self):
        c, spec = self.start(mode="fail", seeds=(1,))
        self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"failed": 2})
        for row in c.store.jobs(spec.project_id):
            self.assertEqual(row["result"]["exit_code"], 7)
            self.assertIn("intentional fixture failure", (c._directory(row["definition"]) / "stderr.log").read_text())
        self.assertEqual(c.review(spec.project_id)["comparisons"][0]["verdict"], "inconclusive")

    def test_timeout_and_cancel_are_terminal(self):
        c, spec = self.start(mode="wait", timeout=0.2, seeds=(1,))
        self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"timed_out": 2})
        other = self.spec(mode="wait", project_id="cancel", seeds=(1, 2), concurrent=1)
        c.create(other, allow_execution=True)
        c.tick(other.project_id)
        c.cancel(other.project_id)
        self.assertEqual(c.wait(other.project_id, timeout=10, interval=0.03)["counts"], {"cancelled": 4})

    def test_timeout_reaps_subprocesses_that_ignore_termination(self):
        c, spec = self.start(mode="child", timeout=0.3, seeds=(1,))
        self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"timed_out": 2})
        for row in c.store.jobs(spec.project_id):
            work = c._directory(row["definition"]) / "work"
            heartbeat = (work / "child.heartbeat").read_text()
            time.sleep(0.12)
            if (work / "child.heartbeat").read_text() != heartbeat:
                os.kill(int((work / "child.pid").read_text()), signal.SIGKILL)
                self.fail("supervisor left a live descendant after timeout")

    def test_metrics_fail_closed_for_missing_nan_and_wrong_identity(self):
        for mode in ("missing", "nan", "wrong_seed"):
            with self.subTest(mode=mode):
                c = self.coordinator()
                spec = self.spec(mode=mode, project_id=mode, seeds=(1,))
                c.create(spec, allow_execution=True)
                self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"invalid": 2})

    def test_source_mutation_invalidates_run(self):
        c, spec = self.start(mode="mutate", seeds=(1,))
        self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"invalid": 2})

    def test_changed_snapshot_and_changed_definition_are_rejected(self):
        c, spec = self.start()
        with self.assertRaisesRegex(ValueError, "frozen definition"):
            c.create(replace(spec, goal="Different experiment"), allow_execution=True)
        (self.source / "behavior.py").write_text("SCORE = 0.3\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            c.create(spec, allow_execution=True)

    def test_budget_reserves_active_run_capacity(self):
        c, spec = self.start(mode="wait", seeds=(1,), timeout=0.2, total=0.2, concurrent=2)
        initial = c.tick(spec.project_id)
        self.assertEqual(initial["counts"], {"starting": 1, "queued": 1})
        status = c.wait(spec.project_id, timeout=10, interval=0.03)
        self.assertEqual(status["status"], "budget_exhausted")
        self.assertEqual(status["counts"], {"timed_out": 1, "queued": 1})

    def test_stale_claim_cannot_be_executed_twice(self):
        c, spec = self.start(seeds=(1,))
        job = c.store.jobs(spec.project_id)[0]
        other = self.coordinator()
        self.assertTrue(c.store.claim(job["id"], 1, 10))
        self.assertFalse(other.store.claim(job["id"], 1, 10))
        self.assertFalse(other.store.claim(c.store.jobs(spec.project_id)[1]["id"], 1, 10))
        self.assertEqual(c.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"succeeded": 2})
        with self.assertRaisesRegex(ValueError, "immutable"):
            c.store.transition(job["id"], "failed", {"status": "failed"})




    def test_cancellation_survives_transport_failure_and_controller_restart(self):
        c, spec = self.start(seeds=(1,), concurrent=1)
        row = c.store.jobs(spec.project_id)[0]
        c.store.claim(row["id"], 1, 30)
        with patch.object(c._runner(spec.project_id), "cancel", side_effect=ConnectionError("offline")):
            c.cancel(spec.project_id)
        c.close()
        self.coordinators.remove(c)
        resumed = self.coordinator()
        self.assertEqual(resumed.wait(spec.project_id, timeout=10, interval=0.03)["counts"], {"cancelled": 2})
        self.assertFalse((resumed._directory(row["definition"]) / "work" / "executions.txt").exists())


    def test_lost_supervisor_is_never_automatically_retried(self):
        c, spec = self.start(seeds=(1,), concurrent=1)
        row = c.store.jobs(spec.project_id)[0]
        c.store.claim(row["id"], 1, 30)
        directory = c._directory(row["definition"])
        c._prepare_job(row["definition"], directory)
        (directory / "started.json").write_text("{}")
        c.tick(spec.project_id)
        result = c.store.job(row["id"])
        self.assertEqual(result["status"], "lost")
        self.assertEqual(result["result"]["elapsed_seconds"], spec.budget.run_timeout_seconds)
        self.assertFalse((directory / "work" / "executions.txt").exists())
        self.assertEqual(c.status(spec.project_id)["status"], "needs_attention")
        self.assertEqual(c.store.jobs(spec.project_id)[1]["status"], "queued")

    def test_completion_published_during_lock_probe_is_not_lost(self):
        directory = self.root / "race"
        directory.mkdir()
        (directory / "started.json").write_text("{}")
        result = {"status": "succeeded", "exit_code": 0}
        def publish(*_):
            supervisor.atomic_json(directory / "status.json", result)
        with patch("tracepilot.experiments.supervisor.fcntl.flock", side_effect=publish):
            self.assertEqual(supervisor.inspect_job(directory), result)

    def test_cli_executes_real_fixture_and_writes_report(self):
        spec = self.spec(seeds=(1,), project_id="cli")
        specification = self.root / "spec.json"
        specification.write_text(json.dumps(spec.to_dict()))
        command = [sys.executable, "-B", "-m", "tracepilot.cli", "research", "run", str(specification),
                   "--execute", "--wait", "--output-root", str(self.root / "out")]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["counts"], {"succeeded": 2})
        self.assertTrue(Path(payload["reports"]["json"]).exists())


if __name__ == "__main__":
    unittest.main()
