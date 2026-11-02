"""Run the real ML pilot and a serial replay; retain timings and prediction agreement."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

from tracepilot.coordinator import ResearchCoordinator
from tracepilot.research.models import ResearchSpec
from tracepilot.research.reporting import write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Interpreter with the pinned ML dependencies")
    parser.add_argument("--output-root", default=os.environ.get("TRACEPILOT_STATE_ROOT", "outputs/research"))
    parser.add_argument("--artifacts-root", default=os.environ.get("TRACEPILOT_ARTIFACTS_ROOT"))
    parser.add_argument("--project-id")
    args = parser.parse_args()
    spec = ResearchSpec.load(Path(__file__).with_name("project.json"))
    if args.project_id:
        spec = replace(spec, project_id=args.project_id)
    coordinator = ResearchCoordinator(args.output_root, artifacts_root=args.artifacts_root)
    runs = []
    try:
        for current in (spec, replace(spec, project_id=spec.project_id + "-serial-replay",
                                    budget=replace(spec.budget, max_concurrent=1))):
            started = time.perf_counter()
            coordinator.create(current, backend={"kind": "local", "python": args.python}, allow_execution=True)
            timing_path = coordinator.report_directory(current.project_id) / "timing.json"
            initial = coordinator.status(current.project_id)
            status = coordinator.wait(current.project_id, timeout=600)
            elapsed = time.perf_counter() - started
            reports = write_report(coordinator, current.project_id)
            timing = {
                "wall_seconds": elapsed, "max_concurrent": current.budget.max_concurrent,
                "measurement": "create, prepare, dispatch and wait; final report rendering excluded",
                "new_project_execution": initial["counts"] == {"queued": current.run_count},
            }
            if timing_path.exists():
                timing = json.loads(timing_path.read_text())
            else:
                timing_path.write_text(json.dumps(timing, indent=2) + "\n")
            runs.append(dict(project_id=current.project_id, status=status["status"], timing=timing, reports=reports))
            print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
            if status["status"] != "completed":
                return 2
        pairs = {}
        for row in coordinator.store.jobs(spec.project_id):
            job, result = row["definition"], row["result"]
            pairs[(job["dataset"], job["variant"], job["seed"])] = result["metrics"]
        agreement = []
        for row in coordinator.store.jobs(spec.project_id + "-serial-replay"):
            job, right = row["definition"], row["result"]["metrics"]
            left = pairs[(job["dataset"], job["variant"], job["seed"])]
            agreement.append({
                "dataset": job["dataset"], "variant": job["variant"], "seed": job["seed"],
                "metrics_equal": left["metrics"] == right["metrics"],
                "predictions_equal": left["validation_predictions"] == right["validation_predictions"],
                "protocol_equal": left["protocol_hash"] == right["protocol_hash"],
                "split_equal": left["split_hash"] == right["split_hash"],
            })
        summary = {
            "schema_version": 1, "runs": runs, "host_platform": platform.platform(),
            "controller_python": platform.python_version(), "experiment_python": args.python,
            "comparison_count": len(agreement),
            "identical_prediction_pairs": sum(row["predictions_equal"] for row in agreement),
            "identical_metric_pairs": sum(row["metrics_equal"] for row in agreement),
            "serial_over_parallel_wall_ratio": runs[1]["timing"]["wall_seconds"] / runs[0]["timing"]["wall_seconds"],
            "timing_limit": "One parallel execution followed by one serial replay on this host; "
                            "order/cache effects are not controlled. Not an agent-quality or robust speedup benchmark.",
            "model_calls": 0, "gpu_used": False, "agreement": agreement,
        }
        destination = coordinator.report_directory(spec.project_id) / "reproducibility.json"
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        print(str(destination), flush=True)
        return 0 if all(all(row[k] for k in ("metrics_equal", "predictions_equal", "protocol_equal", "split_equal"))
                        for row in agreement) else 3
    finally:
        coordinator.close()


if __name__ == "__main__":
    raise SystemExit(main())
