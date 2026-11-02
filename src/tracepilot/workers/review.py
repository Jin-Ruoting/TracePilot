from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

from ..research.models import ResearchSpec, digest


def validate_metrics(job: dict[str, Any], result: dict[str, Any], directory: Path) -> dict[str, Any]:
    root = Path(job["artifact_directory"]) if job.get("artifact_directory") else directory / "work"
    path = root / "metrics.json"
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("successful command did not produce a contained metrics.json")
    if path.stat().st_size > 1_000_000:
        raise ValueError("metrics artifact exceeds size limit")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != result.get("metrics_sha256"):
        raise ValueError("metrics artifact is absent from or differs from the supervisor record")
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
        raise ValueError("metrics artifact must contain an object of metric values")
    for key in ("dataset", "seed", "split_seed", "partition"):
        if value.get(key) != job[key]:
            raise ValueError(f"metrics {key} differs from the frozen experiment")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported metric artifact version")
    score = value.get("metrics", {}).get(job["metric"])
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("primary metric must be a finite number")
    if not isinstance(value.get("protocol"), dict) or not value["protocol"]:
        raise ValueError("metrics must record the training and evaluation protocol")
    split_hash = value.get("split_hash", "")
    if not isinstance(split_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", split_hash):
        raise ValueError("metrics must identify the data split")
    if value["protocol"].get("test_evaluated") is True:
        raise ValueError("selection experiments must not evaluate the reserved test set")
    value["protocol_hash"] = digest(value["protocol"])
    return value


def bootstrap_mean_interval(values: list[float], *, samples: int = 2000) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(20261102)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]


class ReviewWorker:
    """Review all declared seeds; incomplete or incomparable pairs cannot support a conclusion."""

    def review(self, spec: ResearchSpec, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        direction = 1 if spec.direction == "maximize" else -1
        for dataset in spec.datasets:
            for variant in spec.variants:
                if variant.variant_id == spec.baseline:
                    continue
                baseline_scores: list[float] = []
                candidate_scores: list[float] = []
                problems: list[str] = []
                run_ids = []
                for seed in spec.seeds:
                    pair = []
                    for name in (spec.baseline, variant.variant_id):
                        found = [j for j in jobs if j["definition"]["dataset"] == dataset
                                 and j["definition"]["seed"] == seed and j["definition"]["variant"] == name]
                        if len(found) != 1 or found[0]["status"] != "succeeded":
                            problems.append(f"{name}/seed={seed}: missing or unsuccessful")
                            break
                        pair.append(found[0])
                    if len(pair) != 2:
                        continue
                    left, right = [j["result"]["metrics"] for j in pair]
                    if any(left[key] != right[key] for key in ("split_hash", "protocol_hash", "partition")):
                        problems.append(f"seed={seed}: split or training protocol mismatch")
                        continue
                    baseline_scores.append(float(left["metrics"][spec.metric]))
                    candidate_scores.append(float(right["metrics"][spec.metric]))
                    run_ids.extend(j["id"] for j in pair)
                deltas = [direction * (b - a) for a, b in zip(baseline_scores, candidate_scores)]
                interval = bootstrap_mean_interval(deltas)
                complete = len(deltas) == len(spec.seeds) and not problems
                verdict = "inconclusive"
                if complete and interval:
                    if interval[0] > spec.minimum_effect:
                        verdict = "promising"
                    elif interval[1] < -spec.minimum_effect:
                        verdict = "deprioritize"
                rows.append({
                    "dataset": dataset, "candidate": variant.variant_id,
                    "paired_runs": len(deltas), "expected_pairs": len(spec.seeds),
                    "baseline_mean": statistics.mean(baseline_scores) if deltas else None,
                    "baseline_sd": statistics.stdev(baseline_scores) if len(deltas) > 1 else None,
                    "candidate_mean": statistics.mean(candidate_scores) if deltas else None,
                    "candidate_sd": statistics.stdev(candidate_scores) if len(deltas) > 1 else None,
                    "oriented_delta_mean": statistics.mean(deltas) if deltas else None,
                    "paired_delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
                    "bootstrap_95_interval": interval, "verdict": verdict,
                    "problems": problems, "evidence_run_ids": run_ids,
                })
        successful = [j for j in jobs if j["status"] == "succeeded"]
        return {
            "project_id": spec.project_id, "hypothesis": spec.hypothesis.statement,
            "hypothesis_origin": spec.hypothesis.origin,
            "metric": spec.metric, "partition": spec.partition, "direction": spec.direction,
            "minimum_effect": spec.minimum_effect,
            "planned_runs": spec.run_count, "recorded_runs": len(jobs),
            "successful_runs": len(successful),
            "completion_rate": len(successful) / spec.run_count,
            "run_seconds": sum((j["result"] or {}).get("elapsed_seconds", 0) for j in jobs),
            "artifact_coverage": sum(bool((j["result"] or {}).get("metrics_sha256")) for j in successful)
                                 / len(successful) if successful else 0,
            "comparisons": rows,
            "interpretation": "Pilot validation results; bootstrap intervals describe seed variation on a fixed split. "
                              "They are not a novelty claim, independent-dataset confidence interval, or final test result.",
        }
