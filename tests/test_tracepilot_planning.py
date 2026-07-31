from __future__ import annotations

import tempfile
import unittest

from tracepilot.models import PlanStep, TaskContext, TaskPlan, TaskRequest
from tracepilot.planning import (
    DeterministicPlanner,
    PlanValidationError,
    PlanningConfig,
    PlanningLimitError,
    topological_batches,
    validate_plan,
)


TOOLS = {
    "index_workspace",
    "search_index",
    "search_code",
    "inspect_git",
    "web_search",
    "fetch_url",
    "propose_patch",
    "apply_patch",
    "run_command",
}


class TracePilotPlanningTest(unittest.TestCase):
    def test_planner_builds_bounded_parallel_dag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            request = TaskRequest(
                goal="Upgrade a dependency",
                workspace_path=tmp_dir,
                source_urls=("https://example.test/migration",),
            )
            plan = DeterministicPlanner(available_tools=TOOLS).plan(
                request,
                TaskContext(task_id=request.task_id, workspace_path=tmp_dir),
            )

        validate_plan(plan)
        self.assertLessEqual(len(plan.steps), 20)
        self.assertEqual(plan.max_concurrency, 4)
        batches = topological_batches(plan)
        self.assertTrue(all(len(batch) <= 4 for batch in batches))
        self.assertTrue(
            all(
                len(batch) == 1
                or all(step.effect_class == "read" for step in batch)
                for batch in batches
            )
        )
        self.assertIn("index_workspace", {step.step_id for step in plan.steps})

    def test_replanning_is_limited_to_three_total_rounds(self) -> None:
        request = TaskRequest(goal="Diagnose a failing test")
        context = TaskContext(metadata={"evidence_gaps": ["official changelog"]})
        planner = DeterministicPlanner(available_tools=TOOLS)
        first = planner.plan(request, context)
        second = planner.replan(
            request,
            context,
            first,
            failed_steps=("web_search",),
        )
        third = planner.replan(request, context, second)

        self.assertEqual((first.revision, second.revision, third.revision), (1, 2, 3))
        self.assertTrue(
            any(step.step_id == "recovery_search_2" for step in second.steps)
        )
        with self.assertRaises(PlanningLimitError):
            planner.replan(request, context, third)

    def test_requested_mutation_follows_action_plan(self) -> None:
        request = TaskRequest(
            goal="Apply the reviewed compatibility patch",
            metadata={
                "requested_mutation": {
                    "tool_name": "apply_patch",
                    "arguments": {"path": "src/example.py", "patch": "replacement"},
                }
            },
        )

        plan = DeterministicPlanner(available_tools=TOOLS).plan(
            request,
            TaskContext(),
        )
        mutation = plan.step("requested_mutation")
        self.assertEqual(mutation.effect_class, "mutation")
        self.assertEqual(mutation.depends_on, ("propose_patch",))

    def test_validation_rejects_cycles(self) -> None:
        plan = TaskPlan(
            task_id="task",
            steps=(
                PlanStep(step_id="a", tool_name="read", depends_on=("b",)),
                PlanStep(step_id="b", tool_name="read", depends_on=("a",)),
            ),
        )

        with self.assertRaisesRegex(PlanValidationError, "cycle"):
            validate_plan(plan)

    def test_hard_configuration_limits_cannot_be_raised(self) -> None:
        with self.assertRaises(ValueError):
            PlanningConfig(max_steps=21)
        with self.assertRaises(ValueError):
            PlanningConfig(max_rounds=4)
        with self.assertRaises(ValueError):
            PlanningConfig(max_parallel_reads=5)


if __name__ == "__main__":
    unittest.main()
