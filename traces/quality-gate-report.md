# TracePilot Quality Gate
Passed: True
Checks: 20/20

## PASS compile
Command: `/opt/anaconda3/bin/python -m compileall -q src`
Return code: `0`

## PASS unit_tests
Command: `/opt/anaconda3/bin/python -m unittest discover -s tests`
Return code: `0`

Stderr tail:
```text
......................................................................................................................................
----------------------------------------------------------------------
Ran 134 tests in 0.831s

OK
```

## PASS tool_contracts
Command: `/opt/anaconda3/bin/python -m bioml_agent.tool_contracts --validate`
Return code: `0`

Stdout tail:
```text
# Tool Contract Validation
Passed: True
Registered tools: create_review_checkpoint, draft_experiment_plan, read_experiment_context, score_transfer_fit, search_literature
Manifest tools: search_literature, read_experiment_context, score_transfer_fit, draft_experiment_plan, create_review_checkpoint
```

## PASS eval_coverage_markdown
Command: `/opt/anaconda3/bin/python -m bioml_agent.eval_report --out traces/eval-coverage-report.md`
Return code: `0`
Artifacts: `traces/eval-coverage-report.md`

## PASS eval_coverage_json
Command: `/opt/anaconda3/bin/python -m bioml_agent.eval_report --json --out traces/eval-coverage-report.json`
Return code: `0`
Artifacts: `traces/eval-coverage-report.json`

## PASS promotion_artifacts
Command: `/opt/anaconda3/bin/python -m bioml_agent.cli Should we try retrieval-aware conditioning for promoter expression prediction? --experiment-log examples/epicetra_experiment_log.json --eval --audit --trace-out traces/promotion-run.json --report-out traces/promotion-decision-packet.md --report-html-out traces/promotion-decision-review.html`
Return code: `0`
Artifacts: `traces/promotion-run.json`, `traces/promotion-decision-packet.md`, `traces/promotion-decision-review.html`

Stdout tail:
```text
- PASS source_refs_valid: source URLs and DOI refs are parseable
- PASS review_checkpoint_valid: review checkpoint matches recommendation risk
- PASS split_gate_present: plan gates main and diagnostic split claims
- PASS promotion_supported: promotion backed by high-fit evidence
- PASS source_poisoning_guarded: no instruction-like source text detected

## Decision Audit
Passed: True
Decision: ready_for_human_review
Score: 1.000
Summary: Promotion is evidence-grounded but still requires human approval before experiment execution.
- PASS promotion_evidence [critical]: promotion is grounded in cited high-fit top evidence
- PASS abstention_quality [info]: run does not abstain
- PASS split_controls [critical]: promotion gate separates main split, diagnostic split, and leakage boundary
- PASS experiment_matrix [major]: matrix has baseline/control, variants, and reporting stratification
- PASS review_checkpoint [critical]: promotion is gated behind specific human approval actions
- PASS source_grounding [major]: plan claims are tied to evidence ids and source refs
- PASS source_poisoning [info]: no instruction-like source text detected
```

## PASS review_workbench_html
Command: `/opt/anaconda3/bin/python -m bioml_agent.review_workbench traces/promotion-run.json --out traces/review-workbench.html`
Return code: `0`
Artifacts: `traces/review-workbench.html`

## PASS review_workbench_json
Command: `/opt/anaconda3/bin/python -m bioml_agent.review_workbench traces/promotion-run.json --json --out traces/review-workbench.json`
Return code: `0`
Artifacts: `traces/review-workbench.json`

## PASS portfolio_demo
Command: `/opt/anaconda3/bin/python -m bioml_agent.demo --json --out traces/portfolio-demo.json`
Return code: `0`
Artifacts: `traces/portfolio-demo.json`

Stdout tail:
```text
    "baseline_passed_count": 0,
    "improved_count": 4,
    "total": 4
  },
  "checkpoint": {
    "failure_node": "read_experiment_context",
    "failure_type": "ValueError",
    "resumed_recommendation": "promote to small controlled screen",
    "resumed_review_status": "requires_approval",
    "trace_tools": [
      "search_literature",
      "read_experiment_context",
      "score_transfer_fit",
      "draft_experiment_plan",
      "create_review_checkpoint"
    ]
  }
}
```

## PASS reference_catalog_markdown
Command: `/opt/anaconda3/bin/python -m bioml_agent.reference_catalog --out traces/reference-catalog-report.md`
Return code: `0`
Artifacts: `traces/reference-catalog-report.md`

## PASS reference_catalog_json
Command: `/opt/anaconda3/bin/python -m bioml_agent.reference_catalog --json --out traces/reference-catalog-report.json`
Return code: `0`
Artifacts: `traces/reference-catalog-report.json`

## PASS service_interface_markdown
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_interface --out traces/service-interface-report.md`
Return code: `0`
Artifacts: `traces/service-interface-report.md`

## PASS service_interface_json
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_interface --json --out traces/service-interface-report.json`
Return code: `0`
Artifacts: `traces/service-interface-report.json`

## PASS service_runtime_markdown
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_runtime --out traces/service-runtime-demo.md`
Return code: `0`
Artifacts: `traces/service-runtime-demo.md`

## PASS service_runtime_json
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_runtime --json --out traces/service-runtime-demo.json`
Return code: `0`
Artifacts: `traces/service-runtime-demo.json`

## PASS service_api_markdown
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_api --out traces/service-api-readiness.md`
Return code: `0`
Artifacts: `traces/service-api-readiness.md`

## PASS service_api_json
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_api --json --out traces/service-api-readiness.json`
Return code: `0`
Artifacts: `traces/service-api-readiness.json`

## PASS service_mcp_markdown
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_mcp --out traces/service-mcp-readiness.md`
Return code: `0`
Artifacts: `traces/service-mcp-readiness.md`

## PASS service_mcp_json
Command: `/opt/anaconda3/bin/python -m bioml_agent.service_mcp --json --out traces/service-mcp-readiness.json`
Return code: `0`
Artifacts: `traces/service-mcp-readiness.json`

## PASS portfolio_index
Command: `/opt/anaconda3/bin/python -m bioml_agent.portfolio_index build --out traces/index.html`
Return code: `0`
Artifacts: `traces/index.html`, `traces/reference-catalog-report.md`, `traces/review-workbench.html`, `traces/service-interface-report.md`, `traces/service-runtime-demo.md`, `traces/service-api-readiness.md`, `traces/service-mcp-readiness.md`