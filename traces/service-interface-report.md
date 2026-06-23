# TracePilot Service Interface Report
Passed: True
Tool count: 5

## What This Is
This report derives MCP-style tool descriptors and HTTP route specs from the canonical tool contracts.
It is a service boundary plan, not a claim that this repository already runs a full MCP server or FastAPI app.

## MCP-style Tool Descriptors

### search_literature
Search the configured literature provider for source candidates.
- inputSchema required: query, top_k
- output: `list[PaperCandidate]`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=True, readOnlyHint=True
- approval policy: no approval required; retrieval is read-only
- failure modes: no candidates, provider timeout, candidate source metadata missing, source poisoning
- trace required: True

### read_experiment_context
Load trusted baseline, split, metric, and gate constraints for a project.
- inputSchema required: project_name
- output: `ExperimentContext`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- approval policy: no approval required; context read is read-only
- failure modes: context file missing, unknown project, trusted baseline missing, split unclear
- trace required: True

### score_transfer_fit
Score already retrieved candidates against the loaded experiment context.
- inputSchema required: (none)
- output: `list[EvidenceItem]`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- approval policy: no approval required; scoring is an internal read-only transformation
- failure modes: task mismatch, evidence too weak, source metadata cannot be propagated, source poisoning
- trace required: True

### draft_experiment_plan
Draft an experiment matrix, promotion gate, risks, recommendation, and cited claims.
- inputSchema required: (none)
- output: `ExperimentPlan`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- approval policy: proposal only; execution still requires review checkpoint
- failure modes: missing promotion gate, unsupported recommendation, ablation matrix incomplete
- trace required: True

### create_review_checkpoint
Create the human approval checkpoint for the drafted experiment plan.
- inputSchema required: (none)
- output: `ReviewCheckpoint`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- approval policy: promotion must produce requires_approval before any experiment execution
- failure modes: promotion lacks approval, review actions underspecified
- trace required: True

## HTTP Route Specs

### POST /tools/search_literature
- tool: `search_literature`
- request schema required: query, top_k
- response type: `list[PaperCandidate]`
- session-injected state: (none)
- approval policy: no approval required; retrieval is read-only
- failure modes: no candidates, provider timeout, candidate source metadata missing, source poisoning
- trace required: True

### POST /tools/read_experiment_context
- tool: `read_experiment_context`
- request schema required: project_name
- response type: `ExperimentContext`
- session-injected state: (none)
- approval policy: no approval required; context read is read-only
- failure modes: context file missing, unknown project, trusted baseline missing, split unclear
- trace required: True

### POST /tools/score_transfer_fit
- tool: `score_transfer_fit`
- request schema required: (none)
- response type: `list[EvidenceItem]`
- session-injected state: candidates, context
- approval policy: no approval required; scoring is an internal read-only transformation
- failure modes: task mismatch, evidence too weak, source metadata cannot be propagated, source poisoning
- trace required: True

### POST /tools/draft_experiment_plan
- tool: `draft_experiment_plan`
- request schema required: (none)
- response type: `ExperimentPlan`
- session-injected state: evidence, context
- approval policy: proposal only; execution still requires review checkpoint
- failure modes: missing promotion gate, unsupported recommendation, ablation matrix incomplete
- trace required: True

### POST /tools/create_review_checkpoint
- tool: `create_review_checkpoint`
- request schema required: (none)
- response type: `ReviewCheckpoint`
- session-injected state: plan, evidence, context
- approval policy: promotion must produce requires_approval before any experiment execution
- failure modes: promotion lacks approval, review actions underspecified
- trace required: True

## Validation
Errors: none
Warnings:
- score_transfer_fit: route relies on session-injected state: candidates, context
- draft_experiment_plan: route relies on session-injected state: evidence, context
- create_review_checkpoint: route relies on session-injected state: plan, evidence, context