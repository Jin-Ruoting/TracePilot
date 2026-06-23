# TracePilot Service Runtime Demo
Passed: True
Routes: 5
Invocations: 5
Trace events: 5
Final state keys: candidates, context, evidence, plan, review, trace
Approval required: True

## What This Proves
The same contract-derived route specs can execute the full TracePilot planning workflow through a session-backed service boundary.
State-dependent tools receive candidates, context, evidence, and plan from service session state instead of trusting the caller to fabricate internal objects.

## Invocation Trace

### 1. PASS POST /tools/search_literature
- tool: `search_literature`
- status code: `200`
- request keys: query, top_k
- state before: (empty)
- state after: candidates, trace
- trace events added: 1
- approval required: False
- output type: `list`

### 2. PASS POST /tools/read_experiment_context
- tool: `read_experiment_context`
- status code: `200`
- request keys: project_name
- state before: candidates, trace
- state after: candidates, context, trace
- trace events added: 1
- approval required: False
- output type: `ExperimentContext`

### 3. PASS POST /tools/score_transfer_fit
- tool: `score_transfer_fit`
- status code: `200`
- request keys: (none)
- state before: candidates, context, trace
- state after: candidates, context, evidence, trace
- trace events added: 1
- approval required: False
- output type: `list`

### 4. PASS POST /tools/draft_experiment_plan
- tool: `draft_experiment_plan`
- status code: `200`
- request keys: (none)
- state before: candidates, context, evidence, trace
- state after: candidates, context, evidence, plan, trace
- trace events added: 1
- approval required: False
- output type: `ExperimentPlan`

### 5. PASS POST /tools/create_review_checkpoint
- tool: `create_review_checkpoint`
- status code: `200`
- request keys: (none)
- state before: candidates, context, evidence, plan, trace
- state after: candidates, context, evidence, plan, review, trace
- trace events added: 1
- approval required: True
- output type: `ReviewCheckpoint`