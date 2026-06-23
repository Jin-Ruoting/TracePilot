# TracePilot Service API Readiness
Passed: True
Runnable now: False
Endpoints: 7
Tool endpoints: 5

## Optional Dependencies
- `fastapi`: missing
  - Purpose: Build the ASGI app and expose path operations.
  - Install: `python -m pip install -e '.[api]'`
- `uvicorn`: missing
  - Purpose: Run the local ASGI development server.
  - Install: `python -m pip install -e '.[api]'`

## Endpoint Plan

### GET /health
Return service liveness and route count.
- source: `service_runtime.create_fastapi_app`
- request required: (none)
- response schema: `object`
- session state dependencies: (none)
- approval policy: no approval required; liveness read is read-only

### GET /service-interface
Return the contract-derived MCP-style descriptors and HTTP route specs.
- source: `service_runtime.create_fastapi_app`
- request required: (none)
- response schema: `ServiceInterfaceReport`
- session state dependencies: (none)
- approval policy: no approval required; interface read is read-only

### POST /tools/search_literature
Invoke `search_literature` through the session-backed service runtime.
- source: `service_runtime.ServiceRuntime.invoke`
- request required: query, top_k
- response schema: `ServiceInvocationResult`
- session state dependencies: (none)
- approval policy: no approval required; retrieval is read-only

### POST /tools/read_experiment_context
Invoke `read_experiment_context` through the session-backed service runtime.
- source: `service_runtime.ServiceRuntime.invoke`
- request required: project_name
- response schema: `ServiceInvocationResult`
- session state dependencies: (none)
- approval policy: no approval required; context read is read-only

### POST /tools/score_transfer_fit
Invoke `score_transfer_fit` through the session-backed service runtime.
- source: `service_runtime.ServiceRuntime.invoke`
- request required: (none)
- response schema: `ServiceInvocationResult`
- session state dependencies: candidates, context
- approval policy: no approval required; scoring is an internal read-only transformation

### POST /tools/draft_experiment_plan
Invoke `draft_experiment_plan` through the session-backed service runtime.
- source: `service_runtime.ServiceRuntime.invoke`
- request required: (none)
- response schema: `ServiceInvocationResult`
- session state dependencies: evidence, context
- approval policy: proposal only; execution still requires review checkpoint

### POST /tools/create_review_checkpoint
Invoke `create_review_checkpoint` through the session-backed service runtime.
- source: `service_runtime.ServiceRuntime.invoke`
- request required: (none)
- response schema: `ServiceInvocationResult`
- session state dependencies: plan, evidence, context
- approval policy: promotion must produce requires_approval before any experiment execution

## Launch
Install the optional API extras first:

```bash
python -m pip install -e '.[api]'
```

Then run:

```bash
PYTHONPATH=src python -m bioml_agent.service_api serve --host 127.0.0.1 --port 8787
```

FastAPI will expose `/openapi.json` and `/docs` when the optional dependencies are installed.