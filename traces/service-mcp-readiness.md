# TracePilot MCP Readiness
Passed: True
Runnable now: False
Server: TracePilot MCP
Tool bindings: 5
Resources: 2
Prompts: 1
Transports: 2

## Optional Dependencies
- `mcp`: missing
  - Purpose: Build and run the optional FastMCP server.
  - Install: `python -m pip install -e '.[mcp]'`

## Tool Bindings

### search_literature
Search the configured literature provider for source candidates.
- input required: query, top_k
- output: `list[PaperCandidate]`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=True, readOnlyHint=True
- session state dependencies: (none)
- approval policy: no approval required; retrieval is read-only
- runtime path: `ServiceRuntime.invoke('POST', '/tools/search_literature', ...)`

### read_experiment_context
Load trusted baseline, split, metric, and gate constraints for a project.
- input required: project_name
- output: `ExperimentContext`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- session state dependencies: (none)
- approval policy: no approval required; context read is read-only
- runtime path: `ServiceRuntime.invoke('POST', '/tools/read_experiment_context', ...)`

### score_transfer_fit
Score already retrieved candidates against the loaded experiment context.
- input required: (none)
- output: `list[EvidenceItem]`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- session state dependencies: candidates, context
- approval policy: no approval required; scoring is an internal read-only transformation
- runtime path: `ServiceRuntime.invoke('POST', '/tools/score_transfer_fit', ...)`

### draft_experiment_plan
Draft an experiment matrix, promotion gate, risks, recommendation, and cited claims.
- input required: (none)
- output: `ExperimentPlan`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- session state dependencies: evidence, context
- approval policy: proposal only; execution still requires review checkpoint
- runtime path: `ServiceRuntime.invoke('POST', '/tools/draft_experiment_plan', ...)`

### create_review_checkpoint
Create the human approval checkpoint for the drafted experiment plan.
- input required: (none)
- output: `ReviewCheckpoint`
- annotations: destructiveHint=False, idempotentHint=True, openWorldHint=False, readOnlyHint=True
- session state dependencies: plan, evidence, context
- approval policy: promotion must produce requires_approval before any experiment execution
- runtime path: `ServiceRuntime.invoke('POST', '/tools/create_review_checkpoint', ...)`

## Resources

### bioml://service/interface
Contract-derived MCP descriptors and HTTP route specs.
- name: Service Interface
- source: `service_interface.service_interface_to_dict`
- read only: True

### bioml://service/runtime-demo
Deterministic trace showing the session-backed route workflow.
- name: Service Runtime Demo
- source: `service_runtime.run_service_runtime_demo`
- read only: True

## Prompts

### bioml_experiment_review
Review a TracePilot planning decision for evidence, gates, source refs, and approval requirements.
- arguments: question, focus
- source: `service_mcp.create_fastmcp_server`

## Transports

### stdio
Local desktop/client integration with a process-launched MCP server.
```bash
PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport stdio
```

### streamable-http
Remote-capable MCP transport served at the SDK default streamable HTTP path.
```bash
PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport streamable-http
```