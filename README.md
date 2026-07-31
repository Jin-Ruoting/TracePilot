[English](README.md) | [简体中文](docs/README.zh-CN.md)

# TracePilot

TracePilot is a local-first, evidence-driven single-agent system for technical research and decision making. It combines web or frozen sources with local project context, produces a cited action plan, and executes file changes or commands only after explicit human approval.

The default scenarios are dependency upgrades, API migrations, architecture choices, and failure diagnosis. BioML is retained as an optional task pack and compatibility example, not as the core product domain.

> **Current status:** this repository is a local implementation, not a hosted production service. The default path is deterministic and offline so tests and demos do not require credentials. Live LLM and FastAPI support are optional extras.

TracePilot was independently designed and implemented.

## Who It Is For

TracePilot is designed for developers, maintainers, and technical leads working on repository-aware tasks such as dependency upgrades, API migrations, architecture decisions, and failure diagnosis.

These tasks often require more than a single answer: inspect the current code, retrieve versioned documentation, reconcile evidence, plan dependent steps, pause for approval, apply a change, and verify the result. TracePilot uses an agent for that open-ended coordination while keeping permissions, approval checks, and execution rules deterministic. Simple fact lookups should still use search or a smaller RAG workflow.

## What It Demonstrates

- **Complete RAG:** source ingestion, structured chunking, content/version metadata, FTS5 plus embedding retrieval, fusion, reranking, citation binding, faithfulness checks, and prompt-injection checks.
- **Explicit agent architecture:** LLM, planning, memory, tools, and execution are separate modules with typed contracts.
- **DAG planning:** independent read-only research steps can run concurrently; evidence gaps and tool failures can trigger bounded local replanning.
- **Controlled execution:** tools declare `read`, `internal_write`, or `mutation` effects. Mutations require a frozen argument hash and explicit approval.
- **Maintained memory:** working state, immutable run history, and user-promoted decision memory are stored in SQLite WAL and isolated by workspace.
- **Recoverability and inspection:** checkpoints, resumable tasks, trace events, citations, approval records, and execution results remain inspectable.

## Architecture

```text
TaskRequest
  -> source and workspace snapshot
  -> hybrid RAG retrieval
  -> single-agent DAG planning
  -> parallel read-only tools
  -> evidence coverage check and bounded replanning
  -> ActionPlan
  -> policy gate and human approval
  -> approved mutation execution (workspace cwd for commands)
  -> verification
  -> cited report and memory update
```

| Module | Responsibility |
| --- | --- |
| **LLM Gateway** | Structured generation, model and prompt versioning, retries, token/cost metadata, and a deterministic test implementation |
| **Planning Engine** | Dependency-aware `TaskPlan` DAGs, bounded planning rounds, read-only parallelism, and local replanning |
| **Memory Store** | SQLite-backed working memory, immutable run memory, and explicitly promoted decision memory |
| **Tool Registry** | Typed schemas, effect classes, timeout, idempotency, permissions, and state dependencies |
| **Execution Runtime** | Automatic reads/internal writes, approval-gated mutations, workspace path checks, and result verification |

TracePilot deliberately uses one agent. Parallel DAG steps provide concurrency without duplicating context across role-playing agents. Multi-agent orchestration is deferred until an evaluation shows a repeatable quality gain that exceeds its added latency, cost, recovery, and testing complexity.

Detailed architecture notes are currently available in Chinese at [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), covering component boundaries, lifecycle, RAG, memory, and the approval protocol.

## Quick Start

TracePilot requires Python 3.11 or newer.

```bash
python -m pip install -e .
tracepilot ingest examples/technical_decisions/architecture_choice \
  --frozen-sources benchmarks/general_sources.jsonl
tracepilot run "Choose a persistence layer for this local task service" \
  --workspace examples/technical_decisions/architecture_choice \
  --frozen-sources benchmarks/general_sources.jsonl
```

Run the deterministic validation suite and generic benchmark:

```bash
make check
make benchmark
```

Start the local Workbench with the optional API dependencies:

```bash
python -m pip install -e '.[api]'
tracepilot serve --host 127.0.0.1 --port 8787
```

Then open `http://127.0.0.1:8787`. The Workbench exposes the task list, DAG timeline, evidence and citations, memory records, approval review, patch or command previews, execution logs, and final report. Task updates are streamed with SSE.

The default implementation is deterministic and offline. To use the optional OpenAI gateway:

```bash
python -m pip install -e '.[llm]'
```

## CLI

```text
tracepilot ingest
tracepilot run
tracepilot resume
tracepilot approve
tracepilot memory list
tracepilot memory promote
tracepilot memory supersede
tracepilot memory delete
tracepilot serve
```

The command groups follow the task lifecycle:

| Command | Purpose |
| --- | --- |
| `tracepilot ingest` | Parse and index a workspace, URL, or frozen source corpus |
| `tracepilot run` | Create and run a technical research or decision task |
| `tracepilot resume` | Resume a task from durable task and checkpoint state |
| `tracepilot approve` | Approve, modify, or reject a pending mutation |
| `tracepilot memory list` | Inspect memory records for a workspace |
| `tracepilot memory promote` | Save a completed decision as reusable memory |
| `tracepilot memory supersede` | Replace an older decision without erasing its history |
| `tracepilot memory delete` | Delete a selected memory record |
| `tracepilot serve` | Start the FastAPI and SSE Workbench |

Stateful commands use the same database and workspace identity so another
process can find the correct task:

```bash
tracepilot resume TASK_ID \
  --workspace-id WORKSPACE_ID --db .tracepilot/tracepilot.db
tracepilot approve TASK_ID APPROVAL_ID --decision approved \
  --workspace-id WORKSPACE_ID --db .tracepilot/tracepilot.db
tracepilot memory list \
  --workspace-id WORKSPACE_ID --db .tracepilot/tracepilot.db
tracepilot memory supersede MEMORY_ID --content "updated decision" \
  --from-task COMPLETED_TASK_ID \
  --workspace-id WORKSPACE_ID --db .tracepilot/tracepilot.db
```

Superseding a stale memory requires a completed current task so the replacement is bound to fresh source and workspace fingerprints.

## Retrieval And Evidence

The knowledge index is separate from agent memory:

```text
web / URL / local workspace
  -> parse and structured chunk
  -> content hash, source version, and provenance
  -> SQLite FTS5 and deterministic embeddings
  -> query decomposition
  -> BM25 and vector recall
  -> deduplication and rank fusion
  -> structured reranking
  -> context assembly
  -> claim-to-chunk citation binding
  -> coverage, faithfulness, and poisoning checks
```

Every supported factual claim in a final report must point to an `EvidenceChunk`. When evidence is missing or untrusted, the agent marks the uncertainty or abstains instead of presenting an unsupported conclusion.

The offline faithfulness gate is a conservative lexical heuristic with clause-level coverage plus number, negation, and poisoning checks. It is not presented as a general semantic-entailment model.

## Memory Policy

TracePilot does not implement an unrestricted personal memory:

- **Working memory** checkpoints successful task steps automatically.
- **Run memory** appends the plan, tool results, trace, and failure information when a run finishes.
- **Decision memory** is written only after a user explicitly promotes a result.
- A changed workspace commit, file hash, or source version marks related memory as stale.
- New conclusions use `supersedes_id`; history is retained instead of silently overwritten.
- Recall is workspace-scoped by default, and records can be inspected, disabled, superseded, or deleted.
- Recalled decisions are planning hints that trigger current-source revalidation; they are never accepted as evidence or cited directly.

## Tools And Approval

Initial read-only tools cover web search, URL fetches, workspace indexing, index search, file reads, code search, and Git inspection. `propose_patch` writes only an internal artifact. Applying a patch, running an arbitrary command, changing user files, or writing to an external service is a `mutation`.

Read operations and internal trace/index/checkpoint writes can run automatically. A mutation cannot run until the user approves the exact frozen arguments. A changed argument hash invalidates the approval, and mutations are not retried automatically.

The built-in `run_command` uses an exact argument vector and a workspace working directory; it is not an operating-system sandbox. Use an external container or sandbox profile when untrusted commands require process-level isolation.

Thread timeouts are used only for read/internal tools. Mutation handlers are awaited to avoid falsely reporting a timeout while side effects continue; a custom mutation tool must provide its own cancellable timeout. The built-in command tool terminates its process group on timeout.

Every structured patch entry must include the current file's `expected_sha256`; use the SHA-256 of empty bytes when creating a new file. TracePilot checks it again before writing so a stale approval cannot silently overwrite intervening edits.

## Benchmarks

The generic benchmark corpus covers four computer-system tasks:

- dependency upgrade
- API migration
- architecture choice
- test or runtime failure diagnosis

The fixtures use frozen source snapshots and small local workspaces so regression tests do not depend on live search. Product names and case contents in the generic corpus are synthetic. The evaluation surface includes retrieval and citation checks, stale-source handling, prompt injection, DAG validity, recovery, approval enforcement, idempotency, task isolation, and restart behavior. These results demonstrate regression coverage, not superiority over third-party agents or production performance.

## Compatibility

`tracepilot` is the canonical package and CLI. The legacy `bioml_agent` package, `bioml-agent` command, v1 BioML checkpoint mapper, and fixtures remain available during the compatibility period:

```bash
bioml-agent "Should we try graph transfer?"
PYTHONPATH=src python -m bioml_agent.cli "Should we try graph transfer?"
```

BioML-specific types and examples live under `tracepilot.taskpacks.bioml` and `examples/bioml`. They demonstrate how a domain task pack can reuse the generic runtime without entering the default workflow or primary benchmark.

Generated reports, traces, databases, caches, and workbench exports are written locally and intentionally excluded from version control. The quality-gate workflow publishes its outputs as CI artifacts.

## Repository Layout

```text
.
├── src/tracepilot/              # Generic agent, RAG, memory, tools, runtime, API
├── src/bioml_agent/             # Compatibility package and legacy commands
├── examples/technical_decisions/
├── examples/bioml/              # Optional task-pack examples
├── benchmarks/                  # Frozen generic and legacy fixtures
├── docs/
│   ├── ARCHITECTURE.md
│   └── README.zh-CN.md
└── tests/
```

The Chinese version is available at [`docs/README.zh-CN.md`](docs/README.zh-CN.md).
