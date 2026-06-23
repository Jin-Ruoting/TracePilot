# TracePilot

I built TracePilot as a source-grounded planning agent for open-ended technical decisions. The project focuses on the engineering surfaces that are hard to show in a simple chatbot: typed tool contracts, evidence-backed planning, durable state, resumable traces, evaluation coverage, human approval gates, and service-boundary readiness.

The current demo uses a gene-expression experiment-planning scenario as a high-constraint case study. I use that domain to create realistic constraints, but the system itself is a general agent-engineering project: the core ideas are tool boundaries, state, provenance, evaluation, and controlled service interfaces.

## Why I Built It

Many agent examples stop at "LLM plus a few tools." I wanted TracePilot to make the engineering surface explicit:

- **Tool use is explicit**: I define tool schemas, deterministic fallbacks, structured outputs, and auditable traces.
- **Evidence is first-class**: I bind plans to source cards, experiment context, claims, citations, and provenance checks.
- **State is recoverable**: I can stop a run at an intermediate node, write a checkpoint, and resume from it.
- **Evaluation is not a single demo prompt**: I use fixed corpora, abstention cases, source-poisoning cases, baseline comparison, and a local quality gate.
- **Human review is built in**: I generate review packets that expose promotion gates, audit findings, and approval checkpoints.
- **The boundary is productizable**: I can render the same contracts as CLI workflows, static review pages, FastAPI-style endpoints, and MCP-style descriptors.

## Quick Start

The default local workflow only requires Python 3.11+.

```bash
python -m pip install -e .
make check
make serve-portfolio
```

If your machine maps `python3` to an older system Python, use a Python 3.11+ interpreter explicitly:

```bash
make PYTHON=python check
make PYTHON=/path/to/python3.12 check
```

Then open:

```text
http://127.0.0.1:8765/traces/index.html
```

I can also run the main planner directly:

```bash
tracepilot "Should we try retrieval-aware conditioning?"
```

The internal package name is still `bioml_agent` for compatibility:

```bash
PYTHONPATH=src python -m bioml_agent.cli "Should we try graph transfer?"
PYTHONPATH=src python -m unittest discover -s tests
```

## Project Dashboard

I keep the generated artifacts under `traces/` so the repository can show both source code and reproducible outputs.

| Artifact | Purpose |
| --- | --- |
| `traces/index.html` | Static dashboard that links the generated project artifacts |
| `traces/quality-gate-report.md` | Local quality-gate result covering tests, tool contracts, evaluation, service reports, and generated artifacts |
| `traces/promotion-decision-review.html` | Static decision review with evidence, citations, audit results, approval gates, and trace |
| `traces/review-workbench.html` | Product-style review workbench for evidence coverage, claim grounding, audit, evals, and trace inspection |
| `traces/eval-coverage-report.md` | Coverage summary across benchmark corpora and guardrails |
| `traces/reference-catalog-report.md` | Generated summary of the project references and borrowed design patterns |
| `traces/service-interface-report.md` | Tool-contract mapping to service-style interfaces |
| `traces/service-runtime-demo.md` | Session-backed runtime route demo |
| `traces/service-api-readiness.md` | FastAPI adapter plan and OpenAPI preview |
| `traces/service-mcp-readiness.md` | MCP tool/resource/prompt/transport readiness report |

## Architecture

TracePilot is organized around a deterministic planning controller with optional LLM and service adapters.

```text
Question
  -> context provider
  -> source retrieval / frozen literature provider
  -> transfer-fit scoring
  -> plan synthesis
  -> evaluation and audit
  -> review packet / HTML workbench / service reports
```

Key modules:

| Area | Files |
| --- | --- |
| Controller and state | `src/bioml_agent/agent_loop.py`, `controllers.py`, `state_graph.py`, `schemas.py` |
| Tools and contracts | `tools.py`, `tool_contracts.py`, `context.py`, `literature.py` |
| Evaluation and audit | `evals.py`, `benchmark.py`, `comparison.py`, `eval_report.py`, `audit.py`, `source_safety.py`, `source_validation.py` |
| Reports and dashboard | `reporting.py`, `review_workbench.py`, `portfolio_index.py`, `quality_gate.py` |
| Service boundary | `service_interface.py`, `service_runtime.py`, `service_api.py`, `service_mcp.py` |

## Core Commands

Run the full local quality gate:

```bash
make check
```

Run unit tests only:

```bash
make test
```

Regenerate the dashboard:

```bash
make portfolio-index
```

Serve the dashboard locally:

```bash
make serve-portfolio
```

Write JSON quality-gate output:

```bash
make check-json
```

Generate a promotion decision packet:

```bash
PYTHONPATH=src python -m bioml_agent.cli \
  "Should we try retrieval-aware conditioning for promoter expression prediction?" \
  --experiment-log examples/epicetra_experiment_log.json \
  --eval \
  --audit \
  --trace-out traces/promotion-run.json \
  --report-out traces/promotion-decision-packet.md \
  --report-html-out traces/promotion-decision-review.html
```

Regenerate the review workbench:

```bash
PYTHONPATH=src python -m bioml_agent.review_workbench \
  traces/promotion-run.json \
  --out traces/review-workbench.html
```

Run the benchmark comparison:

```bash
PYTHONPATH=src python -m bioml_agent.benchmark benchmarks/eval_cases.jsonl
PYTHONPATH=src python -m bioml_agent.comparison benchmarks/eval_cases.jsonl
```

Validate tool contracts:

```bash
PYTHONPATH=src python -m bioml_agent.tool_contracts --validate
```

## Optional Integrations

The default project runs offline and deterministically. I keep optional adapters separate:

```bash
python -m pip install -e '.[llm]'
python -m pip install -e '.[api]'
python -m pip install -e '.[mcp]'
```

FastAPI-style adapter:

```bash
PYTHONPATH=src python -m bioml_agent.service_api serve --host 127.0.0.1 --port 8787
```

MCP-style adapter:

```bash
PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport stdio
PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport streamable-http
```

## Repository Layout

```text
.
├── README.md
├── README.zh-CN.md
├── Makefile
├── pyproject.toml
├── src/bioml_agent/
├── tests/
├── benchmarks/
├── examples/
└── traces/
```

I intentionally keep a curated set of generated artifacts in `traces/`. Runtime scratch files, caches, bytecode, virtual environments, and local environment files are ignored.

## Current Status

This repository is a local implementation rather than a hosted production service. The default path is deterministic and offline. Optional LLM/API/MCP integrations are scaffolded and documented so I can inspect the boundary without requiring external credentials.

## Chinese README

A Chinese version with the same content is available at [`README.zh-CN.md`](README.zh-CN.md).

## References and Borrowed Patterns

TracePilot is not a fork of any single framework. I used existing agent projects and documentation as design references, then kept the implementation small, offline, and inspectable.

| Source | What I studied | What I borrowed for TracePilot |
| --- | --- | --- |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) and [Agents SDK docs](https://openai.github.io/openai-agents-python/) | Tools, guardrails, handoffs, sessions, tracing, MCP integration, and human-in-the-loop concepts | Controller/tool/trace vocabulary, OpenAI-compatible tool specs, and a future SDK integration path |
| [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | Workflow vs agent tradeoffs, simple composable patterns, and transparent tool interfaces | Deterministic workflow first, model autonomy later, with tool quality treated as part of agent design |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) and [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview) | Stateful long-running agents, durable execution, checkpointing, persistence, and human review of state | `AgentGraphState`, stop-after checkpoints, checkpoint-on-error, and resume-from-checkpoint behavior |
| [modelcontextprotocol](https://github.com/modelcontextprotocol) and [MCP docs](https://modelcontextprotocol.io/docs/getting-started/intro) | Tool/data/workflow protocol boundaries, client/server separation, discovery, and permission surfaces | A tool manifest and approval policy that can later be exposed as MCP tools, resources, and prompts |
| [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas) and [Ragas docs](https://docs.ragas.io/en/stable/) | Systematic evaluation for RAG/agent applications, metrics, datasets, and regression loops | Fixed benchmark corpora, grounding checks, abstention cases, source-poisoning cases, audit rubric, and baseline comparison |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | Difference between autonomous crews and controlled flows | A deliberate choice to avoid decorative multi-agent roles until role decomposition has measurable value |
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | Workspace-agent product shape: files, commands, local runtime, long tasks, and user-facing surfaces | Generated artifact index, review workbench, and a path toward a richer UI |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | Minimal inspectable abstractions, code-agent behavior, and sandbox concerns | A lightweight local loop and explicit permission boundaries before any future experiment execution |
| [microsoft/autogen](https://github.com/microsoft/autogen) | Multi-agent API history, framework status, and migration risk | Historical design context only; TracePilot does not depend on AutoGen as a core framework |

The main design choice from these references is conservative: I start with typed local tools, traceable state, and evaluation before adopting a heavier runtime.
