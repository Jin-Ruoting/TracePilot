# TracePilot

我构建 TracePilot，是为了做一个面向开放技术决策的 source-grounded planning agent。这个项目关注简单聊天机器人很难展示的工程表面：类型化工具契约、基于证据的规划、持久状态、可恢复 trace、评估覆盖、人工审批 gate，以及服务边界就绪度。

当前 demo 使用 gene-expression experiment-planning 场景作为高约束 case study。我用这个领域提供真实约束，但系统本身是一个通用 agent engineering 项目：核心思想是工具边界、状态、provenance、评估和受控服务接口。

## 我为什么构建它

很多 agent 示例停留在“LLM 加几个工具”。我希望 TracePilot 把工程表面显式呈现出来：

- **工具使用是显式的**：我定义工具 schema、确定性 fallback、结构化输出和可审计 trace。
- **证据是一等对象**：我把规划绑定到 source card、实验上下文、claim、citation 和 provenance check。
- **状态可以恢复**：我可以把 run 停在中间节点，写出 checkpoint，再从 checkpoint 继续。
- **评估不是单个 demo prompt**：我使用固定语料、abstention case、source-poisoning case、baseline comparison 和本地质量门禁。
- **内置人工复核**：我生成 review packet，暴露 promotion gate、audit finding 和 approval checkpoint。
- **边界可以产品化**：我可以把同一套 contract 渲染成 CLI workflow、静态 review 页面、FastAPI-style endpoint 和 MCP-style descriptor。

## 快速开始

默认本地流程只需要 Python 3.11+。

```bash
python -m pip install -e .
make check
make serve-portfolio
```

如果你的机器把 `python3` 映射到较旧的系统 Python，请显式使用 Python 3.11+ 解释器：

```bash
make PYTHON=python check
make PYTHON=/path/to/python3.12 check
```

然后打开：

```text
http://127.0.0.1:8765/traces/index.html
```

我也可以直接运行主 planner：

```bash
tracepilot "Should we try retrieval-aware conditioning?"
```

内部包名仍保留为 `bioml_agent` 以保持兼容：

```bash
PYTHONPATH=src python -m bioml_agent.cli "Should we try graph transfer?"
PYTHONPATH=src python -m unittest discover -s tests
```

## 项目 Dashboard

我把生成的 artifact 保存在 `traces/` 下，让仓库同时展示源代码和可复现输出。

| Artifact | Purpose |
| --- | --- |
| `traces/index.html` | 链接生成项目 artifact 的静态 dashboard |
| `traces/quality-gate-report.md` | 本地质量门禁结果，覆盖测试、工具契约、评估、服务报告和生成 artifact |
| `traces/promotion-decision-review.html` | 静态 decision review，包含 evidence、citation、audit result、approval gate 和 trace |
| `traces/review-workbench.html` | 产品风格 review workbench，用于检查 evidence coverage、claim grounding、audit、eval 和 trace |
| `traces/eval-coverage-report.md` | 跨 benchmark corpus 和 guardrail 的覆盖总结 |
| `traces/reference-catalog-report.md` | 生成的项目参考来源和借鉴设计模式总结 |
| `traces/service-interface-report.md` | 工具契约到 service-style interface 的映射 |
| `traces/service-runtime-demo.md` | session-backed runtime route demo |
| `traces/service-api-readiness.md` | FastAPI adapter plan 和 OpenAPI preview |
| `traces/service-mcp-readiness.md` | MCP tool/resource/prompt/transport readiness report |

## 架构

TracePilot 围绕一个 deterministic planning controller 组织，并带有可选 LLM 和 service adapter。

```text
Question
  -> context provider
  -> source retrieval / frozen literature provider
  -> transfer-fit scoring
  -> plan synthesis
  -> evaluation and audit
  -> review packet / HTML workbench / service reports
```

关键模块：

| Area | Files |
| --- | --- |
| Controller and state | `src/bioml_agent/agent_loop.py`, `controllers.py`, `state_graph.py`, `schemas.py` |
| Tools and contracts | `tools.py`, `tool_contracts.py`, `context.py`, `literature.py` |
| Evaluation and audit | `evals.py`, `benchmark.py`, `comparison.py`, `eval_report.py`, `audit.py`, `source_safety.py`, `source_validation.py` |
| Reports and dashboard | `reporting.py`, `review_workbench.py`, `portfolio_index.py`, `quality_gate.py` |
| Service boundary | `service_interface.py`, `service_runtime.py`, `service_api.py`, `service_mcp.py` |

## 核心命令

运行完整本地质量门禁：

```bash
make check
```

只运行单元测试：

```bash
make test
```

重新生成 dashboard：

```bash
make portfolio-index
```

本地启动 dashboard：

```bash
make serve-portfolio
```

写出 JSON 质量门禁结果：

```bash
make check-json
```

生成 promotion decision packet：

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

重新生成 review workbench：

```bash
PYTHONPATH=src python -m bioml_agent.review_workbench \
  traces/promotion-run.json \
  --out traces/review-workbench.html
```

运行 benchmark comparison：

```bash
PYTHONPATH=src python -m bioml_agent.benchmark benchmarks/eval_cases.jsonl
PYTHONPATH=src python -m bioml_agent.comparison benchmarks/eval_cases.jsonl
```

校验工具契约：

```bash
PYTHONPATH=src python -m bioml_agent.tool_contracts --validate
```

## 可选集成

默认项目以离线、确定性方式运行。我把可选 adapter 单独保留：

```bash
python -m pip install -e '.[llm]'
python -m pip install -e '.[api]'
python -m pip install -e '.[mcp]'
```

FastAPI-style adapter：

```bash
PYTHONPATH=src python -m bioml_agent.service_api serve --host 127.0.0.1 --port 8787
```

MCP-style adapter：

```bash
PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport stdio
PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport streamable-http
```

## 仓库结构

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

我有意在 `traces/` 中保留一组精选生成 artifact。运行时 scratch file、cache、bytecode、virtual environment 和本地环境文件会被忽略。

## 当前状态

这个仓库是本地实现，而不是已托管的生产服务。默认路径是确定性且离线的。可选 LLM/API/MCP 集成已经以 scaffold 和文档形式保留，所以我可以在不需要外部凭据的情况下检查边界设计。

## 英文 README

同内容英文版见 [`README.md`](README.md)。

## 参考来源和借鉴模式

TracePilot 不是任何单一框架的 fork。我把现有 agent 项目和文档作为设计参考，然后把实现保持在小型、离线、可检查的状态。

| Source | 我研究了什么 | 我在 TracePilot 中借鉴了什么 |
| --- | --- | --- |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) 和 [Agents SDK docs](https://openai.github.io/openai-agents-python/) | tools、guardrails、handoffs、sessions、tracing、MCP integration 和 human-in-the-loop 概念 | controller/tool/trace 词汇、OpenAI-compatible tool specs，以及未来 SDK 集成路径 |
| [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | workflow 与 agent 的取舍、简单可组合模式、透明工具接口 | 先做 deterministic workflow，再引入模型自主性，并把工具质量视为 agent 设计的一部分 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) 和 [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview) | stateful long-running agents、durable execution、checkpointing、persistence 和 human review of state | `AgentGraphState`、stop-after checkpoint、checkpoint-on-error 和 resume-from-checkpoint 行为 |
| [modelcontextprotocol](https://github.com/modelcontextprotocol) 和 [MCP docs](https://modelcontextprotocol.io/docs/getting-started/intro) | tool/data/workflow protocol boundary、client/server separation、discovery 和 permission surface | tool manifest 和 approval policy，后续可以暴露成 MCP tools、resources 和 prompts |
| [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas) 和 [Ragas docs](https://docs.ragas.io/en/stable/) | 面向 RAG/agent 应用的系统化评估、metrics、datasets 和 regression loops | 固定 benchmark corpora、grounding checks、abstention cases、source-poisoning cases、audit rubric 和 baseline comparison |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | autonomous crews 与 controlled flows 的区别 | 在 role decomposition 有可衡量价值前，避免装饰性 multi-agent roles |
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | workspace-agent product shape：files、commands、local runtime、long tasks 和 user-facing surfaces | 生成 artifact index、review workbench，以及未来更完整 UI 的方向 |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | minimal inspectable abstractions、code-agent behavior 和 sandbox concerns | 轻量 local loop，以及未来执行实验前的显式权限边界 |
| [microsoft/autogen](https://github.com/microsoft/autogen) | multi-agent API history、framework status 和 migration risk | 只作为历史设计背景；TracePilot 不把 AutoGen 作为核心框架依赖 |

这些参考共同导向一个保守的设计选择：我先做类型化本地工具、可追踪状态和评估，再考虑采用更重的 runtime。
