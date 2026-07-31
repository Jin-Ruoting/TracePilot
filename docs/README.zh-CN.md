[English](../README.md) | [简体中文](README.zh-CN.md)

# TracePilot

TracePilot 是一个本地优先、证据驱动的通用技术研究与决策单 Agent。它结合联网或冻结资料与本地项目上下文，生成带精确引用的行动方案，并且只在人工明确批准后执行文件修改或命令。

默认场景是依赖升级、API 迁移、架构选型和故障诊断。BioML 仅作为可选 taskpack 和兼容示例保留，不再是核心产品领域。

> **当前状态：**本仓库是本地实现，不是已托管的生产服务。默认路径可离线、确定性运行，因此测试和演示不依赖外部凭据；实时 LLM 与 FastAPI 是可选能力。

TracePilot 为独立设计与实现的项目。

## 适用对象

TracePilot 面向需要处理代码仓库技术任务的开发者、维护者和技术负责人，典型场景包括依赖升级、API 迁移、架构决策和故障诊断。

这类任务通常不只是回答一个问题，而是需要检查当前代码、检索带版本的技术文档、核对多份证据、规划有依赖的步骤、等待审批、执行变更并验证结果。TracePilot 使用 Agent 处理这种开放式协作，同时把权限、审批和执行规则交给确定性代码。简单事实查询仍然更适合直接搜索或使用轻量 RAG。

## 项目体现什么

- **完整 RAG**：来源接入、结构化分块、内容和版本元数据、FTS5 与 embedding 混合检索、融合、重排、引用绑定、faithfulness 检查和 prompt injection 检查。
- **清晰的 Agent 架构**：LLM、Planning、Memory、Tool、Execution 五个模块有独立职责和类型化契约。
- **DAG 规划**：互不依赖的只读研究步骤可以并行；证据不足或工具失败时，可以在上限内局部重规划。
- **受控执行**：工具必须声明 `read`、`internal_write` 或 `mutation` effect；mutation 必须绑定冻结参数哈希并经过人工审批。
- **可维护记忆**：working state、不可变 run history 与用户确认的 decision memory 使用 SQLite WAL 保存，并按 workspace 隔离。
- **可恢复和可检查**：checkpoint、任务恢复、trace、引用、审批记录与执行结果都可以回看。

## 架构

```text
TaskRequest
  -> 来源与 workspace 快照
  -> 混合 RAG 检索
  -> 单 Agent 生成 DAG
  -> 并行执行只读工具
  -> 证据覆盖检查与有限局部重规划
  -> ActionPlan
  -> Policy Gate 与人工审批
  -> 执行经批准的 mutation（命令以 workspace 为 cwd）
  -> 结果验证
  -> 带引用报告与记忆更新
```

| 模块 | 职责 |
| --- | --- |
| **LLM Gateway** | 结构化生成、模型与 Prompt 版本、重试、token/cost 元数据，以及确定性测试实现 |
| **Planning Engine** | 有依赖关系的 `TaskPlan` DAG、规划轮数上限、只读步骤并行和局部重规划 |
| **Memory Store** | SQLite-backed working memory、不可变 run memory，以及明确由用户提升的 decision memory |
| **Tool Registry** | 类型 schema、effect class、超时、幂等性、权限与状态依赖 |
| **Execution Runtime** | 自动只读/内部写、审批后 mutation、workspace 路径检查和结果验证 |

TracePilot 明确选择单 Agent。DAG 并行已经能让独立步骤并发执行，不需要把上下文复制给多个角色 Agent。只有 benchmark 稳定证明 Multi-Agent 的质量收益大于额外延迟、成本、恢复和评测复杂度时，才会重新考虑这个选择。

更完整的组件边界、生命周期、RAG、记忆与审批协议见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## 快速开始

TracePilot 需要 Python 3.11 或更高版本。

```bash
python -m pip install -e .
tracepilot ingest examples/technical_decisions/architecture_choice \
  --frozen-sources benchmarks/general_sources.jsonl
tracepilot run "为这个本地任务服务选择持久化方案" \
  --workspace examples/technical_decisions/architecture_choice \
  --frozen-sources benchmarks/general_sources.jsonl
```

运行确定性验证套件和通用 benchmark：

```bash
make check
make benchmark
```

安装可选 API 依赖并启动本地 Workbench：

```bash
python -m pip install -e '.[api]'
tracepilot serve --host 127.0.0.1 --port 8787
```

然后打开 `http://127.0.0.1:8787`。Workbench 提供任务列表、DAG 时间线、证据与引用、记忆记录、审批窗口、patch 或命令预览、执行日志和最终报告；任务更新通过 SSE 推送。

默认实现可以离线、确定性运行。使用可选 OpenAI gateway 时安装：

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

这些命令对应完整任务生命周期：

| 命令 | 用途 |
| --- | --- |
| `tracepilot ingest` | 解析并索引 workspace、URL 或冻结来源语料 |
| `tracepilot run` | 创建并运行一个技术调研或决策任务 |
| `tracepilot resume` | 从持久化 task 与 checkpoint 状态恢复任务 |
| `tracepilot approve` | 批准、修改或拒绝待执行 mutation |
| `tracepilot memory list` | 查看 workspace 内的记忆记录 |
| `tracepilot memory promote` | 把已完成决策保存为可复用记忆 |
| `tracepilot memory supersede` | 用新决策替代旧结论，但不删除历史 |
| `tracepilot memory delete` | 删除指定记忆 |
| `tracepilot serve` | 启动 FastAPI 与 SSE Workbench |

跨进程执行时，状态命令必须使用同一个数据库与 workspace identity，才能定位到正确任务：

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

替代 stale memory 时必须提供一个已完成的当前任务，确保新记录绑定最新来源与 workspace fingerprint。

## 检索与证据

知识索引与 Agent 记忆严格分离：

```text
web / URL / 本地 workspace
  -> 解析与结构化分块
  -> 内容哈希、来源版本与 provenance
  -> SQLite FTS5 与确定性 embedding
  -> Query 拆分
  -> BM25 与向量召回
  -> 去重与融合
  -> 结构化重排
  -> Context 组装
  -> claim 与 chunk 引用绑定
  -> coverage、faithfulness 与 poisoning 检查
```

最终报告中每条有依据的事实都必须指向 `EvidenceChunk`。证据缺失或相互冲突时，Agent 会标记不确定性或 abstain，而不是把无依据内容写成结论。

离线 faithfulness gate 是保守的词法启发式：检查 clause 级覆盖率、数字、否定与 poisoning 一致性；它不等价于通用语义蕴含模型。

## 记忆策略

TracePilot 不实现不受约束的个人长期记忆：

- **Working memory** 在每个成功任务步骤后自动 checkpoint。
- **Run memory** 在任务结束时追加计划、工具结果、trace 与失败信息。
- **Decision memory** 只有在用户明确选择提升后才会写入。
- workspace commit、文件哈希或来源版本变化时，相关记忆会标为 stale。
- 新结论通过 `supersedes_id` 替代旧结论，历史不会被静默覆盖。
- 默认只在同一 workspace 内召回，记录可以查看、禁用、替代或删除。
- 被召回的 decision 只作为规划提示，并触发基于当前来源的重新验证；它不能直接作为证据或引用。

## 工具与审批

首批只读工具覆盖 web search、URL 获取、workspace 索引、索引搜索、文件读取、代码搜索和 Git 检查。`propose_patch` 只生成内部 artifact；真正应用 patch、运行任意命令、修改用户文件或写入外部服务都属于 `mutation`。

read 和内部 trace/index/checkpoint 可以自动执行。mutation 必须经过用户对精确冻结参数的审批；参数哈希变化会使原审批失效，并且 mutation 不会自动重试。

内置 `run_command` 使用精确 argv，并把工作目录限定在 workspace，但它不是操作系统级沙箱。对不可信命令需要进程级隔离时，应在外部容器或 sandbox profile 中运行。

线程超时只用于 read/internal 工具。mutation handler 会等待结束，避免系统已报告 timeout 但副作用仍在后台继续；自定义 mutation 必须自行实现可取消超时。内置命令工具超时时会终止对应进程组。

结构化 patch 的每个条目都必须携带当前文件的 `expected_sha256`；新建文件使用空字节的 SHA-256。写入前会再次校验，避免过期审批静默覆盖期间产生的新修改。

## Benchmark

通用 benchmark 覆盖四类计算机任务：

- 依赖升级
- API 迁移
- 架构选型
- 测试或运行时故障诊断

fixture 使用冻结来源快照与小型本地 workspace，因此回归测试不依赖实时搜索。通用语料中的产品名称和案例内容均为合成测试数据。评测覆盖检索与引用、来源过期、prompt injection、DAG 合法性、恢复、审批强制、幂等性、任务隔离和服务重启恢复。这些结果只说明回归覆盖情况，不代表优于其他 Agent 或具备生产环境性能。

## 兼容性

`tracepilot` 是 canonical package 和 CLI。兼容期内仍保留旧 `bioml_agent` package、`bioml-agent` 命令、v1 BioML checkpoint 映射与 fixture：

```bash
bioml-agent "Should we try graph transfer?"
PYTHONPATH=src python -m bioml_agent.cli "Should we try graph transfer?"
```

BioML 类型与示例位于 `tracepilot.taskpacks.bioml` 和 `examples/bioml`。它只用于演示领域 taskpack 如何复用通用 runtime，不进入默认流程和主要 benchmark。

报告、trace、数据库、缓存和工作台导出文件均在本地生成，不纳入版本控制。Quality Gate 会把验证结果作为 CI artifact 提供。

## 仓库结构

```text
.
├── src/tracepilot/              # 通用 Agent、RAG、记忆、工具、执行与 API
├── src/bioml_agent/             # 兼容 package 与旧命令
├── examples/technical_decisions/
├── examples/bioml/              # 可选 taskpack 示例
├── benchmarks/                  # 通用与旧版冻结 fixture
├── docs/
│   ├── ARCHITECTURE.md
│   └── README.zh-CN.md
└── tests/
```

英文版见 [`README.md`](../README.md)。
