# TracePilot

以代码实验支撑 AI / 机器学习科研选题的 Agent。

```text
证据与代码 → 可证伪假设 → 实验规格 → 隔离代码变体 → 实验 → 配对审查
```

当前实现模型接口与可复现实验闭环。示例使用预定义假设；决策后端采用自托管 Kev-4B，生成模型需另行配置。

## 快速开始

推荐 Python 3.12，支持 macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[research]'
make test
make demo
```

示例执行 **3 数据集 × 2 变体 × 5 种子**，再串行重放以验证一致性。已有项目恢复记录；用 `--project-id` 指定新 ID 可重新训练。

## 主要能力

- **证据与提案**：复用本地 RAG，提供假设和补丁生成接口，校验引用与文件版本。
- **路由**：规则优先，Kev-4B 提供 Choice 判断；低置信度返回升级标记。当前记录决策，尚未自动派工。
- **实验**：冻结代码、划分和预算，独立进程执行，持久化状态，支持恢复、超时和取消。
- **审查**：完整种子配对、协议检查、均值/SD 与探索性区间，失败记录保留。
- **远程运行**：通过 GitHub 同步代码；受限启动器隔离写入目录、限制 CPU，最多开放一张 RTX 5090。

```bash
tracepilot research run examples/feature_scaling/project.json --execute --wait
tracepilot research status feature-scaling-v2
tracepilot research resume feature-scaling-v2 --wait
tracepilot research report feature-scaling-v2
```

## 目录

```text
src/tracepilot/
  research/       规格、账本、上下文、CLI 与报告
  coordinator/    作业协调与恢复
  workers/        假设、代码、审查
  routing/        规则与决策模型接口
  experiments/    执行与监督进程
  rag/            来源、检索与引用
  runtime.py      Linux 目录与资源限制
examples/feature_scaling/  唯一完整实验示例
tests/                    核心行为测试
scripts/agent.sh           服务器受限入口
config/                   资源与 Kev 路由配置模板；实际配置不入库
docs/                     架构与部署说明
```

[架构与模型接口](docs/architecture.md) · [服务器部署](docs/remote.md) · [实验协议](examples/feature_scaling/README.md)

结果默认写入 `outputs/research/`，包含 JSON、CSV、Markdown 和 HTML。运行产物、环境、缓存与本机配置不进入 Git。
