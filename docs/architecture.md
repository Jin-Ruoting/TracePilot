# 架构与接口

一个持久化协调器管理 Research、Coding、Review 三类职责。实验任务由冻结规格驱动，模型接口返回可检查提案，尚未启用模型自动连续试验循环。

| 模块 | 职责 |
| --- | --- |
| ResearchWorker | 生成可证伪假设或补丁，拒绝未知引用、未知文件与过期哈希 |
| CascadeRouter | 明确事件走规则；可选 Jev Choice；低置信度返回 escalate |
| CodingWorker | 冻结源码、检查补丁、构建独立变体，不修改原始工作区 |
| ResearchCoordinator | 冻结预算、原子领取作业、恢复、取消与事件记录 |
| Supervisor | 独立执行进程、文件锁去重、超时清理、指标哈希 |
| ReviewWorker | 审查完整配对、协议及划分一致性，生成初步判断 |
| Runtime | Linux 上限制可写目录、CPU 集合和 GPU 设备 |

## 实验契约

`ResearchSpec` 固定目标、假设、源码、argv 命令、数据集、种子、变体、指标、划分和预算。变体需要原文件 SHA-256；基线保持原始代码。更改定义时使用新项目 ID。

作业状态：`queued → starting → running → succeeded / failed / timed_out / cancelled / invalid / lost`。
状态变更与事件在同一 SQLite 事务提交。并发槽和运行时间预算在领取时预留，终态不可改写。

监督进程锁防止重复执行；CLI 恢复后回收已保存结果。取消意图先持久化。监督进程丢失会暂停新增作业，不自动重复未知结果的命令。远程受限命名空间内需保留协调进程，使用 `--wait`；长任务应将整个启动器放到后台运行。

`--output-root` 保存代码副本和控制账本；`--artifacts-root` 可将日志、指标及报告分离到数据盘。普通本地副本不是 OS 沙箱；服务器受限入口使用独立的文件系统与设备隔离。

审查要求所有声明种子成功并可配对，主要指标为有限值，且数据划分、协议和指标文件哈希有效。缺失或不一致时保留 `inconclusive`。少量种子的 bootstrap 只描述固定划分下的训练随机性；`promising` 表示值得继续验证，不等于新颖性成立。

## 模型接口

生成模型采用显式配置的 chat-completions JSON 端点。Jev 使用 System One Choice；`confidence` 与最高选项概率分别保存。规则事件不额外调用模型，阈值需要后续用项目标注集校准。

```bash
tracepilot research context examples/feature_scaling/workspace \
  --goal "研究 feature scaling 对固定预算 MLP 的影响" \
  --evidence examples/feature_scaling/README.md --out outputs/context.json

tracepilot research propose outputs/context.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model YOUR_MODEL --out outputs/hypotheses.json
```

`propose-patch` 使用含代码全文与 SHA-256 的同类上下文。密钥从 `OPENAI_API_KEY` 或指定环境变量读取，提案不会直接触发执行。

```bash
tracepilot research route event.json \
  --endpoint https://api.typesafe.ai/v1/systemone --model jev-latest \
  --confidence-threshold 1.0 --out outputs/route.json
```

Jev 密钥变量默认为 `TYPESAFE_API_KEY`。`route` 返回路由结果；`escalate` 不会自动调用其他模型。当前没有实际模型的路由准确率或选题质量结论。
