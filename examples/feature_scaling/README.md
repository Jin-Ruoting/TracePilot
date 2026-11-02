# MLP 标准化实验

用已知方法验证从假设、代码变体到实验审查的完整流程；不作为新颖性或自主选题能力的证明。

```bash
python -m pip install -e '.[research]'
python examples/feature_scaling/evaluate.py
```

执行 30 次实验，再完整串行重放 30 次，比较指标和逐样本预测。默认恢复已有项目；重新执行需添加 `--project-id 新名称`。

| 项目 | 固定设置 |
| --- | --- |
| 数据 | sklearn 内置 Wine、Breast Cancer、Digits |
| 对照 | 原始输入 MLP / 训练集拟合 StandardScaler + 同一 MLP |
| 种子 | 11、23、37、53、71；划分种子 20261102 |
| 划分 | 约 60% 训练、20% 验证、20% 保留测试 |
| 训练 | 隐藏层 32、Adam、batch 64、学习率 0.001、80 epoch |
| 评测 | 验证集 accuracy；均值、样本 SD、配对 bootstrap |
| 初筛 | 配对完整且改善区间下界 > 0.005 |

标准化仅在训练集拟合。测试集不参与评价或选择。80 epoch 是固定比较预算，收敛提示保留在指标文件中。重放不作为额外独立种子参与统计。

`project.json` 固定规格；`workspace/` 存放原始代码；变体只修改 `preprocess.py`。
结果包含代码/数据/协议哈希、日志、预测、逐次 CSV 和 HTML 报告。

可选绘图：

```bash
python -m pip install -e '.[plots]'
python examples/feature_scaling/figures/gen_fig_pilot.py outputs/research/feature-scaling-v2
```

服务器由受限启动器提供独立的代码目录和结果目录。绘图参数应指向实际报告目录。

方法依据：[sklearn MLP 指南](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)。
