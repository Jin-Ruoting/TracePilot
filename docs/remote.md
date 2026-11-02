# GitHub 同步与受限远程运行

项目代码只通过 GitHub clone / pull 同步，不上传工作区压缩包。运行数据、环境、缓存和实际配置不提交到 GitHub。

## 目录

在各自授权的父目录下建立独立 `tracepilot/` 子目录：

```text
代码父目录/tracepilot/
  repo/                        Git checkout
  config/runtime.local.json    本机策略
  config/router.local.json     Kev-4B 端点与权重版本
  services/kev/                从作者 GitHub 仓库获取的服务代码
  workspaces/                  控制账本与实验代码副本
  home/                        受限进程的配置与小型注册文件

数据父目录/tracepilot/
  conda/envs/agent/             项目 conda 环境
  conda/pkgs/                  conda 包缓存
  services/kev/env/            Kev 服务依赖；与实验环境隔离
  models/huggingface/          Kev 与底座模型权重
  datasets/  checkpoints/
  experiments/                日志、指标、预测与报告
  cache/  tmp/  logs/
```

配置参考 [runtime.example.json](../config/runtime.example.json)。CPU ID 应覆盖不超过一半物理核及逻辑 CPU；每核选择一个硬件线程是更保守的设置。GPU 使用单个 UUID，并在启动时核验为 RTX 5090。

创建 `agent` 环境时，把 `CONDA_ENVS_PATH` 指向数据目录的 `conda/envs`，`CONDA_PKGS_DIRS` 指向 `conda/pkgs`；临时文件和缓存也放在数据目录。环境创建、安装和同步应使用相同的目录及 CPU 限制。不要执行 `conda init` 或修改用户全局 shell 配置。

首次准备好环境并从 GitHub clone 后：

```bash
bash scripts/agent.sh -- python -m pip install -e '.[research]'
bash scripts/agent.sh -- python examples/feature_scaling/evaluate.py --project-id remote-pilot
```

入口自动使用 `agent` 环境的 Python，代码与结果路径由策略注入。默认不开放 GPU。需要使用配置中的唯一 GPU 时：

```bash
bash scripts/agent.sh --gpu -- python YOUR_TRAINING_SCRIPT.py
```

## Kev-4B 服务

使用 [jaredpalmer/kev](https://github.com/jaredpalmer/kev) 的 `kev-1.0` 版本与 `jaredpalmer/kev-4b@v1.0` 权重。Kev 的 NumPy、sklearn 依赖与示例实验不同，服务依赖安装到数据目录的独立 Python 环境；主项目继续使用 conda `agent`。

在服务器仓库目录执行：

```bash
cp config/router.example.json ../config/router.local.json
bash scripts/agent.sh -- python -m tracepilot.routing.kev install
bash scripts/agent.sh -- python -m tracepilot.routing.kev download
bash scripts/agent.sh --gpu -- python -m tracepilot.routing.kev serve
```

安装仅从 GitHub 获取源码并安装服务依赖，不运行测试；`download` 将模型及其固定版本底座下载到数据目录，不执行推理。`serve` 使用本地缓存离线加载权重，绑定 `127.0.0.1:8009`，使用指定的一张 GPU。默认关闭 CUDA Graph 与可选融合算子，限制前缀缓存；没有自行训练或修改模型。

`run`、`resume` 与 `route` 自动使用 `config/router.local.json`。路由模型服务与实验进程分别运行；当前路由判断仍只记录到协调器账本。

## 同步

更新代码使用受限入口，并以 fast-forward 方式同步：

```bash
bash scripts/agent.sh -- git pull --ff-only origin main
```

## 限制如何生效

- Bubblewrap 将主机根目录设为只读，仅绑定两个项目子目录为可写；符号链接不能把配置目录转向其他位置。
- Linux CPU affinity 限定实际 CPU 集合，seccomp 禁止子进程重新扩大或更改 affinity。
- 默认隔离全部 NVIDIA 计算设备；`--gpu` 只绑定配置中的一个 GPU 设备及公共控制设备。
- conda、pip、模型、CUDA、绘图和临时文件缓存都重定向到数据目录。
- 不支持 Bubblewrap 或配置越界时直接报错，不退回无限制执行。

该启动器针对 x86-64 Linux。子进程只能继承已经收紧的资源与文件系统边界；运行者仍应检查实验代码和外部网络操作。这里不包含 GPU 任务排队或独占调度，启动前需确认共享服务器的设备可用性。

使用 `--wait` 完成整批实验。需要断开 SSH 时，将整个受限启动器作为后台任务保留，日志写入数据目录；不要仅启动队列后关闭其命名空间。
