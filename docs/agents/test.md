# 测试与检查记录

## 2026-09-01：手动 Python 环境验证（计划）

- 登录节点轻量检查：`.venv/pyvenv.cfg`、Python/pip 版本、canonical 19 个直接依赖版本、`pip check`、torch CUDA build 与 arch list。
- 预期：Python 3.10.12；torch 2.11.0+cu128 / CUDA 12.8；e3nn 0.4.4；其他直接依赖与 `docs/requirements.txt` 一致；`pip check` 无错误。
- 不在登录节点执行 pytest、MACE forward/backward 或 CUDA kernel；功能验证仅通过版本化 Slurm scripts。
- 环境通过后提交 unit、CPU smoke、GPU smoke；任何失败都保留原始日志并据证修复。

### 实际结果

- 连接前 pull：第 4 次连接成功，远端 commit `3249b76ae5f5dfe544a1508227c876ebfa173aac`。
- Python 3.10.12、pip 26.2.1；canonical 19 个直接依赖版本均匹配，`pip check` 为 `No broken requirements found`。
- 失败门控：torch `2.11.0+cu130`、build CUDA 13.0、驱动 570.211.01，`cuda_available=False`。未运行 pytest、模型或 CUDA kernel。

## 2026-09-01：cu128 索引修复（计划）

- 静态检查：requirements 必须同时包含清华主索引与官方 PyTorch cu128 extra index，且 torch 固定 2.11.0。
- CPython 3.10 manylinux wheelhouse dry-run 必须选择 `torch 2.11.0+cu128` 并解析全部依赖。
- `bash -n ELoRA/scripts/slurm/setup_readiness.sbatch` 必须通过。
- 配置修改不影响业务代码；本地 readiness 回归仍运行 40 项以防安装入口漂移。
- Guqq setup 集成必须报告 torch `2.11.0+cu128`、CUDA 12.8、`pip check` 与全部 import 成功。

### commit 前实际结果

- 双索引与 `torch==2.11.0` PowerShell 静态断言：通过，退出码 0。
- CPython 3.10 / manylinux 目标 pip dry-run：全部固定与传递依赖解析成功，明确选择 `torch-2.11.0+cu128`，退出码 0。
- `bash -n ELoRA/scripts/slurm/setup_readiness.sbatch`：通过，退出码 0。
- readiness 四文件回归：40 passed，退出码 0；临时目录为仓库 `.cache/pytest-cu128-fix`。

## 2026-09-01：editable build isolation 修复（计划）

- 根因重现证据：Job 203 已安装 `torch 2.11.0+cu128` 及全部固定依赖，但 `pip --no-index --no-deps -e ELoRA` 的隔离 build env 无法下载 `setuptools>=42`。
- 修复范围：editable install 增加 `--no-build-isolation`，复用 setup 前一步已安装的 setuptools/wheel；不改变依赖版本或业务代码。
- commit 前检查：SBATCH syntax、静态断言 editable 命令同时含 `--no-index --no-deps --no-build-isolation -e ELoRA`、40 项 readiness 回归。
- Guqq 集成：替代 setup job 必须通过 editable install、`pip check`、固定版本 import，并报告 torch cu128/CUDA 12.8。

### commit 前实际结果

- SBATCH shell syntax：通过，退出码 0。
- editable 命令精确静态断言：`--no-index --no-deps --no-build-isolation -e ELoRA` 唯一匹配，退出码 0。
- 本地真实 pip editable dry-run：build backend 与 editable metadata 均成功，结果为 `Would install mace-torch-0.3.5`，退出码 0。
- readiness 四文件回归：40 passed，退出码 0。

### Guqq setup 集成结果

- setup Job 204：成功；editable `mace-torch 0.3.5` 构建/安装通过，`pip check` 无错误。
- 固定环境：torch `2.11.0+cu128`、CUDA 12.8、e3nn 0.4.4、ASE 3.22.1、NumPy 1.26.4、SciPy 1.15.3、h5py 3.14.0、Matplotlib 3.10.7、Pandas 2.3.3。
- 下一集成门控：Slurm unit readiness、CPU smoke、RTX 5090 GPU smoke。

## 2026-09-01：最终 Guqq readiness 结果

- Job 204 setup：`COMPLETED`, `ExitCode=0:0`, runtime 00:04:26；环境与 import 门控通过。
- Job 205 unit：`COMPLETED`, `ExitCode=0:0`, runtime 00:00:17；40 passed。
- Job 206 CPU smoke：`COMPLETED`, `ExitCode=0:0`, runtime 00:00:05；forward/backward、checkpoint restore、JSON 均成功。
- Job 207 GPU smoke：`COMPLETED`, `ExitCode=0:0`, runtime 00:00:05；RTX 5090、CC 12.0、sm_120、torch CUDA 12.8、实际 CUDA kernel、forward/backward、checkpoint restore、JSON 均成功。
- SCP 回传核验：setup/unit/CPU/GPU stdout 均含 `status=success`；unit/CPU/GPU stderr 为 0 字节；两个 smoke JSON 可解析且字段匹配日志。

### 最终文档提交前检查

- `git diff --check`：通过。
- readiness 标记：精确包含 2 个 `readiness: ready`、0 个 `readiness: not_ready`；报告包含 Jobs 204–207。
- 回传日志：setup 包含 `pip check`、torch `2.11.0+cu128`、CUDA 12.8 与 `status=success`；unit 为 40 passed；CPU/GPU 均为 `status=success`。
- JSON 字段断言：CPU device/checkpoint restore 通过；GPU RTX 5090、CC 12.0、sm_120、torch CUDA 12.8、实际 kernel 与 checkpoint restore 全部通过。
- unit/CPU/GPU stderr 长度均为 0 字节。
- 上述检查通过后按授权删除本地回传副本；服务器原始日志与 smoke JSON 保留，可再次回传。

## 2026-09-01：Goal 0 完成审计修复（计划）

- 梯度白名单：构造同时含冻结/可训练参数的模型，验证 `evaluate()` 正常返回和异常退出后均精确恢复原始 `requires_grad` 状态。
- 数据统计：验证 `class_counts.csv` 对每个分类层级提供逐类比例、split 计数及 atoms/energy/force 数值摘要；同层比例和为 1，split 计数与 retained 数据一致。
- router：对旋转、平移和原子置换后的同一构型使用冻结 symmetry/random 标签，验证 expert id 不变；learned router 仍只接受不变量特征。
- Slurm unit 脚本：静态断言命令为完整 `python -m pytest ELoRA/tests -q`，而非选定文件子集；`bash -n` 必须通过。
- 本地提交前门控：新增针对性测试、完整 `ELoRA/tests`、CPU smoke、SBATCH syntax、`git diff --check` 全部退出码 0。
- Guqq 集成门控：同一已推送提交上通过完整 unit Slurm、CPU smoke 和 RTX 5090 GPU smoke；记录 job ID、脚本 SHA-256、资源、环境、stdout/stderr/JSON 路径、终态与退出码。
- 完整套件新增发现的兼容单元：未配置 adapter 的新模型须能 strict-load 旧 foundation state dict；旧 pickle 中缺少 adapter 属性的 contraction 经加载升级后保持原输出，并能随后配置 adapter。
- 完整套件跨平台入口：测试子进程的 `PYTHONPATH` 使用 `os.pathsep`；Windows 与 Guqq/Linux 均能导入当前 checkout，不跳过原断言。
- foundation training 的 2023 upstream 精确能量向量来自全参数旧策略，与本分支显式 `elora_paper` 白名单不符；改为同时断言模型内 readiness policy metadata 和有限、非退化预测。参数所有权、梯度与确定性由专门 Goal-0 测试继续强制，不用无关旧向量作为 oracle。
- TorchScript 兼容：未配置 adapter 使用零长度非训练 Parameter 保持静态 Tensor 类型；旧 state dict 缺失这些新 key 时由窄加载钩子补齐，strict-load、JIT compile 与新 adapter 配置均须通过。

### commit 前实际结果

- 本地临时环境：Python 3.12.7，torch 2.11.0+cpu，e3nn 0.4.4，ASE 3.22.1，NumPy 1.26.4，SciPy 1.15.3，h5py 3.14.0；源码通过 `PYTHONPATH=ELoRA` 使用当前 checkout。
- 定向 foundation CLI 回归：`.cache/local-readiness-venv/Scripts/python.exe -m pytest ELoRA/tests/test_run_train.py::test_run_train_foundation -q`，`1 passed, 25 warnings`，退出码 0。
- 完整本地套件：`.cache/local-readiness-venv/Scripts/python.exe -m pytest ELoRA/tests -q --basetemp=.cache/pytest-goal0-full-local-final -o cache_dir=.cache/pytest-goal0-full-local-final-cache`，`67 passed, 14 skipped, 976 warnings in 426.07s`，退出码 0；未排除任何测试文件。
- CPU smoke：`PYTHONPATH=ELoRA .cache/local-readiness-venv/Scripts/python.exe ELoRA/scripts/readiness_smoke.py --device cpu --output .cache/goal0-local-cpu-smoke.json`，退出码 0；`checkpoint_restore=true`、loss 有限、输出 shape `[8,16]`。
- Python 静态编译：`python -m compileall -q ELoRA/mace ELoRA/scripts ELoRA/tests`，退出码 0。
- 四个实际 SBATCH 文件的 `bash -n`：通过，退出码 0；静态核对 unit 命令为完整 `python -m pytest ELoRA/tests -q`。
- `git diff --check`：通过，退出码 0（仅 Git 的 LF/CRLF 工作区提示）。
- 两次 foundation 定向复测曾因调用点补丁误落在同文件的相邻测试而失败；最终逐一核对四处 helper 调用，只有 foundation 显式关闭孤立原子零能量断言，完整套件随后通过。

## 2026-09-01：服务器 pull 重试记录提交（计划）

- 仅修改 `docs/agents/gpu.md` 和追加本检查记录；检查 `git diff --check`、readiness 仍为两个 `not_ready`/零个 `ready`、服务器连接记录包含首次超时和下一次 pull 前置条件。

### commit 前实际结果

- `git diff --check`、readiness marker 计数和连接记录字段断言全部通过，退出码 0；本提交不修改业务代码或测试脚本。

## 2026-09-01：服务器简化重连记录提交（计划与结果）

- 仅追加连接解析失败与下一次简化 pull 计划；`git diff --check` 和 readiness marker 计数在提交前复核，预期/实际均通过，退出码 0；不修改业务代码或测试脚本。

## 2026-09-01：Slurm 提交连接记录（计划与结果）

- 仅追加 pull 成功证据与下一连接的作业提交计划；提交前 `git diff --check` 和 readiness marker 计数均通过，退出码 0；业务代码与 Slurm 脚本未改变。

## 2026-09-01：完整 unit Slurm 时限修复（计划）

- Job 208 在完整 `ELoRA/tests -q` 运行 28:33 后收到 SIGTERM，`FAILED`, `ExitCode=0:15`；stdout 只有启动命令，stderr 明确为 Slurm cancellation/terminated，无 pytest 断言失败摘要。
- 修复单元：仅把 `unit_readiness.sbatch` 的 wall time 从 30 分钟提高到 60 分钟，保留完整测试命令、CPU/memory、exact-commit 门控和失败 trap 不变；不删减或跳过测试。
- commit 前检查：四个 SBATCH `bash -n`、静态断言 unit 的 `--time=01:00:00` 且命令精确为 `python -m pytest ELoRA/tests -q`、`git diff --check`、readiness marker 仍为 not_ready。
- Guqq 集成：在新 exact commit 上重投完整 unit；必须出现 pytest 完整通过汇总、`status=success` 和 scheduler `COMPLETED`, `ExitCode=0:0`。

### commit 前实际结果

- 四个 SBATCH `bash -n`、60 分钟时限唯一匹配、完整 pytest 命令唯一匹配、`git diff --check` 与 not_ready marker 断言全部通过，退出码 0；业务代码和测试内容未改变。
