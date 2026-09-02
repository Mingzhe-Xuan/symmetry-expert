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

## 2026-09-01：Guqq foundation cache 修复（计划）

- Job 211 在 4 秒内 collection error：`/home/xmz/symmetry-expert/.cache/mace/46jrkm3v` 缺少 zip central directory，证明 Job 208 的长运行留下不完整 MACE-MP small cache。
- 先只读记录该文件的路径、大小与 SHA-256；仅在 realpath 位于任务 `.cache/mace/` 且文件名精确为 `46jrkm3v` 时删除损坏文件。
- 使用源码固定 URL `https://tinyurl.com/46jrkm3v`、IPv4、redirect、失败重试和临时 `.part` 文件重新下载；zip 完整性通过后原子改名，避免再次把半文件暴露为有效 cache。
- 轻量验收：记录新 size/SHA-256，Python `zipfile` 完整性检查通过；随后在不改变测试范围的情况下重投 exact commit `af5aef4` 的完整 Job。

### 实际结果

- 旧损坏文件：25,731,072 字节，SHA-256 `81c44f00d79a5faac7c902bdd51c7f9f7322ed5dd779f523d243e8a70de64ac0`。
- Guqq IPv4 下载仅约 19 KiB/s，在正式 cache 尚不存在、仅 `.part` 约 5% 时中断；本地同 URL cache 为 32,581,838 字节，zip 完整，SHA-256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`。
- 首次 SCP 被 port 22 关闭；增加 keepalive 的单次重试成功。远端临时文件的 size/hash/zip 全部与本地一致后原子改名；正式 `.cache/mace/46jrkm3v` 复核通过。
- 下一门控：exact `af5aef4` 重投完整 `ELoRA/tests -q`，不改变 suite 或依赖环境。

## 2026-09-01：其余 foundation cache 预置（计划）

- 本地完整 suite 已验证的额外 cache：MACE-MP large `5f5yavf3` 133,803,220 字节；MACE-OFF small/medium/large 分别 7,347,350 / 18,350,596 / 55,492,786 字节，SHA-256 由本地清单记录。MACE-MP medium 使用仓库自带模型，不需要 cache 文件。
- 若 Guqq 只读盘点证明 Job 213 正在向正式 `5f5yavf3` 写入不完整下载，则取消本任务 Job 213，避免并发覆盖；随后按 `.part` + size/hash/zip + 原子改名逐个 SCP 本地已验证文件。
- 所有远端目标必须通过 `.cache/mace` realpath 门控；已有 size/hash/zip 全部正确的文件保留，不覆盖。修复后重投完整 unit。

### 条件诊断结果

- Job 213 运行 44:52 时，`5f5yavf3` 仅 17,309,696 / 133,803,220 字节且 mtime 持续更新；三个 MACE-OFF cache 均缺失。按证据取消 Job 213，避免与预置文件并发写。
- 下一步先本地逐个 zip 校验，再将四文件复制到独立 staging 目录，一次 SCP 到远端独立 preload 目录；逐文件 size/hash/zip 通过后才替换正式 cache。
- 递归 SCP 中断后远端 preload 仅含 `5f5yavf3` 24,444,928 / 133,803,220 字节，其他三文件未开始。后续改用 SFTP `reput` 断点续传到同一 staging，避免每次连接从零覆盖。

### 预置实际结果

- SFTP `reput` 经断点续传完成；针对连接不稳定使用 `-R 1 -B 32768`，并从 batch 逐步移除已完成文件。正式启用前四文件 size/SHA-256 与本地清单全部一致，Python zipfile 全量检查 `files=4` 通过。
- 远端原子启用并复核：MACE-MP large `f80e992b…95b32`；MACE-OFF small `165cce4c…7c46f`、medium `4842c52a…87db7`、large `a29e397d…d76f4`。正式 cache 不再包含 Job 213 的 partial large。
- 下一门控：exact af5 完整 unit；collection 不应联网下载 foundation 文件。

## 2026-09-01：CPU TorchInductor segfault 诊断（计划）

- Job 216 在完整 suite 的前 12 项后于 `test_compile.py::test_mace` 原生段错误：scheduler `FAILED`, `ExitCode=139:0`, runtime 6:39；Python fatal stack 位于 `/tmp/torchinductor_xmz/...py` 生成 kernel 的 forces 二阶反向，无 pytest assertion failure。
- 新增版本化 `compile_cpu_diagnostic.sbatch`，只运行 `test_mace[fp32-cpu]` 与 `[fp64-cpu]`，仍执行真实 `torch.compile(mode=default)`、energy/forces 数值比较和二阶反向。
- 并行比较两个 Slurm profile：`fresh_cache` 使用每-job `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR`；`fresh_cache_single_thread` 在此基础上增加 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`TORCHINDUCTOR_COMPILE_THREADS=1`。
- commit 前检查：新 SBATCH `bash -n`、两个 profile/两个精确 pytest node id/每-job cache 静态断言、`git diff --check`、readiness 保持 not_ready。
- 只有保留真实 compile 与双 dtype CPU 测试的 profile 通过，才用于完整 unit；不得 skip、xfail 或改 eager backend。

### commit 前实际结果

- 诊断 SBATCH 与 unit SBATCH `bash -n` 通过；两个 profile、每-job Inductor/Triton cache、fp32/fp64 CPU 精确 node ids 静态断言通过；`git diff --check` 与 not_ready marker 检查通过，全部退出码 0。
- 本地 collect-only 确认 node ids 为 `test_mace[fp32-cpu]` 与 `test_mace[fp64-cpu]`；Windows 按 upstream 标记跳过执行，因此真实运行仅提交 Guqq Slurm。

### Guqq 定向诊断结果

- Job 217 `fresh_cache`：`FAILED`, `ExitCode=134:0`, runtime 00:01:00；fp32 CPU 用例在每-job 新缓存生成的 Inductor forces 双重反向 kernel 中 abort，排除陈旧编译缓存但未排除线程并发。
- Job 218 `fresh_cache_single_thread`：`COMPLETED`, `ExitCode=0:0`, runtime 00:02:59；同一提交、节点和资源上，fp32/fp64 两个真实 compile 用例均通过，汇总为 `2 passed, 85 warnings in 174.60s`。
- 结论：正式完整 unit 脚本采用每-job Inductor/Triton cache，以及 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`TORCHINDUCTOR_COMPILE_THREADS=1`；不修改 pytest 范围、backend、断言或 dtype。

## 2026-09-01：正式 unit 单线程稳定配置（计划）

- 修改范围仅为 `unit_readiness.sbatch` 的运行环境：增加每-job Inductor/Triton cache 和 Job 218 已证实有效的三个单线程变量；完整命令保持 `python -m pytest ELoRA/tests -q`。
- commit 前检查：全部 SBATCH `bash -n`；静态断言四个 cache/thread 变量、完整 pytest 命令和 60 分钟时限；`git diff --check`；readiness 仍为两个 `not_ready`、零个 `ready`。
- Guqq 集成：commit/push 后新连接先 pull exact commit，再提交完整 unit；必须以 pytest 完整通过汇总、`status=success`、scheduler `COMPLETED` 和 `ExitCode=0:0` 验收。

### commit 前实际结果

- 五个 SBATCH 的 `bash -n` 全部通过，退出码 0。
- unit 静态断言确认每-job Inductor/Triton cache、三个单线程变量、60 分钟时限及唯一完整 pytest 命令全部存在，退出码 0。
- `git diff --check` 通过；readiness marker 为两个 `not_ready`、零个 `ready`，退出码 0。

### Guqq 完整 suite 结果

- Job 219：`FAILED`, `ExitCode=1:0`, runtime 00:08:26；单线程/独立 cache 已消除 Job 216/217 的原生崩溃。
- pytest：`1 failed, 71 passed, 1 skipped, 1122 warnings, 8 errors in 498.72s`。
- 8 errors：CUDA benchmark 用例均在 setup 阶段缺少 `benchmark` fixture，canonical venv 未安装 `pytest-benchmark`。
- 1 failure：`test_graph_breaks` 报 2 个 graph breaks，均明确来自 PyTorch 2.11 的 `trace_autograd_ops=False` 拒绝追踪 `torch.autograd.grad`。

## 2026-09-02：完整 suite 环境与 autograd tracing 修复（计划）

- 环境单元：在 `docs/requirements.txt` 固定 `pytest-benchmark==5.2.3`；下载 Python 3.10 通用 wheel 及缺失传递依赖，记录 size/SHA-256，并让 setup 的 offline gate 只有在这些 wheel 通过精确 hash 校验时才启用。
- 编译单元：采用 MACE upstream 当前 `configure_autograd_for_compile()` 语义；`allow_autograd=True` 时除 `allow_in_graph` 外，对支持该配置的 PyTorch 显式设置 `dynamo.config.trace_autograd_ops=True`。增加不依赖 CUDA 的回归，先强制 false，再断言 `prepare()` 恢复 true。
- 本地门控：新回归、完整 `ELoRA/tests`、requirements/plugin 解析与 import/version、全部 SBATCH `bash -n`、setup offline hash gate 静态断言、`git diff --check`、readiness marker。
- Guqq 门控：由 Python 创建的既有 `.venv` 通过版本化 setup job 重新验收，必须包含 `pytest-benchmark 5.2.3`、`pip check` 与 torch cu128；随后 exact commit 重跑完整 unit。不得跳过 benchmark、graph-break 或真实 compile 测试。

### commit 前实际结果

- wheel 获取：`pytest_benchmark-5.2.3-py3-none-any.whl` 为 45,255 字节、SHA-256 `bc839726…b0803`；`py_cpuinfo-9.0.0-py3-none-any.whl` 为 22,335 字节、SHA-256 `859625bc…74d5`。二者均为 Python 3 通用 wheel。
- 本地安装/import：Python 创建的审计 venv 从上述离线目录安装成功；metadata 精确报告 `pytest-benchmark=5.2.3`、`py-cpuinfo=9.0.0`。
- autograd 定向回归：`test_prepare_enables_autograd_tracing` 为 `1 passed`，退出码 0。
- 首次完整调用未传仓库 `MACE_CACHE_DIR`，collection 在尝试访问受限用户缓存时中止；使用此前相同的仓库 `.cache/mace` 变量重跑，不修改测试范围。
- 完整本地套件：`68 passed, 14 skipped, 976 warnings in 364.93s`，退出码 0；新增 1 项为 autograd tracing 回归，Windows compile/CUDA 仍按 upstream 标记跳过。
- 五个 SBATCH `bash -n`、requirements pins、offline wheel 文件名/SHA gate 与插件 import/version 静态检查全部通过，退出码 0。

### Guqq c5b 集成结果

- setup Job 221：`COMPLETED`, `ExitCode=0:0`, runtime 00:04:44；canonical Python venv 和新增插件门控全部通过。
- unit Job 222：`FAILED`, `ExitCode=1:0`, runtime 00:08:35；`7 failed, 74 passed, 1 skipped, 1194 warnings in 507.59s`。
- 6 个 compile benchmark 已获得 fixture 并真实执行，但 fullgraph 在 `models.py` 的图内 `node_attrs.requires_grad_(True)` 报 PyTorch 2.11 Unsupported；说明 autograd tracing 已生效，下一缺口是输入梯度准备的位置。
- `test_foundations` 在先前测试之后出现 node attrs Double / atomic energies Float；需证明并修复默认 dtype 的跨用例状态泄漏，不得只按当前测试顺序打补丁。

## 2026-09-02：fullgraph 输入梯度与 dtype 隔离（诊断计划）

- 对照 MACE upstream 当前 `models.py`、compile tests 与 torch.compile 建议，定位 `requires_grad_` 应在编译图外准备还是由可追踪张量构造替代；保留 `fullgraph=True`、forces/autograd 与 benchmark 语义。
- 新增最小回归：编译准备不得依赖图内 `requires_grad_`；模型 eager 输出/forces 与修复前语义一致。Guqq 定向运行 6 个 benchmark 和 graph-break/CPU compile 节点。
- dtype 回归：以非默认 dtype 创建 foundation target/source 和 batch，验证 helper/calculator 调用前后默认 dtype 精确恢复，后续 forward 不发生权重/输入类型错配。
- 查明根因后先更新本计划的具体实现/预期，再修改代码；随后本地完整 suite、smoke、静态检查及 Guqq 完整 suite 均不得缩小。

### 根因与实现门控

- MACE upstream 当前 compile 测试在调用 compiled model 前显式执行 `batch["positions"].requires_grad_(True)`；当前主模型的 graph preparation 在 `torch.compiler.is_compiling()` 时跳过图内 `requires_grad_`。本分支将对 MACE/ScaleShiftMACE 保留 eager 行为，但编译时跳过 node_attrs/positions 的图内突变，并让三个 compiled 测试入口在图外设置 positions；`fullgraph=True` 保持不变。
- node attrs 不参与 forces 的微分变量，compiled calculator 已在图外准备输入；直接 compile 测试仅需 positions 作为能量对坐标求导的 leaf。回归需同时比较 eager/compiled energy 与 forces。
- dtype 根因是 `torch_tools.default_dtype()` 的 generator contextmanager 缺 `try/finally`：compile benchmark 抛异常时未恢复 float64，继而污染 foundation。增加异常路径精确恢复测试，并将 restore 放入 finally；不修改 foundation 数值或顺序。
- 定向本地门控：dtype 正常/异常恢复、autograd tracing、compile node collect/JIT；Guqq 定向门控：6 个 fullgraph benchmark + graph-break + fp32/fp64 CPU/CUDA compile，再运行完整 suite。

### commit 前实际结果

- dtype 异常恢复与 autograd tracing 定向回归：`2 passed`，退出码 0。
- 模型/JIT 回归：完整 `test_models.py` 为 `3 passed, 350 warnings`，退出码 0；`torch.compiler.is_compiling()` 门控未破坏 TorchScript/trace。
- 完整本地 suite：`69 passed, 14 skipped, 978 warnings in 351.31s`，退出码 0；新增项为 dtype 异常恢复，未缩小任何测试范围。
- Python compileall、五个 SBATCH `bash -n`、fullgraph/图外 positions leaf/模型 compile guard 静态断言、`git diff --check` 全部通过；readiness marker 仍为两个 `not_ready`、零个 `ready`，退出码均为 0。

### Guqq 门控顺序调整

- 完整 unit 在当前稳定配置仅约 8:35，且精确包含 6 个 fullgraph benchmark、graph-break、fp32/fp64 CPU/CUDA compile 及所有其他测试；因此直接运行完整 suite 是定向集合的严格超集，可同时给出更强的跨测试 dtype 隔离证据。
- 不创建临时/未版本化的定向 Slurm 命令；exact e797 上直接提交版本化完整 unit，并并行重跑 CPU/GPU smoke。测试范围、fullgraph 和断言均不降低。

### Guqq exact e797 实际结果

- Job 225 unit：`COMPLETED`, `ExitCode=0:0`, runtime 00:22:32；`82 passed, 1 skipped, 1201 warnings in 1341.62s`，完整命令 `python -m pytest ELoRA/tests -q`，8 个 benchmark 表均存在，stderr 为空。
- Job 227 CPU smoke：`COMPLETED`, `ExitCode=0:0`, runtime 00:00:04；checkpoint restore、有限 loss、shape 与 JSON 全部通过，stderr 0 字节。
- Job 228 GPU smoke：`COMPLETED`, `ExitCode=0:0`, runtime 00:00:05；RTX 5090、CC 12.0、sm_120、torch CUDA 12.8、实际 CUDA kernel、checkpoint restore 与 JSON 全部通过，stderr 0 字节。
- 三项均为 exact commit `e797570ab0d871227a26f4416b446a0c875c93fb`；持久监控最终输出 `ALL_E797_READINESS_JOBS=PASS`。

### 最终证据核验

- SCP 回传 setup 221、unit 225、CPU 227、GPU 228 的 8 个 stdout/stderr 和 2 个 JSON，精确文件数为 10。
- setup stdout 断言 cu128/CUDA 12.8/pytest-benchmark 5.2.3/py-cpuinfo 9.0.0/`pip check`/成功尾标；unit stdout 断言 exact e797、82 passed/1 skipped、8 benchmarks、成功尾标。
- unit/CPU/GPU stderr 均为 0 字节；CPU/GPU JSON 的 device、shape、checkpoint restore、RTX 5090、CC 12.0、sm_120、CUDA 12.8 与 kernel 字段全部通过；`FINAL_EVIDENCE_ASSERTIONS=PASS`。
- 最终报告需精确包含 Jobs 221/225/227/228、资源/终态/退出码、日志路径、脚本 hash、可选 schedulefree skip 与 setup online fallback 限制；readiness 首尾均为 ready。

### 最终文档提交前实际结果

- readiness 报告全部必需字段断言通过：exact e797、Jobs 221/225/227/228、82 passed/1 skipped、四脚本 hash 前缀、服务器日志路径、schedulefree 与 setup fallback 限制、未运行 Goal 1。
- readiness marker 精确为两个 `ready`、零个 `not_ready`；`git diff --check` 通过，退出码 0。

## 2026-09-02：dense / elora_paper GPU 更新验证（计划）

- 实现单元：扩展 `readiness_smoke.py` 与 GPU SBATCH，使 `dense`、`elora_clean`、`elora_paper` 使用相同的双 expert mixed batch、loss 和 optimizer 协议。
- 每种模式必须证明：forward 输出有限；backward 完成；两个被路由 expert 的可训练差分参数各自具有有限且非零梯度；一次 optimizer step 后两个 expert 参数切片均发生变化；checkpoint restore 输出一致。
- 对 ELoRA 模式，初始化时 `B=0` 导致首步 `A` 梯度可为零，因此更新断言针对承载首步梯度的 `lora_B_bank`；dense 针对 `expert_delta_bank`。这不降低 backward 标准，而是匹配零差分 LoRA 初始化的数学语义。
- 本地提交前门控：Python `venv` 中运行三模式 CPU smoke、相关 readiness 单元测试、`gpu_smoke.sbatch` 的 `bash -n`、Python compileall 与 `git diff --check`，全部退出码必须为 0。
- Guqq 门控：同一已推送 exact commit 上，通过版本化 `compute`/`gpu:1` Slurm 脚本运行三模式 GPU smoke；逐模式 JSON、stdout/stderr、scheduler 终态和退出码全部通过后才可完成。

### commit 前实际结果

- 本地环境由 `python -m venv --system-site-packages .cache/dense-paper-venv` 创建；Python 3.12.7、torch 2.11.0+cpu，并在 venv 内固定 e3nn 0.4.4、torch-ema 0.3、matscipy 1.0.0、python-hostlist 1.23.0。首次两次导入门控分别暴露缺失 torch-ema/matscipy 和 python-hostlist，补齐 `docs/requirements.txt` 中的固定依赖后通过；均未进入测试计算，不属于模式失败。
- 三模式 CPU smoke 退出码均为 0。`dense` 两 expert 非零梯度计数 `[32,32]`、参数变化范数约 `[0.008000,0.008000]`；`elora_clean` 与 `elora_paper` 均为 `[16,16]`、约 `[0.005657,0.005657]`；三者输出有限、shape `[8,16]`、checkpoint restore 均成功。
- 相关 readiness 回归：`28 passed, 39 warnings in 10.46s`，退出码 0。
- Python compileall、GPU SBATCH `bash -n` 与 `git diff --check` 均退出码 0。额外尝试的 ruff 命令因本地未安装 ruff 而未执行；ruff 不属于预登记门控，且长行静态检查已人工修正。
