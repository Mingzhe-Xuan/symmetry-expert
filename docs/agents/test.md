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
