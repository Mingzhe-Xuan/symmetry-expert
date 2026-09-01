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
