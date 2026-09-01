# Goal 0 当前状态与计划

## 当前状态

- Goal 0 本地实现已完成，readiness 仍为 `not_ready`。
- 本地 canonical 依赖为 `docs/requirements.txt`；Guqq 固定目标为 Python 3.10.12、PyTorch 2.11.0+cu128、e3nn 0.4.4。
- setup Job 201 因不可达 IPv6 长超时取消；Job 202 在规则切换期间取消。手动 Python venv 的固定包版本与 `pip check` 通过，但错误安装 torch 2.11.0+cu130，当前驱动下 CUDA 不可用。
- 用户拥有的 `ELoRA/README.md` 与最新 `docs/AGENTS.md` 本地修改保持未提交，除非用户另有指示。
- Guqq→GitHub pull 连续 3 次 TLS 超时；远端 `HEAD`/`origin/main` 仍一致于最新已推送 `3249b76…`，允许继续只读环境核验，不提交新作业。
- setup Job 203 已装好 cu128 依赖，但 editable ELoRA 的离线 build isolation 找不到 setuptools，环境尚未完成。

## 当前计划

1. 修复 canonical requirements：清华主索引 + 官方 PyTorch cu128 extra index。
2. commit 前静态检查、目标解析、SBATCH syntax 与 40 项 readiness 回归已通过并记录。
3. editable `--no-build-isolation` 修复及 commit 前验证已完成；下一步 commit/push、Guqq pull 并重投 setup，通过后进入 Slurm unit/CPU/GPU readiness。
4. 回传并记录作业证据；全部门控通过前不进入 Goal 1。

## 变更记录

- 2026-09-01 15:56 +08:00：根据扩权后的 `docs/AGENTS.md` 建立新的代理记录目录；环境创建策略由 Python `venv` 调整为固定版本 uv。
- 2026-09-01 16:00 +08:00：用户再次明确使用 Python 创建 venv，并已手动尝试构建；撤销未实施的 uv 计划，先验证现有环境。
- 2026-09-01 16:08 +08:00：Guqq GitHub pull 连续 3 次失败；refs/tree 一致性已证明，计划缩短后续 pull timeout，仅继续只读手动环境核验。
- 2026-09-01 16:15 +08:00：第 4 次 pull 成功。手动 venv 被确认误装 cu130；调整计划为双索引修复并由 setup Slurm job 重建。
- 2026-09-01 16:20 +08:00：双索引修复的全部 commit 前检查通过；下一步提交、推送和重投 setup。
- 2026-09-01 16:57 +08:00：Job 203 安装 cu128 成功，但 editable build isolation 离线失败；新增窄修复单元并保持其余门控不变。
- 2026-09-01 17:02 +08:00：editable 修复的四项 commit 前检查全部通过，进入提交与 setup 重投阶段。
