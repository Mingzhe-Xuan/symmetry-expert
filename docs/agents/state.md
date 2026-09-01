# Goal 0 当前状态与计划

## 当前状态

- Goal 0 完成审计修复已通过本地完整门控，readiness 仍为 `not_ready`：等待同一已推送提交在 Guqq 通过完整 Slurm unit、CPU smoke 和 RTX 5090 GPU smoke，并回填最终证据。
- 本地 canonical 依赖为 `docs/requirements.txt`；Guqq 固定目标为 Python 3.10.12、PyTorch 2.11.0+cu128、e3nn 0.4.4。
- setup Job 201 因不可达 IPv6 长超时取消；Job 202 在规则切换期间取消。手动 Python venv 的固定包版本与 `pip check` 通过，但错误安装 torch 2.11.0+cu130，当前驱动下 CUDA 不可用。
- 用户拥有的 `ELoRA/README.md` 本地修改保持未提交，除非用户另有指示；最新版 `docs/AGENTS.md` 已随实现提交。
- Guqq 已成功 fast-forward 到最终验证 commit `dee1e009b352d209a476af83623e14f71a492300`。
- setup Job 204 已完成 Python venv、cu128 固定依赖、editable ELoRA、`pip check` 与 import 门控；环境可用于 Slurm readiness。
- Jobs 205/206/207 的既有选定 unit、CPU smoke、RTX 5090 GPU smoke 均成功，但不能替代修复后同一提交的完整 Slurm 回归。

## 当前计划

1. 提交并推送已通过本地门控的审计修复，保持用户 `ELoRA/README.md` 改动不进入提交。
2. 在 Guqq 新连接先 pull 并核对 exact HEAD，再通过 Slurm 重跑完整 unit、CPU/GPU smoke。
3. 回传并核验 job 日志、JSON、脚本哈希、资源、环境、终态和退出码；回填 readiness 报告。
4. 运行最终文档检查、提交推送、清理任务临时文件并确认工作树只剩用户改动；不启动 Goal 1。

## 变更记录

- 2026-09-01 15:56 +08:00：根据扩权后的 `docs/AGENTS.md` 建立新的代理记录目录；环境创建策略由 Python `venv` 调整为固定版本 uv。
- 2026-09-01 16:00 +08:00：用户再次明确使用 Python 创建 venv，并已手动尝试构建；撤销未实施的 uv 计划，先验证现有环境。
- 2026-09-01 16:08 +08:00：Guqq GitHub pull 连续 3 次失败；refs/tree 一致性已证明，计划缩短后续 pull timeout，仅继续只读手动环境核验。
- 2026-09-01 16:15 +08:00：第 4 次 pull 成功。手动 venv 被确认误装 cu130；调整计划为双索引修复并由 setup Slurm job 重建。
- 2026-09-01 16:20 +08:00：双索引修复的全部 commit 前检查通过；下一步提交、推送和重投 setup。
- 2026-09-01 16:57 +08:00：Job 203 安装 cu128 成功，但 editable build isolation 离线失败；新增窄修复单元并保持其余门控不变。
- 2026-09-01 17:02 +08:00：editable 修复的四项 commit 前检查全部通过，进入提交与 setup 重投阶段。
- 2026-09-01 17:13 +08:00：setup Job 204 全绿，环境门控完成；进入 unit/CPU/GPU Slurm 验证阶段。
- 2026-09-01 17:16 +08:00：Jobs 205/206/207 全绿，证据已 SCP 并核验；Goal 0 readiness 改为 `ready`，进入最终文档与临时文件清理。
- 2026-09-01 17:23 +08:00：最终文档与回传证据一致性检查通过；清理本任务临时文件约 1.64 GiB，仅保留来源不明的 `.cache/uv`。
- 2026-09-01 17:37 +08:00：逐条完成审计推翻先前完成判断；发现服务器未跑完整测试套件及 `evaluate()` 破坏梯度白名单，readiness 恢复为 `not_ready`，进入修复和完整闭环。
- 2026-09-01 17:49 +08:00：完整本地套件暴露 Windows `PYTHONPATH` 分隔符和旧 foundation pickle/state-dict 的 adapter 向后兼容缺口；增加对应测试单元后继续修复，不缩小测试范围。
- 2026-09-01 20:46 +08:00：兼容性与完成审计修复通过完整本地套件（67 passed、14 skipped）、CPU smoke、compile、SBATCH syntax 和 diff 检查；下一步提交推送并在 Guqq 对 exact commit 执行完整 Slurm 闭环。
