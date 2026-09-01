# 进度更新

## 2026-09-01：采用扩展服务器权限和新记录路径

- 最新规范允许在登录节点执行轻量环境管理、依赖下载、安装和诊断，但明显计算负载仍通过 Slurm。
- 记录路径迁移到 `docs/agents/`；旧 `docs/*.md` 保留为历史记录，不删除或覆盖。
- 用户最新明确要求继续使用 Python `venv`，并已手动尝试构建；下一单元先验证现有环境，不预设需要重建。
- Guqq 环境核验前 GitHub pull 连续 3 次 TLS 超时；已按规则记录经验。远端 refs/tree 与最新已推送 commit 完全一致，后续仅进行只读 venv 核验，不提交新作业。
- 第 4 次 pull 成功。手动 venv 的固定依赖与 `pip check` 通过，但 torch 为 cu130，驱动不兼容且 CUDA 不可用；canonical requirements 改为清华主索引 + 官方 cu128 extra index，等待 commit 前验证。
- cu128 索引修复的 commit 前验证完成：双索引静态断言、目标 wheelhouse 解析、SBATCH syntax 均通过，readiness 回归 40 passed。
- setup Job 203 已证明 cu128 双索引修复有效；最后因 editable install 的离线 build isolation 找不到 setuptools 失败。计划增加 `--no-build-isolation` 后按新规范 commit 前验证。
- editable 修复的 commit 前验证全部通过：SBATCH syntax、精确静态断言、真实 editable dry-run 和 40 项 readiness 回归。
- setup Job 204 成功：Python venv、torch cu128/CUDA 12.8、editable ELoRA、`pip check` 和固定版本 import 全部通过。
- Jobs 205/206/207 全部 `COMPLETED`, `ExitCode=0:0`：unit 40 passed、CPU smoke 成功、RTX 5090 GPU smoke 证明 CC 12.0/sm_120/CUDA kernel。证据已 SCP 核验，Goal 0 readiness 更新为 `ready`。
- 最终文档检查通过后清理本任务本地临时文件约 1.64 GiB；保留来源不明的 `.cache/uv`，服务器原始作业证据未删除。
