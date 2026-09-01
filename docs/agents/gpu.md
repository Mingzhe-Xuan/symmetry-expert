# 服务器连接记录

## 2026-09-01 15:56 +08:00：检查手动 Python 环境与 Job 202

- 服务器：`Guqq`。
- 用途：按最新 `docs/AGENTS.md` 检查 setup Job 202 和用户手动构建的 Python `.venv`，确认能否直接进入 Slurm readiness 验证。
- 计划操作：现有持续 SSH 会话若仍有效，先再次执行 `git pull --ff-only`；随后只读检查 job 状态、`pyvenv.cfg`、Python/pip 版本、固定依赖 import/version、`pip check` 与 torch CUDA build/arch。若需新连接，同样先 pull。
- 权限核对：作业状态和虚拟环境检查属于最新规范明确允许的登录节点轻量操作；不在登录节点运行 pytest、模型 forward/backward、CUDA kernel、编译或明显计算任务。
- 状态：Job 202 已在规则切换期间取消；等待检查用户手动环境。

- 连接尝试 1：SSH 已到达 Guqq，但硬前置 `git pull --ff-only` 因 `GnuTLS recv error (-110)` 失败，后续环境命令由 `&&` 阻止，未执行。
- 连接尝试 2 计划：再次连接，先用 HTTP/1.1 执行 `git pull --ff-only`；仅在成功后运行同一组轻量环境检查。
- 连接尝试 2 结果：HTTP/1.1 pull 等待约 120 秒仍无输出，已中断；后续检查未执行。
- 连接尝试 3 计划：仍先执行 `git pull --ff-only`。若 TLS 再次失败，仅当服务器 `HEAD` 与已获取的 `origin/main` tree 完全一致且均显示当前 commit 时，才继续只读环境检查；若不一致则立即停止。
- 连接尝试 3 结果：pull 再次超时；`HEAD` 与 `origin/main` 均为 `3249b76ae5f5dfe544a1508227c876ebfa173aac`，tree diff 为零。远端仅有任务/用户创建的未跟踪 `.venv/`、`requirements.txt` 和既有 `net.sh`。Job 202 已从 controller 清理，`scontrol` 返回 invalid job ID 并使后续链停止，因此环境尚未读取。
- 连接尝试 4 计划：先做短时 pull；若仍失败，仅在上述 refs/tree 一致性再次通过后继续。Job 202 查询改为可选，然后读取手动 `.venv` 和根目录 `requirements.txt`，不读取或修改 `net.sh`。
- 连接尝试 4 结果：`git pull --ff-only` 成功并返回 `Already up to date`。手动 `.venv` 为 Python 3.10.12、pip 26.2.1，19 个直接依赖版本匹配且 `pip check` 通过；但 torch 是 `2.11.0+cu130` / CUDA 13.0，驱动 570.211.01 过旧，`cuda_available=False`。根目录未跟踪 `requirements.txt` 仅配置清华 PyPI mirror，是 cu130 选择根因。
- 下一连接计划：在 canonical requirements 修复、commit 前检查、commit/push 完成后，先 pull 新 commit，再提交版本化 setup Slurm job 重建 `.venv`；不读取或修改 `net.sh`。
