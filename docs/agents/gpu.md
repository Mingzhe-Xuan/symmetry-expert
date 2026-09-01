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

## 2026-09-01 16:25 +08:00：提交 cu128 setup

- 服务器：`Guqq`。
- 用途：pull 已推送且 commit 前验证通过的 `9369c711eb3c9445edb8829f8187b5586dd3dadc`，提交 `setup_readiness.sbatch` 重建 Python `.venv`。
- 计划操作：先 `git pull --ff-only`；核对 HEAD 后执行 `sbatch --export=ALL,EXPECTED_COMMIT=9369c711eb3c9445edb8829f8187b5586dd3dadc ...`；随后只读监控 scheduler 和日志。
- 权限核对：Git pull、Slurm 提交和状态/日志查看符合规范；下载、安装和环境文件变更由版本化 setup job 在任务约定 `.venv/.cache` 中执行，不修改受 Git 管理源码。
- 状态：连接成功，Guqq fast-forward 到 `9369c711eb3c9445edb8829f8187b5586dd3dadc`；提交 setup Job 203。下一连接用于持续只读监控，仍先 pull。

- Job 203 结果：online resolver 明确选择并安装 `torch-2.11.0+cu128` 及 CUDA 12.8 依赖；`python-hostlist` wheel 构建成功。随后 editable ELoRA 的隔离 build env 因 `--no-index` 找不到 `setuptools>=42` 而退出 1，尚未运行 `pip check`/import 门控。
- 同一持续连接的下一操作：本地修复已以 `dee1e009b352d209a476af83623e14f71a492300` push。先在 Guqq 执行 `git pull --ff-only`，核对 exact HEAD，再提交替代 setup job；继续只读监控。
- 持续连接结果：pull 期间 SSH `Connection reset`，没有 pull 完成或 sbatch 输出，故不假定替代 job 已提交。

## 2026-09-01 17:06 +08:00：重连提交 editable 修复

- 用途：新连接先 pull `dee1e00…` 并核对 HEAD，再提交替代 setup job，返回明确 Job ID。
- 权限核对：Git pull、Slurm 提交与环境 cache/venv 变更均在最新规范授权范围；不触碰 `net.sh` 或其他任务数据。
- 状态：连接成功，Guqq fast-forward 到 `dee1e009b352d209a476af83623e14f71a492300`；提交 setup Job 204。下一连接用于持续只读监控，仍先 pull。
- Job 204 结果：成功。editable MACE/ELoRA 构建安装、`pip check`、固定依赖 import 全部通过；报告 torch `2.11.0+cu128`、CUDA 12.8、e3nn 0.4.4，结束时间 `2026-09-01T17:12:35+08:00`。
- 同一连接下一操作：提交 exact commit `dee1e00…` 的 unit readiness、CPU smoke 和 GPU smoke 作业，并持续只读监控。
- Jobs 205/206/207：scheduler 均为 `COMPLETED`, `ExitCode=0:0`。Job 205 为 40 passed；Job 206 CPU smoke 成功；Job 207 报告 RTX 5090、CC 12.0、sm_120、CUDA 12.8 和实际 kernel 成功。持续会话已正常退出。

## 2026-09-01 17:16 +08:00：SCP 回传 readiness 证据

- 用途：仅从 Guqq 回传 Jobs 204–207 的 stdout/stderr 与 CPU/GPU smoke JSON 到本地忽略的 `.cache/guqq-readiness-dee1e00/`。
- 权限核对：SCP 结果回传在本地许可范围；目标是 `.cache`，不会提交生成日志/结果，也不修改服务器文件。
- 状态：SCP 成功；10 个文件回传到本地 `.cache/guqq-readiness-dee1e00/`。stdout/JSON 内容与 scheduler 结果一致，unit/CPU/GPU stderr 均为 0 字节；最终文档检查完成后按授权删除本地副本，服务器原始日志仍可回传。

## 2026-09-01 20:46 +08:00：完成审计修复的完整 Slurm 复验计划

- 服务器：`Guqq`。
- 用途：在本地完整套件与 smoke 已通过后，验证即将推送的审计修复 exact commit；重跑完整 `ELoRA/tests`、CPU smoke 与 RTX 5090 GPU smoke，替代旧 Job 205 的选定文件证据。
- 计划操作：新连接后第一条仓库操作为 `git pull --ff-only`，核对 origin、干净受管工作树和 exact HEAD；只在一致时用版本化 SBATCH 提交 unit/CPU/GPU 作业，立即记录 job IDs，随后只读监控 `squeue`/`scontrol` 与日志。完成后用 SCP 回传 stdout/stderr 和 smoke JSON。
- 权限核对：git pull、状态检查、Slurm 提交/监控和 SCP 回传均在最新规范与用户自动批准的 git 操作范围；所有 pytest、模型与 CUDA 计算均在 Slurm compute job 内执行，不在登录节点直接运行，不修改服务器受 Git 管理源码。
- 连接尝试 1 结果：SSH 到达 Guqq，但前置 `git pull --ff-only` 在 135 秒后因 GitHub 443 连接超时失败；链式命令阻止了 HEAD 核对与所有 Slurm 操作，未提交作业。

## 2026-09-01 20:52 +08:00：短超时重试 pull

- 用途：新连接仍首先执行 `git pull --ff-only`，但用 45 秒外层 timeout 避免 GitHub 停滞；只有 pull 成功并核对 exact HEAD 后才提交完整 Slurm 验证。
- 权限核对：与上一连接相同；短超时只限制网络等待，不绕过 pull 前置条件，不使用旧 refs 代替新 commit。
- 连接尝试 2 结果：远端 shell 因本地/远端嵌套引号不匹配而在解析阶段退出，未执行 `git pull`、状态检查或 Slurm 操作。

## 2026-09-01 20:55 +08:00：简化命令重连

- 用途：去除命令替换与嵌套引号，以简单链式命令在新连接中先限时 `git pull --ff-only`，再输出 HEAD/origin/status；成功后另行提交 exact HEAD 作业。
- 权限核对：与前两次相同；此次只执行 pull 和只读 Git 核对，不在同一命令内提交作业。
- 连接尝试 3 结果：`git pull --ff-only` 成功，Guqq fast-forward 到 `33f7b926227f434a3a6e1ad64742cf7e9b1996a7`；origin 正确，受管 working/index diff 均为空。

## 2026-09-01 20:58 +08:00：提交完整 readiness 作业

- 用途：新连接先 pull 含本记录的最新 commit 并核对 HEAD；随后在 `compute` 分区提交完整 unit、CPU smoke 和 `gpu:1` GPU smoke，立即返回三个 job ID。
- 权限核对：版本化 SBATCH 的 pytest、模型与 CUDA 负载均由 Slurm 执行；登录节点仅做 pull、精确状态核对和 `sbatch` 提交，符合规范。
