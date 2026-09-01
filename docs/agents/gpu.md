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
- 连接结果：pull 到 exact `d4d93997555383fd411a20d08cc728fa7054a5b1`，受管 diff 为空；提交完整 unit Job 208、CPU smoke Job 209、GPU smoke Job 210。

## 2026-09-01 21:01 +08:00：Jobs 208–210 状态监控

- 用途：新连接先 pull（origin 当前仍为作业绑定的 `d4d9399`），只读查看 `squeue`、`scontrol show job` 和已有日志；不修改远端 HEAD，避免尚未启动的作业触发 exact-commit 门控。
- 权限核对：scheduler/日志只读检查属于登录节点轻量操作；不在登录节点运行测试或模型，不取消作业。
- 监控结果：pull/HEAD 仍为 exact `d4d9399`。CPU Job 209 与 GPU Job 210 日志均以 `status=success` 结束且 stderr 为空；GPU 报告 RTX 5090、CC 12.0、sm_120、torch CUDA 12.8 和实际 kernel。Unit Job 208 已在 node221 启动完整 `python -m pytest ELoRA/tests -q`，尚未结束。自定义 `squeue -o` 的格式引号被跨 shell 解析丢失，状态命令报参数错误，但未影响任何作业。

## 2026-09-01 21:03 +08:00：Job 208 完整套件监控

- 用途：新连接先 pull exact d4，再用默认 `squeue -j 208`、`scontrol show job 208` 与 unit 日志只读确认完整套件终态；CPU/GPU 作业不重跑。
- 权限核对：仅轻量 scheduler/日志读取，不修改远端文件、HEAD 或作业状态。
- 监控结果：Job 208 为 `RUNNING`，node221，4 CPU、16 GiB，运行 1:33/上限 30:00；stdout 确认 exact d4、Python 3.10.12 与完整 suite 命令，stderr 为空。

## 2026-09-01 21:05 +08:00：Job 208 后续监控

- 用途：等待后新连接先 pull exact d4，再读取默认 squeue/scontrol 和 unit stdout/stderr，持续到明确终态。
- 权限核对：只读监控，不取消或改变作业。
- 监控结果：Job 208 运行至 3:04，仍为 `RUNNING`，资源/commit/命令不变，stderr 仍为空。

## 2026-09-01 21:07 +08:00：Job 208 终态等待

- 用途：继续等待后新连接先 pull exact d4，读取 Job 208 scheduler 与最终日志；预期本地基线约 7 分钟，不以静默输出判定停滞。
- 权限核对：只读监控，不修改远端文件或作业。
- 监控结果：Job 208 运行至 4:40，仍为 `RUNNING`；无 stderr 或失败迹象。

## 2026-09-01 21:09 +08:00：Job 208 终态复查

- 用途：继续等待后新连接先 pull exact d4，读取 scheduler 与最终 stdout/stderr。
- 权限核对：只读监控。
- 连接结果：前置 pull 在短超时内未成功，链式门控跳过 scheduler 查询；只读取到既有 unit 启动日志，未执行其他远端操作，作业不受影响。

## 2026-09-01 21:11 +08:00：Job 208 终态重试

- 用途：新连接仍先限时 pull exact d4，成功后才读取 scheduler 与最终日志。
- 权限核对：只读监控，不以旧 refs 绕过 pull。
- 连接结果：第二次终态重试的前置 pull 仍在短超时内失败，scheduler 查询再次被门控跳过，作业未受影响。

## 2026-09-01 21:13 +08:00：网络失败下的只读终态核验

- 用途：新连接首先再次尝试 pull；若仍超时，仅当现场证明 `HEAD == origin/main == d4d9399` 且受管 tree/index 干净时，按既有 TLS 故障经验继续只读 `squeue`/`scontrol`/日志核验，不提交新作业或修改远端状态。
- 权限核对：先 pull 的前置顺序不变；fallback 仅扩大只读诊断，不把旧 refs 用作源码同步，也不执行计算。
- 核验结果：pull 超时退出 124；现场证明 `HEAD == origin/main == d4d9399` 且 tree/index 干净。Job 208 仍为 `RUNNING`，运行 9:23/上限 30:00，资源和空 stderr 正常。

## 2026-09-01 21:16 +08:00：Job 208 长运行监控

- 用途：等待后新连接先尝试 pull；在 exact refs/tree 门控下只读查看 squeue/scontrol、可用时的 sstat 利用率以及日志，判断作业仍在计算还是停滞。
- 权限核对：只读 scheduler/日志诊断，不登录计算节点、不运行额外负载、不取消作业。
- 监控结果：pull 仍超时但 exact refs/tree 门控通过；Job 208 运行 11:18，仍为 `RUNNING`、stderr 为空。`sstat` 可返回 batch 统计但集群字段格式异常，不能据其数值作终态判断；继续以 scheduler 与脚本成功尾标为准。

## 2026-09-01 21:18 +08:00：Job 208 继续等待

- 用途：等待后重复 pull-first、exact refs/tree 与 scheduler/log 终态检查；不因 pytest 输出缓冲而提前取消。
- 权限核对：只读监控。
- 监控结果：pull 成功且 exact refs/tree 一致；Job 208 运行 13:50，仍为 `RUNNING`、stderr 为空。

## 2026-09-01 21:21 +08:00：Job 208 资源与终态复查

- 用途：等待后 pull-first 核验，使用窄 `sstat --format ... -P` 读取 CPU/RSS/I/O（若集群支持），并检查 scheduler/log 终态。
- 权限核对：只读 scheduler 诊断，不增加计算负载或改变作业。
- 监控结果：pull/exact refs/tree 均通过；Job 208 运行 16:21，仍 `RUNNING`、stderr 为空。窄 `sstat` 的 AveCPU 仍返回明显损坏的超大时长且 RSS/I/O 为空，故不采用该指标。

## 2026-09-01 21:24 +08:00：Job 208 接近后半程监控

- 用途：等待后 pull-first 检查 scheduler 与日志终态；保持 30 分钟脚本时限，不主动延长或取消。
- 权限核对：只读监控。
- 监控结果：pull/exact refs/tree 通过；Job 208 运行 19:01/30:00，仍 `RUNNING`、stderr 为空。

## 2026-09-01 21:27 +08:00：Job 208 后段终态监控

- 用途：等待后继续 pull-first 检查 scheduler 和最终日志，直到脚本成功尾标或明确 Slurm 失败。
- 权限核对：只读监控，不改变时限或作业状态。
- 监控结果：pull/exact refs/tree 通过；Job 208 运行 21:40/30:00，仍 `RUNNING`、stderr 为空。

## 2026-09-01 21:30 +08:00：Job 208 时限前监控

- 用途：等待后继续 pull-first 检查；保留 Slurm 30 分钟硬时限，记录最终 pytest 汇总或 timeout 失败。
- 权限核对：只读监控。
- 监控结果：pull/exact refs/tree 通过；Job 208 运行 25:12/30:00，仍 `RUNNING`、stderr 为空。

## 2026-09-01 21:34 +08:00：Job 208 最终时限窗口

- 用途：在剩余时限窗口继续 pull-first 监控，接受成功或 Slurm timeout 的真实结果，不修改测试范围或脚本时限。
- 权限核对：只读监控。
- 监控结果：pull/exact refs/tree 通过；Job 208 运行 27:54/30:00，仍 `RUNNING`、stderr 为空。

## 2026-09-01 21:37 +08:00：Job 208 时限后终态核验

- 用途：等待超过 30 分钟边界后 pull-first 读取最终 scheduler 和完整日志，确认是成功还是 timeout。
- 权限核对：只读终态核验。
- 终态结果：pull/exact refs/tree 通过；Job 208 为 `FAILED`, `Reason=JobLaunchFailure`, `ExitCode=0:15`, runtime 28:33。stderr 记录 Slurm cancellation 与 task Terminated，stdout 没有 pytest 断言失败摘要；按 `--time=00:30:00` 与 `--signal=B:TERM@60` 判定为时限窗口不足，readiness 保持 not_ready。

## 2026-09-01 21:41 +08:00：60 分钟完整 unit 重投计划

- 用途：本地把版本化 unit wall time 调整到 60 分钟并通过提交前检查、commit/push 后，新连接先 pull exact commit，再只重投完整 unit；Jobs 209/210 的成功 smoke 证据继续有效。
- 权限核对：只调整 Slurm 资源时限，不改变测试范围或业务代码；重投和监控仍由 Slurm/登录节点轻量操作完成。
- 连接结果：Guqq fast-forward 到 exact `af5aef4d1f741a6ce1251db8c2d6d55b7ee5653e`，受管 diff 为空；提交完整 unit Job 211。

## 2026-09-01 21:44 +08:00：Job 211 监控

- 用途：新连接先 pull exact af5，再只读检查 Job 211 的 scheduler 与 stdout/stderr，持续至完整 pytest 汇总和终态。
- 权限核对：只读监控，不修改远端 HEAD 或作业。
- 终态结果：Job 211 为 `FAILED`, `ExitCode=2:0`, runtime 4 秒。pytest collection 明确报告 `.cache/mace/46jrkm3v` 为损坏 zip（缺 central directory）；stderr 仅为 srun exit 2 与失败 trap。确认 Job 208 长运行不是完整测试计算，而是首次 foundation 下载留下半文件。

## 2026-09-01 21:46 +08:00：损坏 foundation cache 修复

- 用途：新连接先 pull exact af5；只处理任务 cache 中精确的 `.cache/mace/46jrkm3v`，记录旧 size/hash 后删除，IPv4 重试下载到 `.part`，zip 验证后原子改名并记录新 size/hash。
- 权限核对：最新规范允许在登录节点下载模型及做轻量文件校验；目标严格位于项目任务 `.cache/mace`，不修改 Git 源码、系统环境或其他缓存。
- 连接尝试 1：跨 PowerShell/SSH 的嵌套引号再次被剥离，远端 Bash 在解析阶段退出；未执行 pull、删除、下载或其他状态变更。

## 2026-09-01 21:48 +08:00：Base64 固定脚本重试 cache 修复

- 用途：把同一固定 Bash 脚本编码后传给远端解码执行，避免跨 shell 引号歧义；仍先 pull exact af5，并执行 exact cache/part realpath 门控后才删除损坏文件和下载。
- 权限核对：编码只改变命令传输方式，不扩大目标、网络来源或操作权限。
- 连接结果：pull exact af5 成功；确认旧损坏文件 size 25,731,072、SHA-256 `81c44f00d79a5faac7c902bdd51c7f9f7322ed5dd779f523d243e8a70de64ac0` 后删除。IPv4 下载到 `.part-job211` 仅约 19 KiB/s，预计 29 分钟；在 5% 时主动中断，正式 cache 文件仍不存在，只有不完整 `.part`。

## 2026-09-01 21:51 +08:00：SCP 已验证本地 foundation cache

- 用途：本地 `.cache/mace/46jrkm3v` 已证明 size 32,581,838、zip 完整、SHA-256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`；按服务器网络过慢时允许的 SCP 路径传到 Guqq 的精确临时名 `.cache/mace/46jrkm3v.part-job211`。
- 权限核对：SCP 是规范明确允许的大文件传输方式；覆盖的仅是本任务刚创建且已中断的不完整 `.part`，不触碰正式 cache、Git 源码或其他任务文件。传输后另起已记录的 SSH 连接先 pull，再核对 hash/zip 并原子改名。
- 传输尝试 1：4 秒内被 Guqq port 22 关闭，SCP 退出 1；未假定临时文件完整，正式 cache 仍未创建。

## 2026-09-01 21:52 +08:00：SCP 单次重试

- 用途：对同一本地已验证文件与同一远端临时路径重试一次；成功后必须在独立 SSH 核验，失败则停止重复 SCP 并恢复 IPv4 `.part` 下载。
- 权限核对：目标与上一尝试完全相同，不扩大覆盖范围。
- 传输结果：增加 SSH keepalive 后 SCP 退出 0；文件仅位于临时名，尚未视为有效 cache。

## 2026-09-01 21:54 +08:00：SCP 文件核验与原子启用

- 用途：新 SSH 连接先 pull exact af5；核对 `.part-job211` size、SHA-256 与 zip 完整性均等于本地证据后，删除不存在/损坏的正式目标并原子 `mv` 到 `46jrkm3v`，再复核正式文件。
- 权限核对：只在精确 `.cache/mace` realpath 门控内移动本任务临时文件；不触碰其他 cache 或源码。
- 连接结果：pull exact af5 成功；临时文件 size 32,581,838、SHA-256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`、zip 完整性全部通过，已原子改名并再次复核正式文件。

## 2026-09-01 21:56 +08:00：修复 cache 后重投完整 unit

- 用途：新连接先 pull exact af5 并核对 tree/index，随后只重投完整 unit；立即记录 job ID，再持久只读监控。
- 权限核对：测试仍通过版本化 Slurm 脚本运行，不在登录节点计算；不重复已通过的 CPU/GPU smoke。
- 连接结果：pull/tree/index exact af5；提交完整 unit Job 213。

## 2026-09-01 21:57 +08:00：Job 213 持久监控

- 用途：新连接先 pull exact af5，随后每 30 秒只读 squeue，作业离队后读取 scontrol 与完整 stdout/stderr。
- 权限核对：持久 SSH 只减少重复连接；不修改远端状态或作业。
- 监控进展：Job 213 已运行超过 39 分钟仍在 collection/测试静默阶段。`test_foundations.py` import 会依次加载 MACE-MP large 与三个 MACE-OFF 模型；本地盘点显示这些缓存合计约 215 MiB，Guqq 约 19 KiB/s 外网不足以在 60 分钟内完成。

## 2026-09-01 22:21 +08:00：Job 213 下载状态诊断与条件取消

- 用途：新连接先 pull exact af5；只读列出四个预期 cache 的 size/mtime/hash 与 Job 213 状态。仅当 `5f5yavf3` 等文件被当前作业写成小于本地已验证大小的不完整下载时，取消本任务 Job 213，防止与后续 SCP 并发写同一路径。
- 权限核对：只诊断任务 cache 与本任务 Slurm job；条件取消仅用于避免已证明无法在时限内完成的慢下载，不触碰其他作业或数据。
- 连接尝试 1：Guqq port 22 在命令执行前关闭；未完成 pull/cache 读取或 scancel，持久监控会话与 Job 213 不受影响。

## 2026-09-01 22:22 +08:00：条件诊断单次重试

- 用途与权限：与上一尝试完全相同；仅重试一次 exact cache size 诊断和有证据条件下的 Job 213 取消。
- 连接结果：keepalive 重试仍在执行前被 port 22 关闭；持久监控显示 Job 213 运行至 42:06 后，已主动关闭监控 SSH（不影响 Slurm job）以释放连接槽。

## 2026-09-01 22:24 +08:00：释放持久连接后的 cache 诊断

- 用途：释放旧监控 SSH 后建立单一连接，仍先 pull exact af5，再执行同一 cache size 条件诊断/取消逻辑。
- 权限核对：不扩大上一计划；仅解决 SSH 并发连接限制。
- 连接结果：port 22 仍在命令执行前关闭，未 pull、诊断或取消；判断为临时连接限流，Job 213 继续由 Slurm 运行。

## 2026-09-01 22:25 +08:00：限流冷却后重试

- 用途：等待至少一分钟后重试同一 pull-first 条件诊断；若仍无法连接，则停止高频重连，让 Job 213 的 60 分钟时限自然保护。
- 权限核对：与前述条件诊断一致。
- 连接结果：pull exact af5 成功；`5f5yavf3` 仅 17,309,696 / 133,803,220 字节、mtime 为当前下载，三个 MACE-OFF cache 均缺失。Job 213 运行 44:52，按条件执行 `scancel 213`，避免并发覆盖。

## 2026-09-01 22:27 +08:00：四个 foundation cache 单次目录 SCP

- 用途：本地逐文件 zip/hash 复核后，将四个文件复制到 `.cache/preload-job213/` staging；一次递归 SCP 到远端 `.cache/mace/preload-job213/`，避免直接覆盖正式 cache 和多次连接。
- 权限核对：本地/远端 staging 均为本任务 cache；传输仍使用规范允许的 SCP，不修改源码或系统环境。传输后独立 SSH 先 pull，再验证并原子启用。
- 传输结果：约一分钟后远端主动关闭连接，SCP 退出 1；不假定 staging 中任何文件完整。

## 2026-09-01 22:29 +08:00：远端 preload 完整性盘点

- 用途：新 SSH 先 pull exact af5，只读列出 preload staging 的文件名、size、SHA-256；与本地清单对比，确定可保留的完整文件和需补传的最小集合。
- 权限核对：只读检查任务 staging，不移动或删除文件。
- 连接尝试 1：前置 pull 超时，链式门控未执行目录读取。

## 2026-09-01 22:31 +08:00：preload 盘点 pull fallback

- 用途：新连接先短 pull；若失败，仅在现场证明 `HEAD == origin/main == af5` 且 tree/index 干净后继续只读 staging size/hash 与 Job 213 终态检查。
- 权限核对：沿用既有 TLS fallback，只读诊断不执行状态变更。
- 盘点结果：pull 超时但 exact refs/tree 门控通过；preload 仅含 `5f5yavf3` 24,444,928 字节、SHA-256 `ee82e9…f52a`（预期为未完成前缀），其余三文件未开始。Job 213 已从 controller 清理。

## 2026-09-01 22:34 +08:00：SFTP reput 断点续传

- 用途：用本地 ignored batch 对四个已验证文件依次执行 SFTP `reput` 到远端 preload；连接中断时保留已传前缀，记录后重跑从断点继续，直到 batch 退出 0。
- 权限核对：目标仍是独立任务 staging，不覆盖正式 cache；SFTP 是与 SCP 等价的授权文件传输路径，仅增加断点续传能力。
- 尝试 1：约一分钟后远端关闭连接，SFTP 退出 1；`reput` 保留远端前缀。

## 2026-09-01 22:36 +08:00：SFTP reput 重试 2

- 用途与权限：冷却一分钟后重跑同一 ignored batch，从 staging 已有偏移继续，不改变任何目标。
- 结果：连接保持超过一分钟后被远端关闭并报 broken pipe，退出 1；继续保留断点。

## 2026-09-01 22:39 +08:00：SFTP reput 重试 3

- 用途与权限：冷却后再次运行同一 batch；全部文件传完时 batch 应退出 0，否则继续按连接级记录。
- 结果：约一分钟后再次 broken pipe，退出 1；断点保留。

## 2026-09-01 22:41 +08:00：SFTP 三次后的 staging 盘点

- 用途：冷却后新 SSH 先 pull/fallback exact af5，只读列出 preload 文件大小，确认断点进展和当前文件。
- 权限核对：只读任务 staging，不修改文件。
- 盘点结果：pull 成功；`5f5yavf3` staging 已增至 99,913,728 / 133,803,220 字节，证明 `reput` 正确续传；其他文件尚未开始。

## 2026-09-01 22:43 +08:00：SFTP reput 重试 4

- 用途与权限：同一 batch 从 large 剩余约 33.9 MiB 继续，随后进入三个较小文件；目标不变。
- 结果：约 21 秒后远端关闭连接并 broken pipe，退出 1；需盘点是否恰好完成 large。

## 2026-09-01 22:44 +08:00：重试 4 后盘点

- 用途与权限：冷却后 pull-first 只读列出 preload sizes，确定下一最小传输。
- 盘点结果：pull 超时但 exact refs/tree 通过；large staging 为 128,688,128 / 133,803,220 字节，仅剩约 5.1 MiB。

## 2026-09-01 22:46 +08:00：SFTP reput 重试 5

- 用途与权限：完成 large 剩余前缀并继续三个 MACE-OFF 文件；同一 staging batch。
- 结果：约 14 秒后 broken pipe，退出 1；盘点确认实际完成范围。

## 2026-09-01 22:47 +08:00：重试 5 后盘点

- 用途与权限：冷却后 pull-first 只读列出 staging sizes。
- 盘点结果：pull 超时但 exact refs/tree 通过；large 为 130,736,128 / 133,803,220 字节，还差约 3.1 MiB。

## 2026-09-01 22:49 +08:00：SFTP 低并发 reput 重试 6

- 用途：同一 batch 增加 `-R 1 -B 32768`，把并发请求降为 1、buffer 降为 32 KiB，减少远端 SFTP 压力并继续断点。
- 权限核对：仅调整传输参数，文件和目标不变。
- 结果：连接维持约一分钟；batch 进入 small 后因远端文件不存在而报 `stat remote: No such file`，退出 1。说明 `reput` 需要远端占位文件，需先核对 large 并为三个缺失 staging 创建零长度占位。

## 2026-09-01 22:51 +08:00：核对 large 并创建缺失 staging 占位

- 用途：新 SSH 先 pull/fallback exact af5，核对 large size/hash；仅在 preload realpath 门控内对三个不存在的 MACE-OFF staging 执行 `touch`，使后续 `reput` 可从 offset 0 开始。
- 权限核对：零长度占位只在本任务 preload 目录，不是正式 cache，不覆盖已有文件。
- 结果：pull exact af5；large size 133,803,220、SHA-256 `f80e992b65ab8f88fdf26964511357c022e92704e4d9bcd086652635a8495b32` 通过。三个 MACE-OFF 占位均为 0 字节。

## 2026-09-01 22:53 +08:00：SFTP 低并发 reput 重试 7

- 用途与权限：`-R 1 -B 32768` 同一 batch；large 已完整会快速跳过，三个占位从 offset 0 上传并可续传。
- 结果：`reput` 对已完整 large 报 destination same size or larger 并使 batch 停止；三个 MACE-OFF 未开始。需从 ignored batch 移除已完成项。

## 2026-09-01 22:54 +08:00：仅 MACE-OFF 的 SFTP batch

- 用途与权限：本地 ignored batch 改为只含三个 MACE-OFF `reput`；远端占位和目标不变，继续低并发传输。
- 结果：低并发连接稳定超过 2 分钟后被远端关闭，退出 1；断点保留，需盘点三个文件。

## 2026-09-01 22:57 +08:00：MACE-OFF staging 盘点

- 用途与权限：冷却后 pull-first 只读核对三个 staging size，移除本地 batch 中已完成项后续传剩余最小集合。
- 盘点结果：pull 成功；small 完整 7,347,350 字节，medium 8,486,912 / 18,350,596 字节，large 0 / 55,492,786 字节。large MACE-MP 仍完整。

## 2026-09-01 22:59 +08:00：medium/large SFTP 续传

- 用途与权限：从 ignored batch 移除已完成 small，仅低并发 `reput` medium 与 large。
- 结果：约一分钟后远端关闭连接，退出 1；断点保留。

## 2026-09-01 23:01 +08:00：medium/large 盘点

- 用途与权限：冷却后 pull-first 只读核对 size，继续移除已完成项。
- 盘点结果：pull 超时但 exact refs/tree 通过；medium 为 17,825,792 / 18,350,596 字节（剩约 0.5 MiB），large 仍为 0。

## 2026-09-01 23:03 +08:00：medium/large SFTP 续传 2

- 用途与权限：同一低并发 batch，先完成 medium，再进入 large。
- 结果：约一分钟后远端关闭连接，退出 1；断点保留。

## 2026-09-01 23:05 +08:00：续传 2 后盘点

- 用途与权限：冷却后 pull-first 只读核对 medium/large sizes。
- 连接尝试 1：port 22 在执行前关闭，未读取 staging。

## 2026-09-01 23:06 +08:00：续传 2 盘点重试

- 用途与权限：冷却后重试同一 pull-first 只读 size 盘点。
- 盘点结果：pull 成功；medium 完整 18,350,596 字节，large 为 1,212,416 / 55,492,786 字节，small/MP-large 保持完整。

## 2026-09-01 23:08 +08:00：仅 MACE-OFF large 续传

- 用途与权限：从 ignored batch 移除完整 medium，只对最后一个 large 低并发 `reput`。
- 结果：连接稳定约 3 分钟后被远端关闭，退出 1；断点保留。

## 2026-09-01 23:12 +08:00：最后文件盘点

- 用途与权限：冷却后 pull-first 只读核对 MACE-OFF large size；完整则进入四文件 hash/zip 验证，不完整则只续传剩余。
- 盘点结果：pull 超时但 exact refs/tree 通过；MACE-OFF large 为 32,964,608 / 55,492,786 字节，剩约 22.5 MiB。

## 2026-09-01 23:15 +08:00：最后文件续传 2

- 用途与权限：同一低并发 batch 只续传 MACE-OFF large 剩余部分。
- 结果：约两分钟后远端关闭连接，退出 1；断点保留。

## 2026-09-01 23:18 +08:00：最后文件终态盘点

- 用途与权限：冷却后 pull-first 核对最终 size；完整则进入四文件 hash/zip 验证。
- 盘点结果：pull 超时但 exact refs/tree 通过；MACE-OFF large 为 41,385,984 / 55,492,786 字节，还差约 14.1 MiB。

## 2026-09-01 23:20 +08:00：最后文件续传 3

- 用途与权限：同一低并发 batch 续传剩余约 14.1 MiB。
- 结果：SFTP batch 退出 0，最后一个 `reput` 与 `bye` 均完成；进入统一完整性验证。

## 2026-09-01 23:23 +08:00：四文件验证与原子启用

- 用途：新 SSH 先 pull/fallback exact af5；对 preload 四文件逐一断言本地清单的 size/SHA-256，并用 Python zipfile 全量检查。全部通过后才在 `.cache/mace` exact realpath 门控下以 staging 原子替换正式 cache，再复核。
- 权限核对：仅替换已证明损坏/缺失且属于本任务的四个 foundation cache；不触碰 small `46jrkm3v`、源码或其他缓存。
- 连接结果：pull 超时但 exact refs/tree 门控通过；四个 preload 的 size/SHA-256 与本地清单全部一致，zipfile 全量检查通过。原子替换正式 cache 后四个 hash 再次通过，preload 目录已空并移除。

## 2026-09-01 23:25 +08:00：完整 foundation cache 后重投 unit

- 用途：新连接先 pull/fallback exact af5，核对 tree/index 后提交完整 unit；不再允许 foundation 下载，持久监控直至终态。
- 权限核对：版本化 Slurm 完整 suite，登录节点只提交/监控；CPU/GPU smoke 不重跑。
- 连接结果：pull 超时但 exact refs/tree 门控通过；提交完整 unit Job 216。

## 2026-09-01 23:27 +08:00：Job 216 持久监控

- 用途：新连接先 pull/fallback exact af5，随后每 30 秒只读 squeue，离队后获取 scontrol 与完整 stdout/stderr。
- 权限核对：持久 SSH 只用于监控，不修改远端状态。
- 终态结果：Job 216 为 `FAILED`, `ExitCode=139:0`, runtime 6:39；stdout 已通过 12 项后停止。stderr 为 Python segmentation fault，当前线程位于 `/tmp/torchinductor_xmz` 生成 kernel 的 forces 二阶反向，调用来自 `test_compile.py::test_mace`；不是 assertion failure 或 foundation 下载。

## 2026-09-01 23:36 +08:00：CPU Inductor 两 profile 诊断计划

- 用途：本地新增并验证版本化诊断 SBATCH、commit/push 后，新连接先 pull exact commit，提交 fresh per-job cache 与 fresh-cache+single-thread 两个定向 CPU compile jobs，比较终态。
- 权限核对：两项均是完整 suite 失败点的最小 Slurm 复现，不在登录节点计算，不跳过真实 compile/forces backward。
