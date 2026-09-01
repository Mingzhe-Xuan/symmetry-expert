# 服务器连接记录

## 2026-09-01：Goal 0 Slurm 闭环（计划连接）

- 服务器：`Guqq`
- 用途：在本地最终测试、commit/push 和 wheelhouse 哈希完成后，同步固定 commit，并仅通过 Slurm 创建 `/home/xmz/symmetry-expert/.venv`、运行 selected unit、CPU smoke 和 RTX 5090 GPU smoke。
- 计划操作：先只读核对 origin/branch/clean status、`git pull --ff-only`、commit、`sinfo`/partition/node/GRES；通过 SCP 把本地离线 wheelhouse 放入仓库 `.cache/`；提交四个版本化 sbatch，轮询 `squeue`/`scontrol` 并回传日志和 JSON。
- 限制：不在登录节点安装、测试、计算或编辑；不使用 `sacct`；不下载正式数据或模型；不进入 Goal 1。
- 状态：已连接。Guqq 已 pull 至 `098db4a20def32bdbebfa70e15924ecf61413e13`；`compute` UP、`node221` IDLE、`Gres=gpu:1`。远端仓库仅有既存未跟踪 `net.sh`，未移动、删除或编辑。
- 传输：50-wheel wheelhouse 中小文件已传输；820 MB torch wheel 的 SCP/SFTP 连接连续中断 3 次，最大保留 offset 约 447 MB。停止重复单流，改用 13 × ≤64 MiB 哈希分片；尚未提交 Slurm job。

## 2026-09-01 02:19 +08:00

- 服务器：`Guqq`
- 用途：只读核验 Slurm 版本、分区、GPU 资源、任务提交参数、项目路径与 Python/Conda 环境，以判断 `docs/slurm_template.md` 是否适配服务器。
- 计划操作：登录后仅执行状态与配置查询命令；不编辑服务器文件，不直接运行计算任务，不下载数据，不提交训练任务。
- 权限核对：符合 `docs/AGENTS.md` 中“服务器负责查看代码、拉取同步和通过 Slurm 提交任务”的范围。本次不执行 `git pull`，因为尚未定位服务器项目目录且只进行系统级只读检查；定位后如需进入项目执行任务，将先核对工作树并执行 `git pull`。
- 结果：未连接到服务器。当前受限执行环境将 `Guqq` 解析为普通主机名 `guqq`，且无法读取用户 SSH config，连接在 DNS 解析阶段失败；服务器端没有执行任何命令或产生任何修改。

## 2026-09-01 02:23 +08:00

- 服务器：`Guqq`
- 用途：再次连接并只读核验集群实际 Slurm 版本、分区、GPU/GRES、QoS、账户限制、现有作业脚本和项目环境，据此校正 Goal 文档中的 Slurm 工作流。
- 计划操作：仅查询 SSH/Slurm/项目状态并读取已有脚本；不修改服务器文件，不直接运行计算任务，不提交作业，不下载数据。
- 权限核对：符合 `docs/AGENTS.md` 的服务器查看权限。定位项目目录后，若后续需要执行任务，将先按规范同步代码；本次只做模板适配性审计。
- 结果：成功以用户 `xmz` 连接 `node221`。实测 Slurm 21.08.5，cluster `ustc-gu-221`；唯一默认分区为 `compute`，单节点 48 CPU、257787 MiB、通用 GRES `gpu:1`。GPU 为 NVIDIA GeForce RTX 5090，32607 MiB，驱动 570.211.01，compute capability 12.0。`accounting_storage/none`，因此 `sacct` 不可用；最大 job array 为 1001。非交互 shell 仅直接发现 `/usr/bin/python3` 3.10.12，未发现全局 conda/uv/module。`/home/xmz/expert` 不存在，尚需确定远端仓库和环境路径。未发现现有 SBATCH 脚本。使用 `sbatch --test-only` 验证 CPU 模板、`--gres=gpu:1` 和 `--gpus-per-node=1` 均可解析；随后确认用户队列为空，test-only 显示的 job ID 均不存在，没有提交或运行实际作业。服务器端未修改任何文件。

## 2026-09-01：同步 symmetry-expert 仓库

- 服务器：`Guqq`
- 用途：按用户明确要求，将 `https://github.com/Mingzhe-Xuan/symmetry-expert.git` 克隆到 `/home/xmz/symmetry-expert`，作为后续 Goal 的远端仓库。
- 计划操作：先确认目标路径不存在，再执行一次 `git clone`；随后只读核对 origin、当前分支、commit 与工作树状态。不提交 Slurm 作业，不运行项目代码。
- 权限核对：用户已明确授权本次服务器端 clone；目标限定为 `/home/xmz/symmetry-expert`。
- 结果：Guqq 直连 GitHub 443 在约 133 秒后超时，失败的 `git clone` 自动清理了目标目录。随后在本地从指定 URL 创建干净浅克隆，并通过 `scp` 传到服务器；因 Windows checkout 换行导致首次传输工作树显示 5 个文档被修改，故在服务器从已传输的 `.git` 对象库重新执行 Linux checkout。最终仓库位于 `/home/xmz/symmetry-expert`，origin 为指定 GitHub URL，分支 `main`，commit `b09a25a299b4d0750c56fc5b0379d376ed0694e0`，shallow repository，工作树干净。首次 SCP 目录保留在 `/home/xmz/symmetry-expert-scp-backup-20260901`，可恢复但不应作为任务目录。本次未提交 Slurm 作业或运行项目代码。

## 2026-09-01：复核 GitHub 连接并同步

- 服务器：`Guqq`
- 用途：根据用户反馈复核服务器 GitHub 连接，并在干净工作树上执行 fast-forward-only pull。
- 计划操作：只读核对 origin、分支和状态；仅在工作树干净时执行 `git pull --ff-only`，随后记录 commit。不运行项目代码或提交 Slurm 作业。
- 权限核对：符合 `docs/AGENTS.md` 中服务器负责 `git pull` 同步的范围。
- 结果：先核对 `/home/xmz/symmetry-expert` 的 origin、`main` 分支、commit 与工作树，确认工作树干净；随后 `git pull --ff-only` 成功访问 GitHub 并返回 `Already up to date.`。commit 仍为 `b09a25a299b4d0750c56fc5b0379d376ed0694e0`。随后执行 `git fetch --unshallow origin`，仓库已补全为普通非 shallow 仓库；远端当前总计 1 个 commit，工作树保持干净。本次没有运行项目代码或提交 Slurm 作业。

## 2026-09-01: resumable wheelhouse transfer and readiness submission

- Server: `Guqq`.
- Purpose: resume the interrupted local-to-server wheelhouse upload with a workspace-local `rsync` client, then submit the versioned setup and readiness checks through Slurm after the upload is complete.
- Planned operations: first run `git pull --ff-only` in `/home/xmz/symmetry-expert`; inspect commit/status and transfer progress; use only `rsync`/SSH transport to populate the ignored `.cache/wheelhouse`; submit `setup_readiness.sbatch`, selected unit tests, CPU smoke, and RTX 5090 GPU smoke; inspect scheduler state and logs.
- Permission check: Git synchronization, SCP-equivalent file transfer, read-only server inspection, and Slurm submission are within `docs/AGENTS.md`. No package installation, testing, computation, direct file editing, or internet download will run on the login node.
- Status: connected successfully; `git pull --ff-only` reported up to date at `522b348af15a6ed07a18edd46569c04ee6a7f507`, `/usr/bin/rsync` is available, and `compute` is up with `gpu:1`. The known unrelated untracked `net.sh` remains untouched. No wheel transfer or Slurm job was started before the requirements update. Local portable `rsync` 3.5.0 and its `popt` dependency were downloaded from official MSYS2 packages into ignored `.cache` and verified against the published SHA-256 values.

## 2026-09-01: server-side dependency download fallback

- Server: `Guqq`.
- Purpose: pull the tested requirements/setup commits and submit the environment setup through Slurm, using the newly authorized server internet access only when the transferred wheelhouse is incomplete.
- Planned operations: `git pull --ff-only` first; submit `setup_readiness.sbatch` with the exact commit; inspect `squeue`, `scontrol`, and job logs. The compute job may download required libraries/packages/wheels from PyPI and the official PyTorch CUDA 12.8 index into `/home/xmz/symmetry-expert/.cache/pip`.
- Permission check: explicitly allowed by the updated `docs/AGENTS.md`. No dependency download, installation, testing, or computation will run directly on the login node.
- Status: connected; `git pull --ff-only` fast-forwarded Guqq to `1afac68636e8b8a9c2c60f6199b934f1c424ed2f`. Login-node environment checks found Python 3.10.12 and `/usr/bin/curl`. The known unrelated `?? net.sh` remains untouched. Submitted setup Slurm Job 201. The next connection will be a persistent read-only scheduler/log monitoring session and will begin with another `git pull --ff-only`.

- Monitoring result: persistent session began with `git pull --ff-only` (`Already up to date`). Job 201 selected the online fallback but its pip process remained on an unreachable IPv6 socket in `SYN-SENT` for more than 6 minutes. A permitted login-node environment check reached PyPI with HTTP 200 over the working path. Job 201 was cancelled (`CANCELLED`, `ExitCode=0:15`) before changing the timeout strategy; no readiness test was reached.
- Next action in the same persistent session: after a tested fix is pushed, run `git pull --ff-only` again and submit a replacement setup job with the exact new commit.
