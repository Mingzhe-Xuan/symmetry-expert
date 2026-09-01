# Guqq 服务器 Slurm 任务提交模板

本模板遵循 `docs/AGENTS.md`：代码只在本地修改和提交；连接服务器前记录 `docs/gpu.md`；服务器首先执行 `git pull`；单元测试、数据统计、smoke test 和训练均通过 Slurm 提交，禁止在登录节点直接运行，也不得直接修改服务器文件。

## 0. Guqq 实测配置

以下信息于 2026-09-01 通过 `ssh Guqq` 只读查询，并用 `sbatch --test-only` 验证模板参数：

| 项目 | Guqq 实测值 |
|---|---|
| Slurm | 21.08.5，cluster `ustc-gu-221` |
| 分区 | `compute`，默认且唯一分区 |
| 节点 | `node221`，单节点 |
| CPU / 内存 | 48 CPU，257787 MiB Slurm RealMemory |
| GPU | 1 × NVIDIA GeForce RTX 5090，32607 MiB，compute capability 12.0 |
| 驱动 | 570.211.01 |
| GRES | 通用 `gpu:1`，没有 GPU type 标签 |
| 最大 job array | 1001 |
| accounting | `accounting_storage/none`；`sacct` 不可用于可靠的历史终态查询 |
| 本项目远端目录 | `/home/xmz/symmetry-expert`；origin 已核对，完整仓库且工作树干净 |
| 非交互 shell | 仅直接发现 `/usr/bin/python3` 3.10.12；未发现全局 `conda`、`uv` 或 `module` |

`--gres=gpu:1` 与 `--gpus-per-node=1` 均通过 `sbatch --test-only`；本模板采用 `--gres=gpu:1`。CPU 任务不请求 GPU。RTX 5090 需要框架支持 `sm_120`；正式训练前必须在 Slurm GPU smoke test 中验证 PyTorch/CUDA、设备识别和实际 CUDA kernel，不能只运行 `nvidia-smi`。

集群状态可能变化。每个 Goal 启动时仍须重新运行 `sinfo`、`scontrol show partition compute` 和 `scontrol show node node221`，并记录快照；若与本节冲突，以当次实测为准并先更新模板。

## 1. 使用前检查

提交前必须确认：

- 本地相关单元测试已经通过，命令和结果已写入 `docs/test.md`。
- 当前代码已经 commit 并 push；`docs/update.md` 已记录本次进展。
- 大型数据集和模型权重未进入 Git，必要文件已按 `docs/AGENTS.md` 通过 `scp` 同步。
- `docs/env.md` 已记录环境、Python、CUDA、PyTorch 和主要依赖版本。
- 连接服务器的用途已在 `docs/gpu.md` 中记录。

模板中的 `REQUIRED_*` 必须在本地替换后再提交到 Git。不要在服务器上编辑脚本。

本项目的服务器仓库固定为 `/home/xmz/symmetry-expert`，虚拟环境固定为 `/home/xmz/symmetry-expert/.venv`。Goal 0 已获用户授权在该目录创建虚拟环境并安装依赖，但不得写入系统 Python、全局环境或仓库目录以外的位置。

## 2. 通用单节点任务模板

将下面内容复制为版本控制内的 `.sbatch` 文件，例如 `scripts/slurm/unit_test.sbatch`。默认资源适合单元测试或小型 smoke test；训练任务应按实际情况调整时间、CPU、内存和 GPU。

```bash
#!/usr/bin/env bash
#SBATCH --job-name=REQUIRED_JOB_NAME
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:TERM@60

# GPU 任务启用下一行；CPU 单元测试保持注释。
##SBATCH --gres=gpu:1

set -Eeuo pipefail

on_error() {
    status=$?
    echo "[error] exit_code=${status} host=$(hostname) time=$(date --iso-8601=seconds)" >&2
    exit "${status}"
}
trap on_error ERR

: "${EXPECTED_COMMIT:?Submit with EXPECTED_COMMIT set to the pulled Git commit}"

# 必须在本地改成服务器上的仓库绝对路径，然后 commit、push、git pull。
PROJECT_DIR="/home/xmz/symmetry-expert"

cd "${PROJECT_DIR}"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Commit mismatch: expected=${EXPECTED_COMMIT}, actual=${ACTUAL_COMMIT}" >&2
    exit 2
fi

echo "job_id=${SLURM_JOB_ID}"
echo "job_name=${SLURM_JOB_NAME}"
echo "host=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "project_dir=${PROJECT_DIR}"
echo "commit=${ACTUAL_COMMIT}"
echo "start_time=$(date --iso-8601=seconds)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi
fi

# Guqq 非交互 shell 当前没有全局 conda、uv 或 module；使用明确的 venv 绝对路径。
source /home/xmz/symmetry-expert/.venv/bin/activate

python --version

# 按任务替换命令。使用数组可保留参数边界，避免 eval。
CMD=(python -m pytest tests/REQUIRED_TEST_TARGET -q)

echo "command:"
printf ' %q' "${CMD[@]}"
printf '\n'

srun --ntasks=1 "${CMD[@]}"

echo "end_time=$(date --iso-8601=seconds)"
echo "status=success"
```

### GPU 训练时的最小改动

- 将 `##SBATCH --gres=gpu:1` 改为 `#SBATCH --gres=gpu:1`。
- 保持 `--partition=compute`；按任务需要调整 `--time`、`--cpus-per-task` 和 `--mem`，但不得超过单节点 48 CPU 与 257787 MiB。
- 把 `CMD` 换成训练入口；单 GPU 仍保留 `srun --ntasks=1`。
- Guqq 当前只有 1 个节点和 1 张 GPU，不运行多节点或多 GPU DDP。`ELoRA/scripts/distributed_example.sbatch` 中的 `gpu` 分区、2 节点、20 tasks、20 GPU 和 `--exclusive` 均不适配本集群，不得直接使用。

## 3. 标准提交步骤

以下命令仅展示顺序。连接服务器前，先在本地完成 `docs/gpu.md` 记录、commit 和 push。

```bash
ssh Guqq
cd /home/xmz/symmetry-expert
git pull
git status --short
git rev-parse HEAD

sbatch \
  --export=ALL,EXPECTED_COMMIT="$(git rev-parse HEAD)" \
  scripts/slurm/unit_test.sbatch
```

`git status --short` 应为空；若服务器存在未提交修改，不要覆盖或清理，停止并调查来源。`sbatch` 返回的 job ID 必须立即记录。

## 4. 查看状态与结果

查看调度状态不等于直接运行计算任务，可以在登录节点执行：

```bash
squeue --job REQUIRED_JOB_ID
scontrol show job REQUIRED_JOB_ID
```

Guqq 当前禁用了 Slurm accounting，因此 `sacct` 会返回 `Slurm accounting storage is disabled`。任务排队或运行时使用 `squeue`；任务结束后应立即抓取 `scontrol show job`，并以批处理脚本写出的明确退出状态、stdout/stderr 和预期产物作为长期记录。不得把 `sacct` 作为完成 Goal 的必要条件。

任务结束后检查：

- `slurm-<job-name>-<job-id>.out`
- `slurm-<job-name>-<job-id>.err`
- `scontrol show job` 中可获得的终态与退出码，以及脚本自身记录的程序退出码
- 模型 checkpoint、指标、统计文件等任务产物

不要只依据 stdout 中出现“通过”判断成功；还要核对 Slurm 状态、退出码和预期产物。结果需要回传时，在本地通过 `scp` 拉取，不在服务器上使用互联网下载工具。

## 5. 必须记录的字段

每个任务至少在 `docs/test.md`、实验表或 `slurm_jobs.csv` 中记录：

| 字段 | 内容 |
|---|---|
| run ID | 项目内唯一标识 |
| Git commit | 与 `EXPECTED_COMMIT` 一致的完整哈希 |
| Slurm job ID | `sbatch` 返回值 |
| 提交脚本 | 仓库内路径和脚本哈希 |
| 任务类型 | unit test、smoke、statistics、benchmark 或 train |
| 资源 | partition、节点、CPU、内存、GPU、time limit |
| 环境 | Python、CUDA、PyTorch、环境名或 lockfile |
| 输入 | 数据、split、checkpoint 和配置哈希 |
| 输出 | stdout、stderr、checkpoint 和结果路径 |
| 终态 | `COMPLETED`、`FAILED`、`TIMEOUT`、`CANCELLED` 等 |
| 退出码 | Slurm `ExitCode` 与程序退出码 |
| 重试来源 | 首次提交留空；重试时填写前一个 job ID |

若同一任务连续失败 3 次，应停止重复提交，查阅并更新 `docs/lessons.md`，记录原因和新的处理方案后再继续。
