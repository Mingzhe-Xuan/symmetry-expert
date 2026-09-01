# ELoRA 对称专家预实验：Codex Goal 入口

本文将 `docs/preexperiment_elora_experts.md` 中的研究计划拆成两个具有单一目标、明确验收条件和停止条件的 Codex Goal。不要把整份预实验矩阵作为一个 Goal 直接启动。

## 推荐执行顺序

1. 先执行 [Goal 0：实现与就绪性验证](goal_0_elora_readiness.md)。
2. 只有 Goal 0 输出 `readiness: ready`，并且正式数据、基础模型与计算预算均已指定后，才执行 [Goal 1：正式预实验](goal_1_elora_preexperiment.md)。
3. Goal 1 完成后再由研究者决定是否进入更大规模、多性质或多 foundation model 阶段；不得自动扩展实验范围。

## 共同依据

- `docs/AGENTS.md`：最高优先级的开发、服务器、Slurm、测试与记录规范；两个 Goal 全程强制遵守。
- `docs/slurm_template.md`：经 Guqq 实机检查的单节点提交模板与集群限制；每次启动时仍须重新核验状态。
- `docs/elora_paper_logic.md`：ELoRA 论文的实验逻辑与仓库审计。
- `docs/preexperiment_elora_experts.md`：本项目预实验的完整研究规范。
- `docs/plan_v2.md`：总体问题、笛卡尔积实验空间与长期路线。
- `ELoRA/paper.pdf`：原始论文。

若不同文档发生冲突，以本入口页和对应 Goal 文档中的执行边界、验收与停止条件为准；科学定义与实验动机仍以预实验规范和 `plan_v2.md` 为准。

## 强制服务器工作流

两个 Goal 都不能只在本地完成。Codex 必须逐条遵守 `docs/AGENTS.md`，并按以下闭环执行：

1. 本地修改前规划测试，在 `docs/state.md` 记录计划；每个 commit 级进展写入 `docs/update.md`，测试计划与结果写入 `docs/test.md`，环境写入 `docs/env.md`。
2. 本地运行与改动对应的快速单元测试；测试失败时不得提交，也不得为通过测试而降低标准。
3. 本地提交并推送代码。大型数据集和模型权重不得进入 Git，必要时通过 `scp` 同步。
4. 每次连接服务器前，先在 `docs/gpu.md` 记录连接用途并确认权限，然后使用 `ssh Guqq`。
5. 服务器上先执行 `git pull` 同步最新代码；禁止直接编辑服务器文件。
6. 服务器使用 Slurm 提交单元测试、smoke test、数据统计和训练任务；禁止在登录节点直接运行任务。
7. 保存 job ID、提交脚本、commit、环境、stdout/stderr、退出码和测试结果；必要结果通过 `scp` 拉回本地，并同步到对应文档。
8. 同一任务连续失败 3 次时，先查阅并更新 `docs/lessons.md`，再继续尝试。

缺少“本地测试通过—代码同步—服务器 Slurm 测试/任务通过—结果回传与记录”中的任何一环，都不得将 Goal 标为完成。

Guqq 当前只有 `compute` 分区、单节点和单张 GPU，且禁用了 Slurm accounting。不得使用 ELoRA 自带的 `gpu` 分区、2 节点/20 GPU 示例，也不得依赖 `sacct` 作为历史终态来源；具体模板和实测依据见 `docs/slurm_template.md`。

## Goal 1 启动前必须冻结

| 字段 | 当前值 | 要求 |
|---|---|---|
| `DATASET_PATH` | `AUTO_PLAN` | 必须位于 `/home/xmz/symmetry-expert/data/` |
| `DATASET_VERSION` | `AUTO_FREEZE` | Codex 自主选择并记录版本、来源与内容哈希 |
| `FOUNDATION_MODEL` | `AUTO_PLAN` | 必须位于 `/home/xmz/symmetry-expert/artifacts/foundation/` |
| `FOUNDATION_MODEL_HASH` | `AUTO_COMPUTE` | 下载或同步后计算内容哈希 |
| `GPU_DEVICE` | `Guqq/node221: 1×RTX 5090, 32607 MiB` | 2026-09-01 实测；启动时复核 |
| `REMOTE_REPO_PATH` | `/home/xmz/symmetry-expert` | origin、分支、commit 与干净状态已核验 |
| `REMOTE_VENV_PATH` | `/home/xmz/symmetry-expert/.venv` | Goal 0 可在此创建环境并安装包 |
| `OUTPUT_ROOT` | `/home/xmz/symmetry-expert/outputs/preexperiment/` | 所有正式输出均置于仓库目录下 |
| `COMPUTE_BUDGET` | `AUTO_FREEZE` | Codex 根据 pilot 和单卡资源估算 GPU-hours/最大 run 数并在训练前冻结 |
| `SYMMETRY_LABEL_MODE` | `AUTO_FREEZE` | 优先 parent label；否则固定 spglib 版本与容差 |
| `PLANNING_MANIFEST` | `/home/xmz/symmetry-expert/experiments/preexperiment/plan_frozen.yaml` | 首个正式 run 前生成并冻结 |

上述 `AUTO_PLAN`、`AUTO_FREEZE` 和 `AUTO_COMPUTE` 项由 Codex 自主完成，无需逐项等待用户确认，但必须在首个正式 run 前写入 planning manifest。所有数据、checkpoint、缓存、环境、脚本、日志和输出必须位于 `/home/xmz/symmetry-expert/` 下，并将大型文件加入 `.gitignore`。无法形成自洽且满足资源约束的冻结方案时，Goal 1 才报告阻塞。

## 状态约定

- `ready`：所有验收项通过，可以启动下一 Goal。
- `not_ready`：仍有失败项；必须列出失败测试、复现命令和下一步。
- `complete`：当前 Goal 的全部交付物与停止条件满足。
- 科学假设未获得正向结果不属于阻塞或失败。完整、可复现的阴性结果同样可以完成 Goal 1。
