# Goal 1：ELoRA 对称专家正式预实验

## 启动参数

下表中的 `AUTO_*` 项由 Codex 自主规划并在首个正式 run 前冻结，无需逐项等待用户确认：

| 参数 | 值 |
|---|---|
| `DATASET_PATH` | `AUTO_PLAN`，必须位于 `/home/xmz/symmetry-expert/data/` |
| `DATASET_VERSION` | `AUTO_FREEZE` |
| `FOUNDATION_MODEL` | `AUTO_PLAN`，必须位于 `/home/xmz/symmetry-expert/artifacts/foundation/` |
| `FOUNDATION_MODEL_HASH` | `AUTO_COMPUTE` |
| `GPU_DEVICE` | `Guqq/node221: 1×RTX 5090, 32607 MiB`（启动时复核） |
| `REMOTE_REPO_PATH` | `/home/xmz/symmetry-expert` |
| `REMOTE_VENV_PATH` | `/home/xmz/symmetry-expert/.venv` |
| `OUTPUT_ROOT` | `/home/xmz/symmetry-expert/outputs/preexperiment/` |
| `COMPUTE_BUDGET` | `AUTO_FREEZE`，根据 pilot 估算 GPU-hours 与最大 run 数 |
| `SYMMETRY_LABEL_MODE` | `AUTO_FREEZE`，优先 parent label，否则固定 spglib 版本与容差 |
| `PLANNING_MANIFEST` | `/home/xmz/symmetry-expert/experiments/preexperiment/plan_frozen.yaml` |

## 可直接提交给 Codex 的 Goal

> /goal 首先完整阅读并全程严格遵守 docs/AGENTS.md 和经 Guqq 实机核验的 docs/slurm_template.md。在 Goal 0 的 docs/preexperiment_readiness.md 为 readiness: ready 后，依据 docs/preexperiment_elora_experts.md 执行 ELoRA 对称专家正式预实验。自主规划数据集、checkpoint、标签流程和单卡计算预算，将所有数据、环境、缓存、脚本、日志、checkpoint 与输出限制在 /home/xmz/symmetry-expert/ 下，并在首个正式 run 前冻结 plan_frozen.yaml。Stage 0 先按空间群执行 n_g<100 类别删除审计；若因此删除的去重独立结构总数严格大于 2000，则自动把主分类粒度切换为晶系，在原始合格去重集合上按晶系重新执行 n_g<100 筛选，后续主专家、Learned-K、Random-K、宏平均指标和全部矩阵均使用晶系；不得根据模型结果选择粒度。启动时重新核验 Guqq 状态；使用唯一的 compute 分区和单节点单张 RTX 5090，GPU 请求为通用 gres gpu:1，不得照搬 ELoRA 的 gpu 分区、2 节点/20 GPU 示例，也不得依赖已禁用的 sacct。代码与配置必须在本地测试、提交并推送；每次连接服务器前更新 docs/gpu.md，服务器在 /home/xmz/symmetry-expert 先 git pull，激活 /home/xmz/symmetry-expert/.venv，所有数据统计、单元测试、benchmark 和训练都通过 Slurm 提交。保存完整任务记录并执行 S_scope × P_update × R_router × N_data 核心矩阵。输出 docs/preexperiment_results.md 和可机读结果；阴性科学结果可以完成 Goal，但核心主比较的技术运行必须成功，不能用 failed-reproducible 替代。完成后停止，不自动扩展到其他模型或性质。

## 单一目标

在固定基础模型、数据划分和训练预算下，判断 ELoRA 专家训练相较共享模型与直接参数更新是否改善小样本性能或效率，并描述该效应随 scope、update、router 与训练数据量的变化。

## 前置检查

- 已完整阅读 `docs/AGENTS.md`，并将其视为服务器操作、Slurm、单元测试和文档记录的强制验收规范。
- 已重新核验 `docs/slurm_template.md` 中的 Guqq 快照；远端仓库与环境分别为 `/home/xmz/symmetry-expert` 和 `/home/xmz/symmetry-expert/.venv`。
- Goal 0 报告存在且为 `readiness: ready`。
- Codex 已自主完成所有 `AUTO_*` 项，数据与 checkpoint 可读且哈希一致，planning manifest 已冻结。
- 根据 pilot 估算的预算能够覆盖冻结后的最低核心矩阵；否则在训练前缩减矩阵并写入 planning manifest，不得边看结果边修改。
- 代码版本、环境 lockfile、GPU 型号和随机种子已记录。

任一条件不满足时只输出阻塞说明，不开始正式训练。

## 路径与自主规划权限

- Codex 有权自主选择预实验数据、MACE checkpoint、标签生成设置、预算和输出组织，无需为每个 `AUTO_*` 项请求用户确认。
- 所有服务器端文件必须位于 `/home/xmz/symmetry-expert/` 下。推荐分别使用 `data/`、`artifacts/foundation/`、`.venv/`、`.cache/`、`experiments/preexperiment/` 和 `outputs/preexperiment/`。
- 大型数据、权重、虚拟环境、缓存和训练输出必须加入 `.gitignore`；Git 只跟踪配置、索引、哈希、统计摘要、脚本和轻量结果。
- 自主规划必须写入 `plan_frozen.yaml`，至少包括数据来源/版本/路径/哈希、checkpoint、标签方式、分类门控结果、split、矩阵、seed、预算、停止门槛和预期产物。

## 强制服务器执行闭环

1. 本地修改代码或配置前，在 `docs/state.md` 记录阶段、计划与验证方式。
2. 每次 commit 前运行相关本地单元测试，并把命令、环境、退出码和结果写入 `docs/test.md`；失败时不得提交或降低标准。
3. 本地测试通过后提交并推送；在 `docs/update.md` 记录 commit 级进展。大型数据和权重不得提交到 Git。
4. 每次 `ssh Guqq` 前先在 `docs/gpu.md` 记录连接用途并检查权限；连接后首先在服务器项目目录执行 `git pull`，不得直接编辑服务器文件。
5. 服务器上的单元测试、数据统计、benchmark 与全部训练必须通过 Slurm 提交，禁止在登录节点直接运行。
6. 每个 Slurm 任务必须绑定唯一 run ID，并记录代码 commit、配置与数据哈希、提交脚本、job ID、分区/GPU、环境、stdout/stderr、退出码、重试关系和输出路径。
7. 结果按 `docs/AGENTS.md` 通过 Git 或 `scp` 回传并进入可机读记录；每轮同步后核对本地与服务器 commit 一致。
8. 同一任务连续失败 3 次时，停止盲目重试，查阅并更新 `docs/lessons.md` 后再继续。

Guqq 当前只允许本计划使用单节点单 GPU 路径：`--partition=compute` 和 `--gres=gpu:1`。不添加 `--account`、`--qos`、typed GRES、多节点、多 GPU 或 `--exclusive`。由于 accounting disabled，运行期使用 `squeue`，结束后立即抓取 `scontrol show job`，长期终态以批处理脚本的退出状态、stdout/stderr 和预期产物共同确认。

## Stage 0：数据统计、筛选与冻结

训练前必须生成并人工可读地检查：

- `dataset_summary.md`
- `class_counts.csv`
- `removed_classes.csv`
- `split_counts.csv`
- `dataset_manifest.json`
- `selected_categories.json`
- `classification_decision.json`
- `nested_train_subsets.json`
- `split_hash.txt`
- `figures/` 中的类别频数、长尾、性质分布和交叉分布图

统计必须描述：数据版本与来源、原始和去重后结构数、构型与 parent structure 数、元素和原子数分布、目标性质分布、晶系/点群/空间群逐类数量和占比、各粒度不平衡指标及晶系—点群—空间群交叉表。

先在完成标签有效性检查、标准化和去重后的独立结构集合上统计空间群。对每个空间群应用 `n_g < 100` 整类删除规则，并计算因此删除的去重独立结构总数 `N_removed_SG`，不能用 configuration/frame 数替代。

- 若 `N_removed_SG <= 2000`：冻结 `classification_level: space_group`，以筛选后的空间群为主分类。
- 若 `N_removed_SG > 2000`：冻结 `classification_level: crystal_system`；放弃空间群删除后的集合，回到应用空间群删除前的同一合格去重集合，按晶系统计并重新应用 `n_g < 100` 整类删除规则。后续主专家和评价均按晶系分类。

该门控只由训练前数据统计决定，不得根据任何模型误差更改。`classification_decision.json` 必须记录阈值 2000、严格比较符号、空间群逐类计数、`N_removed_SG`、最终分类粒度、晶系重筛选结果、输入 manifest 哈希和生成代码 commit。

选定分类粒度后才执行按 parent structure 分组的 train/val/test 划分。每个 split 逐类计数必须与 manifest 守恒，且不得发生 parent 泄漏。嵌套训练子集必须从同一最大训练池构造并冻结。

Stage 0 未通过验收时，不得开始任何用于论文结论的训练。

## Stage 1：ELoRA 基准复现

先按 `docs/elora_paper_logic.md` 中识别的原论文协议运行仓库自带或最接近的可执行 benchmark，验证训练、评估和 checkpoint 链路。报告原论文设置、当前实现的对应关系、不可复现差异、指标与资源消耗。

若关键基准无法复现，保留日志并判定是否影响核心矩阵；不得无记录地更换协议。

## Stage 2：低成本筛选

在固定小数据量和单 seed 上筛选：

- `scope`: `readout`, `tail_1`, `no_first`, `full`
- `router`: `shared`, `primary_symmetry`, `learned`, `random_control`，其中 `primary_symmetry` 由 Stage 0 冻结为 `space_group` 或 `crystal_system`
- `update`: 至少 `dense` 与 `elora_clean`

筛选只用于选择进入核心矩阵的 scope 和合理 rank，不用于报告最终显著性。选择规则和阈值必须在查看多 seed 结果前冻结。

## Stage 3：核心矩阵

最低核心矩阵为：

- `P_update`: `dense`, `elora_clean`
- `R_router`: `shared`, `primary_symmetry`, `learned`, `random_control`
- `N_data`: 每个保留类别 `25`, `50`, `100`, `200` 个训练结构；若数据不足，使用预注册的最大可行嵌套层级并记录。
- `seed`: `1`, `2`, `3`
- `S_scope`: 使用 Stage 2 冻结的一个主 scope；如预算允许，再加入一个对照 scope。

每个计划单元最终必须处于以下一种状态：`succeeded`、`failed-reproducible` 或 `skipped-by-preregistered-gate`。不得静默遗漏单元。

`primary_symmetry` 的专家数等于最终保留类别数。`learned` 使用相同的专家数 K，`random_control` 使用相同 K 和相同类别频率进行 parent 级随机分组，以保持容量和每专家样本量可比。

`failed-reproducible` 只表示运行记账完整，不代表科学实验已经完成。所有未被预注册门控跳过的核心主比较必须具有成功运行及足够 seed，才能完成 Goal；环境、代码、OOM 或数据管线失败必须修复或使 Goal 保持阻塞。

## 运行纪律

- 所有方法共享相同 split、嵌套子集、优化预算、早停规则和评估代码。
- 每个 run 保存完整配置、命令、代码 commit、环境、数据/split 哈希、seed、指标曲线、最佳 checkpoint、耗时、峰值显存和 trainable 参数量。
- 失败 run 保留 stderr、最后 checkpoint 和最小复现信息；重试次数与规则统一。
- 训练中不根据验证结果新增有利于某一方法的超参数搜索。
- 所有正式 run 和服务器测试必须由版本化的 Slurm 提交脚本产生；交互式登录节点输出不得作为正式结果。

## 分析与判据

至少报告每个 cell 的均值、标准差、单 seed 结果和配对差值，并给出效果量或 bootstrap 置信区间。主要比较为最终 `primary_symmetry` 专家对同 update/scope/data/seed 下 `shared` 的差值；`random_control` 用于区分“额外参数容量”与“有意义路由”。宏平均、最差类别误差和类别方差均按 Stage 0 冻结的空间群或晶系计算。同时报告精度—参数量、精度—显存和精度—训练时间。

“先受益后饱和”只能在效应随 `N_data` 的曲线和不确定性支持时成立。若专家无提升、只在部分数据量提升或被 random control 解释，也应如实形成完成结论。

具体主要指标、次要指标、正向提升阈值和多重比较策略以 `docs/preexperiment_elora_experts.md` 的预注册标准为准，并在首个正式 run 前冻结。

## 必须交付

- Stage 0 的全部统计与冻结文件。
- `classification_decision.json` 与 `plan_frozen.yaml`。
- `runs.csv` 或等价数据库：一行一个 run，包含状态与全部追踪字段。
- `metrics_long.csv`：长表形式的逐 split、逐指标结果。
- `failure_registry.md`：失败、重试和跳过理由。
- `slurm_jobs.csv`：run ID、commit、Slurm job ID、提交脚本、资源、状态、退出码、stdout/stderr 与输出路径。
- 学习曲线、主效应/交互效应、效率 Pareto 图及类别级误差图。
- `docs/preexperiment_results.md`：数据统计、实现与协议、复现结果、核心矩阵、统计检验、阴性结果、局限性和下一步建议。

## 完成、阻塞与停止条件

当且仅当以下条件全部满足时完成：

- 数据统计和类别筛选完整，所有正式 run 引用同一冻结 manifest 与 split hash。
- 所有预注册单元都有终态，且不存在未解释的缺失结果。
- 所有未被预注册门控跳过的核心主比较均有足够的成功 run 和 seed；技术性 `failed-reproducible` 不得替代主结果。
- 本地单元测试及同一已推送 commit 的服务器 Slurm 单元测试均通过；所有正式结果均可追溯到 Slurm job ID。
- 结果表能从原始 run 记录重建，关键图表可由脚本复现。
- `docs/preexperiment_results.md` 明确回答主问题，并区分数据支持、阴性结果与推测。
- `docs/test.md`、`docs/update.md`、`docs/env.md`、`docs/gpu.md` 以及必要的 `docs/lessons.md` 已按 `docs/AGENTS.md` 更新。

发现“没有提升”不构成阻塞；它是合法结果。只有缺少必要输入、Goal 0 未就绪、数据不满足筛选要求或计算资源无法覆盖冻结后的最低矩阵时，才报告阻塞。完成后立即停止，不自动进入其他数据集、其他 foundation model 或更多性质的 Stage 4。
