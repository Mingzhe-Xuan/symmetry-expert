# Goal 0：ELoRA 对称专家实现与就绪性验证

## 可直接提交给 Codex 的 Goal

> /goal 首先完整阅读并全程严格遵守 docs/AGENTS.md 和经 Guqq 实机核验的 docs/slurm_template.md，再依据 docs/elora_paper_logic.md、docs/preexperiment_elora_experts.md、docs/plan_v2.md 和 ELoRA/paper.pdf，完成 ELoRA 对称专家预实验的可运行准备：实现可配置的 scope、update、router、rank 与专家差分参数 bank；建立强制数据统计工具；补齐单元测试和 CPU/GPU smoke test。Guqq 的 REMOTE_REPO_PATH 固定为 /home/xmz/symmetry-expert，REMOTE_VENV_PATH 固定为 /home/xmz/symmetry-expert/.venv；用户已授权在该仓库下创建虚拟环境和安装依赖，但不得修改系统或全局环境。先核对 origin、commit 和干净状态；代码必须在本地测试、提交并推送，在连接服务器前记录 docs/gpu.md，服务器先 git pull，随后在唯一的 compute 分区通过 Slurm 提交环境安装、服务器单元测试和 smoke test，GPU 请求使用通用 gres gpu:1，禁止在登录节点直接运行或修改文件。Guqq 只有单节点单张 RTX 5090，不得照搬 ELoRA 的 2 节点/20 GPU 示例；Slurm accounting 已禁用，不得依赖 sacct。将 job ID、日志、scontrol 状态和结果同步写入 docs/test.md、docs/update.md、docs/env.md 与 docs/preexperiment_readiness.md。不要下载大型正式数据，不要运行正式实验矩阵。只有本地与服务器 Slurm 验证全部通过且 readiness=ready 时才完成并停止；不得自动进入正式预实验。

## 单一目标

让 ELoRA 仓库具备运行 `S_scope × P_update × R_router × N_data` 预实验的工程能力，并用可重复测试证明实现正确。此 Goal 不回答“专家是否优于直接训练”的科学问题。

## 必读顺序

1. `docs/AGENTS.md`
2. `docs/elora_paper_logic.md`
3. `docs/slurm_template.md`
4. `docs/preexperiment_elora_experts.md`
5. `docs/plan_v2.md`
6. `ELoRA/paper.pdf`
7. ELoRA 仓库中的训练入口、模型定义、checkpoint 与测试代码

`docs/AGENTS.md` 是本 Goal 的强制执行规范，不是背景材料；其服务器权限、Slurm、测试、环境和记录要求均属于验收条件。

## 允许范围

- 修改 ELoRA 仓库内配置、模型、训练、评估、checkpoint 和测试代码。
- 新增小型合成数据或仓库现有样例用于测试。
- 运行单元测试和短 GPU smoke test。
- 在 `/home/xmz/symmetry-expert/.venv` 创建、重建或更新项目虚拟环境，并在其中安装 Goal 0 所需依赖。
- 更新与实现直接相关的文档。

## 禁止范围

- 不执行正式超参数矩阵或多 seed 长训练。
- 不把未指定的数据、checkpoint 或 GPU 当作默认值猜测。
- 不因 smoke test 上出现正向或负向结果而作科学结论。
- 不使用 root 权限，不修改系统 Python、全局 site-packages、系统 CUDA 或 `/home/xmz/symmetry-expert/` 以外的环境目录。

## 虚拟环境授权与验收

- `REMOTE_VENV_PATH` 固定为 `/home/xmz/symmetry-expert/.venv`，不再是待用户填写项。
- 环境创建与依赖安装属于 Goal 0 的明确授权范围，但必须通过版本化 Slurm setup 脚本执行，不能直接在登录节点运行安装命令。
- Codex 可自主选择兼容 RTX 5090、PyTorch、CUDA、MACE、e3nn 与 ELoRA 的版本组合；选择理由、精确版本、安装命令和包源写入 `docs/env.md`。
- wheel、临时下载和构建缓存若需长期保留，必须放在 `/home/xmz/symmetry-expert/.cache/`；大型文件加入 `.gitignore`。不得把依赖安装到仓库外或系统环境。
- 环境验收至少包括：Python 与关键包导入、版本输出、依赖一致性检查、PyTorch 识别 RTX 5090、`sm_120` 支持，以及实际 CUDA tensor/kernel 成功执行。

## 强制同步、Slurm 与测试闭环

每次代码迭代必须完成：

1. 修改前在 `docs/state.md` 记录当前阶段与测试计划。
2. 本地执行相关单元测试；将命令、环境、结果和退出码写入 `docs/test.md`。
3. 本地测试通过后提交并推送代码，同时在 `docs/update.md` 记录 commit 级进展。
4. 连接服务器前在 `docs/gpu.md` 写明连接用途；之后执行 `ssh Guqq`，进入项目后首先 `git pull`。
5. 提交前重新核验 `compute` 分区、`node221` 与 `gpu:1`；使用 `docs/slurm_template.md` 提交服务器单元测试和 CPU/GPU smoke test。即使测试很短，也不得绕过 Slurm 在登录节点直接执行。
6. 在 `docs/test.md` 和 readiness 报告中保存 commit、Slurm 脚本、job ID、分区/设备、环境、stdout/stderr 路径、退出码和测试摘要。
7. 服务器失败后在本地修复并重新走完整闭环；连续失败 3 次时按 `docs/AGENTS.md` 更新 `docs/lessons.md`。

服务器只负责拉取同步与 Slurm 任务提交。大型数据或权重不进入 Git；确需传输时按 `docs/AGENTS.md` 使用 `scp`，并记录来源、目标与哈希。

Guqq 的 `REMOTE_REPO_PATH` 为 `/home/xmz/symmetry-expert`，`REMOTE_VENV_PATH` 为 `/home/xmz/symmetry-expert/.venv`；每次任务前仍须核对 origin、commit 与干净工作树。在 `.venv` 创建且环境验收通过前保持 `readiness: not_ready`。GPU smoke 必须验证 RTX 5090 的 PyTorch CUDA 可用性、`sm_120` 支持和实际 CUDA kernel；只看到 `nvidia-smi` 成功不算通过。

## 必须实现的配置维度

- `update_mode`: `dense`, `elora_clean`, `elora_paper`。
- `scope`: `readout`, `tail_1`, `no_first`, `full`；若模型结构无法逐字对应，给出确定性的层映射表。
- `router`: `shared`, `crystal_system`, `point_group`, `space_group`, `learned`, `random_control`。
- ELoRA 参数：`rank`, `alpha`, `num_experts`, `expert_map`。
- 数据量与随机性：`train_size`, `seed`, `split_manifest`。

## 实现约束

1. 共享参数只在内存和 checkpoint 中保存、加载一次。
2. 专家差异仅保存为差分参数 bank；计算时按样本路由到对应专家。
3. mixed-expert batch 必须可正确前向与反向，且不能复制完整 backbone。
4. `shared`、`random_control` 与各类专家 router 使用相同训练协议，避免额外容量或训练步数混杂。
5. 必须输出 trainable-parameter manifest：参数名、shape、scope、共享/专家归属、是否训练、参数量。
6. checkpoint 必须记录配置、标签映射、数据 manifest 哈希、代码版本和随机种子。
7. 若实现 merge/unmerge，必须测试数值等价与可逆性；否则明确禁用并说明推理路径。

## 强制数据统计工具

在任何正式训练前，统计工具必须能生成：

- `dataset_summary.md`
- `class_counts.csv`
- `removed_classes.csv`
- `split_counts.csv`
- `dataset_manifest.json`
- `classification_decision.json`
- `figures/` 中的类别分布、长尾与交叉分布图

统计至少包括总结构数、去重规则、元素/原子数/性质分布，以及晶系、点群、空间群的逐类样本数和占比。先对空间群应用 `n_g < 100` 删除审计；若删除的去重独立结构总数严格大于 2000，必须回到空间群删除前的合格去重集合，以晶系作为主分类并重新应用 `n_g < 100` 删除规则。否则主分类保持为空间群。决定必须写入 `classification_decision.json`，且发生在 split 之前。统计还应报告 imbalance ratio、熵或有效类别数，以及晶系—点群—空间群交叉表。

## 最低测试清单

1. 零差分初始化与基础模型输出一致。
2. 梯度只进入配置允许的共享参数或当前专家差分。
3. 更新专家 A 不改变专家 B 的差分参数。
4. 共享 backbone 参数对象只有一份。
5. mixed-expert batch 与逐专家拆分计算一致。
6. 标量输出保持不变性；力或张量输出按任务要求保持等变性。
7. checkpoint 保存—恢复后输出、router 映射和参数状态一致。
8. merge/unmerge 数值等价，或未实现时有显式禁用测试。
9. trainable-parameter 与显存统计可重复。
10. 数据清洗前后计数守恒。
11. 所有 `n_g < 100` 类别均被删除，所有保留类别均满足阈值。
12. 同一 parent structure 的构型不会跨 train/val/test 泄漏。
13. 各 split 逐类计数之和与 manifest 一致。
14. 对确定性 symmetry router，旋转、平移和原子置换不改变路由标签。
15. 至少一个最小配置完成 GPU 前向、反向、保存、恢复和评估。
16. 分类门控边界测试覆盖 `N_removed_SG=2000`（保持空间群）和 `N_removed_SG=2001`（切换晶系），并验证晶系筛选从空间群删除前的数据集合重新计算。

## 检查点

- A：仓库与论文逻辑审计完成，输出实现映射与已知偏差。
- B：配置、参数 bank 与 router 前向完成，核心单元测试通过。
- C：数据统计、类别删除、group split 和 manifest 完成。
- D：checkpoint、恢复、参数统计和 GPU smoke 完成。
- E：在服务器通过 Slurm 运行完整测试套件和 smoke test，回传日志并生成最终 readiness 报告。

每个检查点都在 `docs/preexperiment_readiness.md` 记录日期、commit、命令、结果和未解决问题。

## 完成与停止条件

只有同时满足以下条件才可标记完成：

- 上述功能已实现且最低测试清单全部通过。
- 本地测试通过，且同一已推送 commit 在服务器通过 Slurm 完成对应单元测试。
- 至少一个服务器 Slurm smoke run 可从配置、提交脚本和 job ID 复现。
- `docs/test.md`、`docs/update.md`、`docs/env.md`、`docs/gpu.md` 已按 `docs/AGENTS.md` 更新。
- `docs/preexperiment_readiness.md` 含实现映射、本地与服务器测试命令和结果、Slurm job ID 与日志位置、已知限制、Goal 1 所需配置示例，并以 `readiness: ready` 结尾。
- 工作树中的改动范围和未提交文件已清楚列出。

若测试失败，保持 `readiness: not_ready`，记录最小复现，不得将 Goal 标为完成。完成后立即停止，不自动启动 Goal 1。
