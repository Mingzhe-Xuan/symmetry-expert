# ELoRA 对称专家预实验就绪性报告

readiness: ready

日期：2026-09-01
目标：仅完成 Goal 0 工程能力和验证，不运行 Goal 1 正式矩阵。

本地实现 commit：`f42866353c7a778bd2f10963f90aa2159a0687e1`。Guqq 最终验证 commit：`dee1e009b352d209a476af83623e14f71a492300`。

## 检查点

| 检查点 | 状态 | 证据 |
|---|---|---|
| A 论文与仓库审计 | complete locally | `paper.pdf` 23 页方法、实验及附录已核对；实现边界见下表 |
| B 配置、bank、router | complete locally | readiness 单元测试与真实 MACE 等变性测试 |
| C 数据统计与 split | complete locally | 合成数据、2000/2001、pre-SG fallback、group leakage 测试 |
| D checkpoint、统计、smoke | complete locally | checkpoint metadata、merge/unmerge、CPU smoke |
| E Guqq Slurm | complete | setup 204、unit 205、CPU smoke 206、GPU smoke 207；均 `COMPLETED`, `ExitCode=0:0` |

## 论文逻辑与实现映射

论文 ELoRA 对每条允许的 SO(3) tensor-product path 独立分解 `ΔW = BA`，把路径消息先降至 rank `R` 再投影到输出，并证明 `W0 + BA` 保持等变性。论文没有 symmetry router 或 expert bank；本实现将相同的路径合法性扩展为 `A[K,...]`、`B[K,...]`，不把完整 backbone 复制 K 份。

| 要求 | 实现 |
|---|---|
| update | `dense`, `elora_clean`, `elora_paper`；paper 模式保留历史 radial/部分 contraction dense 解冻 |
| scope | `readout`=全部 readout/scale-shift；`tail_1`=最后 interaction/product + readout；`no_first`=除首 interaction/product 外 + readout；`full`=全部合格层 + readout |
| router | shared、三类冻结 symmetry label、learned straight-through top-1、parent-frozen random control |
| expert storage | 一份 shared state；eligible symmetric-contraction 的 dense full-rank delta bank 或 path-wise low-rank bank |
| mixed batch | graph id 扩展至 node；按 expert 分组计算 delta，共享 contraction 只计算一次 |
| learned router | 只接受构型级预计算 invariant `router_features`，带 balance loss 与 diagnostics |
| manifest/checkpoint | 参数名/shape/scope/owner/trainable/numel；配置、expert map、split hash、seed、code version |
| merge | 单专家显式 merge/unmerge 且数值可逆；多专家明确禁止 merge，推理保持 unmerged |
| statistics | JSON/JSONL 输入，去重、严格 `<100` 删除、严格 `>2000` fallback、group split、固定 train order/hash、9 张图及全部要求文件 |

Dense shared 模式直接更新 scope 内原参数。Dense multi-expert 为避免 K 份 backbone，只在与 ELoRA 可公平比较的 eligible symmetric-contraction 路径保存 full-rank `ΔW_g`；interaction/readout 仍共享，其中 readout 按统一协议 dense 训练。正式结果必须按此确定性映射报告，不能把它描述为 K 个 full-finetuned MACE。

## 最低测试清单映射

1. 零差分、5. mixed batch、2–4. 梯度隔离/共享唯一性：`test_zero_delta_and_mixed_expert_consistency`、`test_expert_gradient_isolation_and_shared_uniqueness`。
2. scope/update/optimizer 白名单：12 项参数化 policy test。
3. 专家隔离：LoRA 与 dense bank slice gradient tests。
4. 能量/力等变性：真实 MACE routed energy/force test；path scalar/vector test。
5. checkpoint/router/config：metadata round trip 和 smoke restore。
6. merge：单专家可逆、多专家显式 RuntimeError。
7. 参数/optimizer/非零梯度/内存：`parameter_statistics`。
8. 数据 10–13、16：合成统计产物、计数守恒、threshold、split leakage、2000/2001 和 pre-SG fallback tests。
9. router 14：冻结 symmetry/random labels来自 manifest；learned router只读 invariant features，测试旋转/平移/置换不改输入与路由。
10. GPU 15：Guqq Job 207 在 RTX 5090 上证明 CC 12.0、`sm_120`、torch CUDA 12.8、实际 CUDA kernel、forward/backward 与 checkpoint restore。

## 可复现入口

- 数据统计：`python -m mace.readiness.dataset_statistics INPUT.json OUTPUT_DIR --minimum-class-size 100 --fallback-threshold 2000 --seed 123`
- CPU smoke：`python ELoRA/scripts/readiness_smoke.py --device cpu`
- 服务器脚本：`ELoRA/scripts/slurm/{setup_readiness,unit_readiness,cpu_smoke,gpu_smoke}.sbatch`

## Goal 1 配置示例（仅示例，未执行）

```text
--update_mode=elora_clean --scope=tail_1 --router=space_group \
--elora_rank=16 --elora_alpha=16 --num_experts=K \
--expert_map=EXPERT_MAP.json --split_manifest=dataset_manifest.json \
--train_size=100 --seed=123
```

等容量随机对照只把 router 改为 `random_control` 并复用相同 K、split、train_size、seed 和优化预算。learned 路由还必须提供冻结定义的 invariant `router_features`。

## 最终门控结果

- 最终本地门控已通过：selected 40 passed、training CLI 3 passed、CPU smoke/compile/shell/diff checks 全部退出码 0。
- 实现与证据 commit 已推送；`601f45c` 的 post-commit 本地门控为 43 passed，CPU smoke 成功。
- 50-wheel offline wheelhouse 曾在本地通过 SHA-256 全量校验；因 SSH 大文件传输不稳定，最终使用已授权的服务器在线依赖路径，清华 PyPI 主索引 + 官方 PyTorch cu128 extra index。
- Guqq setup Job 204：`COMPLETED`, `ExitCode=0:0`；Python 3.10.12 venv、editable MACE 0.3.5、`pip check`、torch 2.11.0+cu128 / CUDA 12.8 与固定依赖 import 全部通过。
- Guqq unit Job 205：`COMPLETED`, `ExitCode=0:0`；40 passed。
- Guqq CPU smoke Job 206：`COMPLETED`, `ExitCode=0:0`；forward/backward、checkpoint restore 与 JSON 产物成功。
- Guqq GPU smoke Job 207：`COMPLETED`, `ExitCode=0:0`；RTX 5090、CC 12.0、`sm_120_supported=true`、实际 CUDA kernel、forward/backward、checkpoint restore 与 JSON 产物成功。
- stdout/stderr 与 smoke JSON 已经 SCP 回本地核验；unit/CPU/GPU stderr 均为空。Goal 0 readiness 门控全部完成，未启动 Goal 1。

readiness: ready
