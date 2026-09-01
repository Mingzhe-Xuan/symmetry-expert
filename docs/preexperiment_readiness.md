# ELoRA 对称专家预实验就绪性报告

readiness: not_ready

日期：2026-09-01
目标：仅完成 Goal 0 工程能力和验证，不运行 Goal 1 正式矩阵。

## 检查点

| 检查点 | 状态 | 证据 |
|---|---|---|
| A 论文与仓库审计 | complete locally | `paper.pdf` 23 页方法、实验及附录已核对；实现边界见下表 |
| B 配置、bank、router | complete locally | readiness 单元测试与真实 MACE 等变性测试 |
| C 数据统计与 split | complete locally | 合成数据、2000/2001、pre-SG fallback、group leakage 测试 |
| D checkpoint、统计、smoke | complete locally | checkpoint metadata、merge/unmerge、CPU smoke |
| E Guqq Slurm | pending | environment/unit/CPU/GPU job ID 与日志待回填 |

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
10. GPU 15：Guqq GPU Slurm smoke 尚待执行。

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

## 未解决门控

- 最终本地门控已通过：selected 40 passed、training CLI 3 passed、CPU smoke/compile/shell/diff checks 全部退出码 0。
- 代码尚未 commit/push。
- 50-wheel offline wheelhouse 已生成并通过 SHA-256 全量校验，尚未 SCP。
- Guqq setup、unit、CPU/GPU smoke 尚未通过，job ID、`scontrol`、stdout/stderr 和产物尚未回填。

readiness: not_ready
