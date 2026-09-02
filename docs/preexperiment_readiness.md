# ELoRA 对称专家预实验就绪性报告

readiness: ready

日期：2026-09-02
目标：仅完成 Goal 0 工程能力和验证，不运行 Goal 1 正式矩阵。

本地实现与 Guqq 最终验证 commit：`e797570ab0d871227a26f4416b446a0c875c93fb`。

## 检查点

| 检查点 | 状态 | 证据 |
|---|---|---|
| A 论文与仓库审计 | complete | `paper.pdf` 23 页方法、实验及附录已核对；实现边界见下表 |
| B 配置、bank、router | complete | readiness 单元测试与真实 MACE 等变性测试 |
| C 数据统计与 split | complete | 合成数据、2000/2001、pre-SG fallback、group leakage 测试 |
| D checkpoint、统计、smoke | complete | checkpoint metadata、merge/unmerge、exact e797 CPU/GPU smoke；Job 231 三 update mode GPU 更新矩阵 |
| E Guqq Slurm | complete | Job 225 完整 suite 与 Jobs 227/228 双 smoke 全绿 |

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
10. GPU 15：Guqq Job 228 在 RTX 5090 上证明 CC 12.0、`sm_120`、torch CUDA 12.8、实际 CUDA kernel、forward/backward 与 checkpoint restore；补充 Job 231 对 `dense`、`elora_clean`、`elora_paper` 分别证明双 expert 非零梯度与 optimizer 参数更新。

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

- 本地 Python venv 完整套件：`69 passed, 14 skipped, 978 warnings in 351.31s`；Windows 跳过 Linux/CUDA compile 用例。dtype 异常恢复、JIT、CPU smoke、compileall、5 个 SBATCH syntax、fullgraph 静态门控和 diff 检查全部通过。
- Guqq setup Job 221：`COMPLETED`, `ExitCode=0:0`, runtime 00:04:44；`/usr/bin/python3 -m venv` 创建 Python 3.10.12 环境，torch `2.11.0+cu128` / CUDA 12.8、pytest-benchmark 5.2.3、py-cpuinfo 9.0.0、editable MACE 0.3.5、imports 与 `pip check` 全部通过。
- Guqq unit Job 225：exact e797，compute/node221/4 CPU/16 GiB，`COMPLETED`, `ExitCode=0:0`, runtime 00:22:32；完整命令 `python -m pytest ELoRA/tests -q`，结果 `82 passed, 1 skipped, 1201 warnings in 1341.62s`，8 个 benchmark 全部真实执行，stderr 为空。唯一 skip 是未安装可选 `schedulefree` 时该模块的声明式跳过，不影响 Goal 0 要求。
- Guqq CPU Job 227：exact e797，2 CPU/8 GiB，`COMPLETED`, `ExitCode=0:0`, runtime 00:00:04；forward/backward、checkpoint restore、有限 loss、shape `[8,16]` 与 JSON 成功，stderr 为空。
- Guqq GPU Job 228：exact e797，4 CPU/16 GiB/`gres:gpu:1`，`COMPLETED`, `ExitCode=0:0`, runtime 00:00:05；RTX 5090、CC 12.0、`sm_120_supported=true`、torch CUDA 12.8、实际 CUDA kernel、forward/backward、checkpoint restore 与 JSON 成功，stderr 为空。
- Guqq 三模式 GPU Job 231：exact `ff234d67f5d049e34fd6bbb2b23c005359cad4e8`，node221/compute，4 CPU/16 GiB/`gres:gpu:1`，`COMPLETED`, `ExitCode=0:0`, runtime 00:00:17。`dense` 两 expert 的 `expert_delta_bank` 非零梯度计数 `[32,32]`、更新范数约 `[0.008000,0.008000]`；`elora_clean` 与 `elora_paper` 两 expert 的 `lora_B_bank` 均为 `[16,16]`、约 `[0.005657,0.005657]`。三者均在 RTX 5090/CUDA 12.8/CC 12.0 上输出有限、shape `[8,16]`、checkpoint restore 成功；stderr 0 字节。
- Job 231 证据：`/home/xmz/symmetry-expert/slurm-elora-gpu-smoke-231.{out,err}` 与 `ELoRA/artifacts/readiness/gpu-smoke-{dense,elora-clean,elora-paper}.json`；GPU SBATCH SHA-256 `6225e4fe…a0f0d`，smoke Python SHA-256 `e54b2a84…54b31`。
- 服务器原始证据：`/home/xmz/symmetry-expert/slurm-elora-{env-221,unit-225,cpu-smoke-227,gpu-smoke-228}.{out,err}` 与 `ELoRA/artifacts/readiness/{cpu-smoke,gpu-smoke}.json`。SCP 回传后内容、JSON 和 stderr 字节数独立核验通过。
- Slurm script SHA-256：setup `b8d15390…3160d`；unit `5647ce58…fa39`；CPU `9b2ab606…a5d8`；GPU `bdf15209…65f0`。

## 完成审计与最终闭环

- 2026-09-01 逐条复核 Goal 原文后确认：Job 205 的命令只包含 4 个选定测试文件，并非要求的完整 `ELoRA/tests`。
- 代码审计发现 `mace.tools.train.evaluate()` 会把评估前冻结的参数在评估后统一设为可训练，破坏 scope/update 白名单。
- 在修复、完整本地测试和同一已推送提交的 Guqq Slurm 完整测试/CPU/GPU smoke 全部通过并回填证据前，旧 Jobs 204–207 只作为历史证据，不能支持完成结论。
- 审计修复最终覆盖评估梯度白名单、逐类数据统计、冻结 router 刚体变换/置换不变性、旧 foundation state/pickle、TorchScript、完整 Slurm suite、compiler cache/线程隔离、PyTorch 2.11 autograd tracing、fullgraph 图外梯度 leaf 和异常路径 dtype 恢复。
- exact e797 的完整 Guqq 闭环已完成；上述旧 Jobs 204–207 仅保留为历史，不再作为最终完成依据。

## 已知限制

- `schedulefree` 是仓库可选依赖，未列入 Goal 0 canonical requirements；其独立测试模块按源码声明跳过 1 项。所有 Goal 0 强制功能、compile、benchmark、CPU/GPU smoke 均已运行通过。
- setup Job 221 发现旧 wheelhouse manifest 缺文件/不匹配，因而按版本化脚本回退到授权的在线/缓存安装；setup stderr 保留 607 字节 hash 诊断，但 job、`pip check`、版本与 imports 全部成功。unit/CPU/GPU stderr 均为 0 字节。
- Job 231 验证的是核心 `SymmetricContraction` 的 mixed-expert 更新路径与 checkpoint，并非完整数据训练或 Goal 1 科学实验；三种模式的 update policy/scope 另由 readiness 单元测试覆盖。
- 本报告只证明正式预实验的工程就绪性；未运行 Goal 1 数据统计或训练矩阵，也不包含科学性能结论。

readiness: ready
