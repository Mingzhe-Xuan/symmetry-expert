# Goal 0 测试记录

## 2026-09-01：`docs/requirements.txt`（计划，等待 commit 后执行）

- 检查 canonical requirements 与旧兼容入口是否可由 pip 正确解析，并确认锁定的 19 个直接依赖未变化。
- `bash -n ELoRA/scripts/slurm/setup_readiness.sbatch`：验证修改后的 Slurm setup shell syntax。
- `python -m pytest ELoRA/tests/test_elora_readiness.py ELoRA/tests/test_modules.py ELoRA/tests/test_models.py ELoRA/tests/test_data.py -q`：对应 readiness 回归套件。

## 2026-09-01 本地相关测试

环境：Windows，Python 3.12.7，PyTorch 2.13.0 CPU，e3nn 0.4.4，ASE 3.22.1；cache 均在仓库内。

验证代码 commit：`f42866353c7a778bd2f10963f90aa2159a0687e1`。

| 命令 | 结果 | 退出码 |
|---|---|---:|
| `python -m pytest tests/test_elora_readiness.py -q` | 23 passed（修正统一 readout policy 后） | 0 |
| `python -m pytest tests/test_elora_readiness.py::test_mace_expert_energy_invariance_and_force_equivariance -q` | 1 passed；真实 MACE energy invariant / force equivariant | 0 |
| `python -m pytest tests/test_elora_readiness.py tests/test_modules.py tests/test_models.py tests/test_data.py -q` | 36 passed，3 个旧 policy assertion 失败；根因确认后已修复 | 1 |
| offline full suite（排除 foundation 文件） | 51 passed, 14 skipped, 5 failed；其中 2 个仍是下载型 foundation 测试，3 个为上游 2023/2024 full-MACE 精确能量快照 | 1 |
| `python scripts/readiness_smoke.py --device cpu` | forward/backward/optimizer/checkpoint restore/eval 成功，shape `[8,16]` | 0 |
| `bash -n scripts/slurm/*.sbatch` | 4 个脚本语法通过 | 0 |

离线 full suite 中的三个训练 CLI 均成功退出并生成可加载模型。精确能量快照早于本分支原有的 ELoRA 梯度白名单，不能作为本分支 oracle；测试已改为验证有限输出、孤立原子参考、数量和非退化分布。下载型 `test_mace_off` 与 foundation training 不属于无网络 Goal 0 单元闭环，服务器也不下载模型。

## 待执行的最终本地门控

- `python -m pytest tests/test_elora_readiness.py tests/test_modules.py tests/test_models.py tests/test_data.py -q`：40 passed，退出码 0；最终 post-fix run 64.85 s，覆盖 learned-router 的真实 MACE backward。
- `python -m pytest tests/test_run_train.py::test_run_train tests/test_run_train.py::test_run_train_missing_data tests/test_run_train.py::test_run_train_no_stress -q`：3 passed，退出码 0，85.49 s。
- `python scripts/readiness_smoke.py --device cpu`：checkpoint restore true，loss 1.110556960105896，output `[8,16]`，退出码 0。
- `python -m compileall -q mace scripts/readiness_smoke.py`、四个 sbatch 的 `bash -n`、`git diff --check`：全部退出码 0。

## Guqq Slurm（待提交）

| run | script | job ID | commit | result |
|---|---|---|---|---|
| environment | `ELoRA/scripts/slurm/setup_readiness.sbatch` | pending | pending | pending |
| unit | `ELoRA/scripts/slurm/unit_readiness.sbatch` | pending | pending | pending |
| CPU smoke | `ELoRA/scripts/slurm/cpu_smoke.sbatch` | pending | pending | pending |
| GPU smoke | `ELoRA/scripts/slurm/gpu_smoke.sbatch` | pending | pending | pending |

## 2026-09-01 post-commit 验证

- 已推送 commit：`601f45cd040ba713e6212f4f8d1443b76cd40542`。
- 命令：selected readiness/modules/models/data suite 与三个 training CLI cases 合并运行。
- 结果：43 passed，432 warnings，92.32 s，退出码 0。
- 随后运行 CPU smoke：checkpoint restore true，loss 1.110556960105896，output `[8,16]`，退出码 0。
- 该顺序遵循更新后的 `docs/AGENTS.md`：先 commit，再执行并记录对应单元测试。
