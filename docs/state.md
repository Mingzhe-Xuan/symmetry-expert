# 当前任务：Goal 0 ELoRA 对称专家实现与就绪性验证

日期：2026-09-01

当前阶段：检查点 D（本地实现与验证）收尾；检查点 E（Guqq Slurm 验证）尚未开始。

## 实现计划

- [x] 完整核对 Goal 0、`docs/AGENTS.md`、论文逻辑、Slurm 规范、预实验方案和本地论文。
- [x] 审计训练入口、模型、symmetric contraction、checkpoint、数据与现有测试，形成需求到代码/测试的映射。
- [x] 实现统一的 update/scope/router/rank/alpha/expert 配置、共享骨干上的专家差分参数 bank、路由和训练参数策略。
- [x] 实现训练参数清单、checkpoint 元数据与显式 merge/unmerge 策略。
- [x] 实现强制数据统计、分类门控、group split 和 manifest 工具。
- [x] 补齐 Goal 0 最低测试清单、CPU smoke 和 GPU smoke/Slurm 脚本。
- [x] 运行并记录 commit 后本地测试，提交并推送验证 commit。
- [ ] 在 Guqq 上只通过 Slurm 完成环境、服务器测试和 RTX 5090 GPU smoke，回传并记录证据。
- [ ] 逐项审计验收材料；全部通过后才把 `docs/preexperiment_readiness.md` 标记为 `readiness: ready`。

## 预注册本地测试

- 配置枚举、scope/update 梯度白名单与 trainable-parameter manifest。
- 零差分输出、专家隔离、共享参数唯一性、mixed-expert batch 一致性。
- 标量不变性以及力/张量等变性；确定性 symmetry router 对旋转、平移和原子置换不变。
- checkpoint 保存/恢复；merge/unmerge 数值等价与可逆，或显式禁用行为。
- 参数、optimizer、非零梯度和内存统计可复现。
- 数据清洗计数守恒、`n_g < 100` 删除、2000/2001 分类边界、parent group split、防泄漏、split/manifest 计数一致。
- 小型合成数据统计产物完整性与 CPU 端到端 smoke。

## 远端测试门控

远端连接前必须先完成本地测试、记录 `docs/test.md`、提交并推送、更新 `docs/update.md` 与 `docs/gpu.md`。远端仓库固定为 `/home/xmz/symmetry-expert`，虚拟环境固定为 `/home/xmz/symmetry-expert/.venv`；所有安装和测试均通过 `compute` 分区 Slurm 作业执行，不在登录节点直接计算或修改文件。
