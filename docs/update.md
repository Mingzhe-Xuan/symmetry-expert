# 进度更新

## 2026-09-01：Goal 0 ELoRA 对称专家本地实现

- 实现 commit：`f42866353c7a778bd2f10963f90aa2159a0687e1`。
- 完成可配置 update/scope/router/rank/alpha/K、共享 backbone 上的专家 delta bank、mixed-expert forward/backward 和 learned top-1 router。
- 将逐 batch 字符串式 `requires_grad` 改为 optimizer 构造前一次性白名单，并增加参数 manifest、统计和 checkpoint readiness metadata。
- 完成强制数据统计、严格分类门控、parent group split、固定 train order/hash 和图表产物。
- 新增 Goal 0 单元测试、真实 MACE energy/force 等变性测试、CPU/GPU smoke 和四个 Guqq Slurm 脚本。
- 实现已 commit；文档证据 follow-up 与 push、Guqq Slurm 尚待完成，`readiness` 保持 `not_ready`。

## 2026-09-01：Goal 0 commit 后验证与推送

- `f42866353c7a778bd2f10963f90aa2159a0687e1` 与证据 commit `601f45cd040ba713e6212f4f8d1443b76cd40542` 已推送到 `origin/main`。
- 按更新后的开发规范，在 commit 后运行对应本地测试：43 passed；CPU smoke 成功。
- 下一阶段为固定最终推送 commit 的 Guqq wheelhouse SCP 与四个 Slurm jobs。

## 2026-09-01：扩展 Introduction 的课题边界

- 将总体动机从单一原子势扩展为等变材料模型的结构域参数高效适配。
- 保留能量—力作为第一阶段 benchmark，强调其守恒性与等变性验证价值。
- 新增标量、张量和频率依赖性质的分层扩展路线，并区分各自的输出表示、池化、损失与物理约束。
- 补充 CGCNN、MatTen、晶体张量等变网络、EATGNN 和 TSENN 等相关工作。
- 校验章节编号、引用编号与 Markdown 格式。
