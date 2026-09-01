# 进度更新

## 2026-09-01：setup Job 201 IPv6 连接修复

- Job 201 正确进入在线 fallback，但 pip 在不可达 IPv6 socket 上持续 `SYN-SENT`；PyPI 的工作路径由登录节点 HTTP 200 检查确认。
- 取消 Job 201，将 pip timeout 从 120 秒降为 10 秒并保留 20 次重试，使 urllib3 快速尝试 IPv4；经验与代码注释已同步。
- 提交后计划运行 SBATCH syntax、requirements 目标解析和 readiness 回归，再同步并提交替代 job。

## 2026-09-01：服务器依赖下载授权

- 用户更新 `docs/AGENTS.md`，明确允许服务器联网下载必要 libraries/packages/wheels。
- setup Slurm job 调整为“完整且哈希通过的 wheelhouse 优先，否则在线下载”，pip cache 固定在仓库忽略的 `.cache/pip`；登录节点仍不下载或安装。
- 策略 commit：`4dbccde`。提交后 requirements 目标平台离线解析、SBATCH shell syntax 和 40 项 readiness 回归均通过。

## 2026-09-01：统一环境依赖入口

- 按用户要求新增 `docs/requirements.txt`，固定 Guqq 的 Python 3.10.12 / Linux x86_64 环境直接依赖，并声明官方 PyTorch CUDA 12.8 wheel 索引。
- Slurm setup 改为直接读取该清单；旧 `ELoRA/requirements-readiness.txt` 保留为兼容引用，避免两份版本锁定漂移。
- 环境清单 commit：`c906ec5`。提交后目标平台 pip 离线解析、SBATCH shell syntax 均通过；readiness 回归在工作区临时目录下 40 passed。

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

## 2026-09-01：Guqq 大 wheel 分片传输恢复

- Guqq 已 fast-forward 到 `098db4a20def32bdbebfa70e15924ecf61413e13`；`compute/node221/gpu:1` 状态复核通过。
- 普通 SCP 和 resumable SFTP 在传输 820 MB torch wheel 时累计中断 3 次；已按规范停止同策略重试并把经验写入 `docs/lessons.md`。
- 新策略把 torch wheel 切为 13 个不超过 64 MiB 的分片；分片 manifest 全部通过，重组 SHA-256 与原 wheel 完全相同。
- setup Slurm job 将在 compute 节点验证分片、重组、原子替换并再次验证完整 wheelhouse manifest。

## 2026-09-01：扩展 Introduction 的课题边界

- 将总体动机从单一原子势扩展为等变材料模型的结构域参数高效适配。
- 保留能量—力作为第一阶段 benchmark，强调其守恒性与等变性验证价值。
- 新增标量、张量和频率依赖性质的分层扩展路线，并区分各自的输出表示、池化、损失与物理约束。
- 补充 CGCNN、MatTen、晶体张量等变网络、EATGNN 和 TSENN 等相关工作。
- 校验章节编号、引用编号与 Markdown 格式。
