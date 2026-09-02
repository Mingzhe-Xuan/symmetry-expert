# 进度更新

## 2026-09-01：采用扩展服务器权限和新记录路径

- 最新规范允许在登录节点执行轻量环境管理、依赖下载、安装和诊断，但明显计算负载仍通过 Slurm。
- 记录路径迁移到 `docs/agents/`；旧 `docs/*.md` 保留为历史记录，不删除或覆盖。
- 用户最新明确要求继续使用 Python `venv`，并已手动尝试构建；下一单元先验证现有环境，不预设需要重建。
- Guqq 环境核验前 GitHub pull 连续 3 次 TLS 超时；已按规则记录经验。远端 refs/tree 与最新已推送 commit 完全一致，后续仅进行只读 venv 核验，不提交新作业。
- 第 4 次 pull 成功。手动 venv 的固定依赖与 `pip check` 通过，但 torch 为 cu130，驱动不兼容且 CUDA 不可用；canonical requirements 改为清华主索引 + 官方 cu128 extra index，等待 commit 前验证。
- cu128 索引修复的 commit 前验证完成：双索引静态断言、目标 wheelhouse 解析、SBATCH syntax 均通过，readiness 回归 40 passed。
- setup Job 203 已证明 cu128 双索引修复有效；最后因 editable install 的离线 build isolation 找不到 setuptools 失败。计划增加 `--no-build-isolation` 后按新规范 commit 前验证。
- editable 修复的 commit 前验证全部通过：SBATCH syntax、精确静态断言、真实 editable dry-run 和 40 项 readiness 回归。
- setup Job 204 成功：Python venv、torch cu128/CUDA 12.8、editable ELoRA、`pip check` 和固定版本 import 全部通过。
- Jobs 205/206/207 全部 `COMPLETED`, `ExitCode=0:0`：unit 40 passed、CPU smoke 成功、RTX 5090 GPU smoke 证明 CC 12.0/sm_120/CUDA kernel。证据已 SCP 核验，Goal 0 readiness 更新为 `ready`。
- 最终文档检查通过后清理本任务本地临时文件约 1.64 GiB；保留来源不明的 `.cache/uv`，服务器原始作业证据未删除。
- Goal 0 完成审计发现 Job 205 只跑选定测试文件，且 `evaluate()` 会破坏 requires-grad 白名单；先前 `ready` 结论撤回，按预注册测试修复后重走本地提交与 Guqq Slurm 闭环。
- 完成审计修复及兼容性加固：评估精确恢复梯度白名单、旧 foundation state/pickle 与 TorchScript 兼容、数据逐类统计补全、冻结路由刚体变换/置换测试、跨平台测试入口和完整 Slurm unit 命令均已实现；本地完整套件 `67 passed, 14 skipped`，CPU smoke 与静态检查通过，进入提交及 Guqq 完整 Slurm 复验。
- Guqq Jobs 209/210 CPU 与 RTX 5090 GPU smoke 成功；完整 unit Job 208 保持全套测试但在 28:33 收到 Slurm SIGTERM，未出现 pytest 断言失败。readiness 保持 not_ready，仅扩大 unit wall time 后重投。
- 60 分钟 Job 211 在 collection 阶段确认 `.cache/mace/46jrkm3v` 是不完整 zip；将按任务 cache 权限精确修复该 foundation download，验证完整性后重投，不修改测试标准。
- 损坏 foundation cache 已通过本地完整文件 SCP、远端 size/SHA-256/zip 校验和原子改名修复；保留完整 suite 与 60 分钟时限，重投 exact af5 unit。
- 补齐 MACE-MP large 与三个 MACE-OFF cache：不稳定连接下用 SFTP reput 断点续传，四文件 size/SHA-256/zip 全部核验并原子启用；完整 unit 将不再依赖运行时下载。
- Job 216 排除 foundation 下载后在 CPU TorchInductor 生成 kernel 的 forces 二阶反向 segfault；转为版本化 Slurm 两 profile 定向诊断，保持真实 compile 测试语义。
- Jobs 217/218 完成对照：独立 compiler cache 仍 abort，而独立 cache + 单线程的 fp32/fp64 真实 compile 全绿；正式 unit 将采用后者并保持完整测试范围。
- Job 219 证明正式单线程配置消除 compiler crash；完整 suite 进一步发现 canonical venv 缺 benchmark plugin，以及 PyTorch 2.11 需要显式 autograd tracing。readiness 保持 not_ready，进入依赖与 upstream-compatible 修复。
- Job 221 环境全绿；Job 222 将剩余失败收敛到 fullgraph 图内梯度突变和异常路径 dtype 泄漏。采用 upstream 的图外 positions leaf 约定并给 dtype context 加 finally，本地完整 suite 69 passed。
- exact e797 最终闭环全绿：Job 225 完整 suite 82 passed、1 skipped，Jobs 227/228 CPU/GPU smoke 均完成且 stderr 为空；进入最终 readiness 文档与临时证据清理。
- 最终 10 文件证据回传与独立断言通过；readiness 报告已回填 exact e797、Jobs 221/225/227/228、脚本 hash、日志路径和已知限制，准备最终文档提交。
- 2026-09-02：补充 dense / elora_paper 更新验证：readiness smoke 现对三个 update mode 逐 expert 断言有限 forward、非零梯度、optimizer 参数变化和 checkpoint restore；本地 Python venv 三模式 smoke、28 项相关回归和静态门控通过，等待提交后 Guqq GPU 复验。
- 2026-09-02：业务提交 `e0110fb` 已推送并通过提交后三模式 CPU smoke；首次 Guqq 同步因 GitHub TLS -110 在 pull 阶段停止，尚无 GPU job，进入已记录的短重试。
- 2026-09-02：第二次 Guqq GitHub pull 短重试仍未返回输出/job ID；按既有成功经验转为 exact Git bundle 传输与 bundle fast-forward pull，GPU 作业仍未提交。
