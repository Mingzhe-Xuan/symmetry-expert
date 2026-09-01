# 经验与故障处理

## Guqq 在线包安装的 IPv6 停滞

- Job 201 的 pip socket 长时间处于 IPv6 `SYN-SENT`，但登录节点 PyPI HTTP 检查成功，说明可用路径为 IPv4。
- 服务器下载工具应优先显式使用 IPv4；依赖管理器需要短网络超时和可恢复缓存，避免单个不可达地址阻塞数分钟。
- 后续在线安装保留短 timeout 与可恢复 cache；当前最新规范要求继续使用 Python `venv`。旧 `docs/lessons.md` 保留更早的 SCP/SFTP 和环境兼容经验。

## Guqq GitHub pull 连续 TLS 超时

- 2026-09-01 环境核验前连续 3 次 `git pull --ff-only` 失败：一次明确 `GnuTLS recv error (-110)`，两次长时间无输出后超时/中断；HTTP/1.1 未消除问题。
- 规范要求连接后先执行 pull，因此每次都先尝试同步。只有在服务器 `HEAD`、现有 `origin/main` 和本地已推送 commit 完全相同，且 `git diff --quiet HEAD origin/main` 通过时，才允许继续纯只读环境诊断；任何源码修改、作业提交仍等待成功 pull 或新的已验证同步。
- 连续失败后停止长时间重复等待，后续诊断连接使用短 timeout，并保留 refs/tree 一致性证据。

## PyTorch 镜像不能替代 CUDA 专用索引

- `torch==2.11.0` 配合普通 PyPI/镜像会在 Linux 选中 `2.11.0+cu130`；Guqq 驱动 570.211.01 无法加载该 build，表现为 `pip check` 正常但 `torch.cuda.is_available()` 为 false。
- 清华镜像可作为通用包主索引，但 RTX 5090 环境必须同时保留官方 `https://download.pytorch.org/whl/cu128` extra index，并在集成检查中断言 torch local version、`torch.version.cuda` 和实际 CUDA kernel，不能只依赖包版本号或 `pip check`。

## 相邻测试调用点的机械替换不能替代逐项核对

- foundation CLI 回归需要放宽的只是预训练模型保留非零原子能量这一条断言；同文件另有三个非 foundation 调用必须继续要求孤立原子能量为零。
- 两次补丁因上下文过宽而落到相邻调用，造成同一失败重复。此类窄语义修改应在编辑后立即用带上下文的 `rg -n -C` 列出全部调用点，逐项核对参数，再运行定向测试；不能只确认 helper 定义已变化。

## 完整套件集成失败必须按层拆解

- Jobs 208/211/216 连续三次未通过，但分别属于 Slurm 时限、损坏/缺失外部模型 cache、CPU Inductor 原生崩溃；不能把它们统称为“测试慢”或通过扩大时限掩盖。
- foundation 下载函数会直接写正式 cache 名；连接中断会留下被下次误认为有效的半文件。服务器慢网下应使用本地已验证文件、独立 staging、断点续传、size/SHA-256/zip 三重校验和同文件系统原子改名。
- 原生 compiler 崩溃先以精确 pytest node id 在 Slurm 中复现，并隔离每-job 编译 cache、线程配置；只有保留原 backend、dtype、forward/forces backward 与数值断言的配置通过，才能进入完整套件，不能用 skip/xfail/eager 规避。
