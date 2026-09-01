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
