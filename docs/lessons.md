# 经验与故障处理

## Guqq 网络连接恢复

当 Guqq 无法访问 GitHub 或其他必要网络资源时：

1. 按 `docs/AGENTS.md` 先在 `docs/gpu.md` 记录本次连接用途。
2. 连接服务器：

   ```bash
   ssh Guqq
   ```

3. 在 Guqq 上运行网络处理脚本：

   ```bash
   bash ~/symmetry-expert/net.sh
   ```

   也可以从本地以远程命令形式执行：

   ```bash
   ssh Guqq 'bash ~/symmetry-expert/net.sh'
   ```

4. 脚本启动后等待一段时间，不要立即重复启动多个实例。随后用原本失败的最小只读命令复查，例如 `git ls-remote` 或在干净工作树中执行 `git pull --ff-only`。
5. 将脚本执行时间、可见输出、等待时长、复查命令和结果记录到 `docs/gpu.md` 或相应任务记录中。

注意：`Guqq:~/symmetry-expert/net.sh` 是“服务器及路径”的描述，不是本地 `bash` 可以直接执行的文件路径；必须先通过 SSH 进入 Guqq，或使用上述 `ssh Guqq '...'` 形式。不要在网络尚未恢复时并发反复执行下载、clone 或 pull。

## 训练失败原因

1. ELoRA 的训练依赖于修改后的 e3nn 库。

```sh
pip install git+https://github.com/hyjwpk/ELoRA.git@main
```

## Guqq 大文件 SCP/SFTP 连续断线

2026-09-01 向 Guqq 传输约 820 MB 的 PyTorch wheel 时，普通 `scp -r` 一次、带 keepalive 的 SFTP `reput` 两次均被远端关闭连接；每次连接只能持续传输约 100–170 MB。重复从头上传既慢又无法提高成功概率。

处理原则：

1. 首次失败后先按文件名和 byte size 审计远端部分结果，不覆盖已经完整的文件。
2. 单次偶发中断可用 SFTP `reput` 从远端 byte offset 恢复。
3. 连续 3 次中断后停止重试单个大流，改在本地把大文件切为较小分片并为分片生成独立 SHA-256 manifest。
4. 分片只通过 SCP 传到仓库 `.cache/`；由版本化 Slurm setup job 在 compute 节点校验分片、重组到临时文件、原子替换，并再次用原始 wheelhouse manifest 验证完整 wheel。
5. 不在登录节点执行重组、哈希或安装；不删除用户文件，也不依赖不稳定连接维持到整个大文件结束。

## Guqq pip 优先 IPv6 导致连接停滞

2026-09-01，setup Job 201 在线升级 pip 时超过 6 分钟无下载进展。登录节点只读检查确认 PyPI 返回 HTTP 200，但 Slurm 中 pip 进程的 socket 一直处于 IPv6 `SYN-SENT`；集群的 IPv4 路径正常，IPv6 路由不可达。

处理原则：保留较多 pip retries 处理短暂抖动，但不要把 connect/read timeout 设得过长。Guqq 在线安装固定使用 `--timeout 10 --retries 20`，使 urllib3 快速跳过不可达 IPv6 地址并尝试 IPv4。出现同样症状时先检查 socket 状态和最小 HTTP HEAD，不要让每个重试等待数分钟。Job 201 已取消，修正后用新 commit 重新提交。
