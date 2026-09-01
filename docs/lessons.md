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
