# Goal 0 环境记录

## 本地验证环境

- 日期：2026-09-01
- 路径：`ELoRA/.venv`（仓库内、未进入 Git）
- Python：3.12.7
- PyTorch：2.13.0 CPU
- e3nn：0.4.4
- NumPy：1.26.4
- ASE：3.22.1
- 安装方式：workspace-local uv cache，editable `ELoRA`；未修改系统或全局环境。
- 兼容处理：PyTorch 2.6+ 默认 weights-only loading 与 e3nn 0.4.4 packaged constants 的冲突由 `mace/__init__.py` 中仅允许 Python `slice` 的窄 allowlist 处理；模型对象读取显式使用 `weights_only=False`。

## Guqq 预注册环境

- 固定路径：`/home/xmz/symmetry-expert/.venv`
- Python：系统 `/usr/bin/python3` 3.10.12，仅用于建立项目 venv。
- PyTorch：2.11.0 CUDA 12.8 wheel。
- e3nn：0.4.4；ASE 3.22.1；其余精确版本见 `ELoRA/requirements-readiness.txt`。
- 安装脚本：`ELoRA/scripts/slurm/setup_readiness.sbatch`。
- 安装源：本地生成并经 SCP 传输的 `/home/xmz/symmetry-expert/.cache/wheelhouse`；Slurm 内使用 `pip --no-index`，安装前执行 `sha256sum --check SHA256SUMS`。
- 本地 wheelhouse：50 个 wheel，连同 manifest 共 936,197,999 bytes（892.83 MiB）；`SHA256SUMS` 全部 50 项验证通过。`python-hostlist` 1.23.0 是 Python 3.10 发布但仅有 sdist，已在本地构建为纯 Python universal wheel。

选择理由：RTX 5090 是 compute capability 12.0。PyTorch 2.7 首次加入 Blackwell/CUDA 12.8 支持；2.11 仍提供官方 cu128 wheel。Guqq 驱动 570.211.01 不满足 PyTorch 2.12 默认 CUDA 13.0 所要求的 580.65.06，因此锁定 2.11/cu128。参考：[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)、[PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch 2.12 release](https://pytorch.org/blog/pytorch-2-12-release-blog/)、[NVIDIA CUDA platform](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)。

## Guqq 验收门控

wheelhouse 已在本地解析和校验，但 Guqq 环境尚未安装验证，状态为 `not_ready`。必须由 setup Slurm job 证明 import、精确版本和 `pip check` 成功；随后 GPU smoke 必须证明设备名、capability 12.0、`sm_120` 架构列表和实际 CUDA kernel 均成功。
