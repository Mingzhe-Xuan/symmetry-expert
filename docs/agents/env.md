# 环境记录

## Guqq 目标环境

- 项目：`/home/xmz/symmetry-expert`。
- 虚拟环境：`/home/xmz/symmetry-expert/.venv`，由 Python `venv` 创建。
- Python：系统 `/usr/bin/python3` 3.10.12，作为 venv interpreter。
- 目标依赖：`docs/requirements.txt`；PyTorch 2.11.0+cu128、e3nn 0.4.4、NumPy 1.26.4、ASE 3.22.1，其余版本见 canonical 文件。
- 安装工具与来源：pip 26.2.1。清华 PyPI mirror 作为主索引，官方 PyTorch cu128 index 必须作为额外索引；否则 torch 2.11 会选中与当前驱动不兼容的 cu130 build。
- 缓存：任务约定的 `/home/xmz/symmetry-expert/.cache/`；完整 wheelhouse 可优先，不完整时允许联网下载。
- 手动构建检查：Python 3.10.12；19 个直接依赖版本匹配；`pip check` 通过；但 torch `2.11.0+cu130` / CUDA 13.0 导致 `cuda_available=False`，因此环境不合格，等待 setup Slurm job 重建为 cu128。
- Job 203：固定依赖阶段已安装 torch `2.11.0+cu128` / CUDA 12.8；editable ELoRA build isolation 失败，故该 venv 尚未完成最终 `pip check`/import 验证。
