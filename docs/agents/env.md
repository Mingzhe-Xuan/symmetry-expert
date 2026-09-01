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
- Job 204：Python venv 完整重建成功；torch `2.11.0+cu128` / CUDA 12.8、全部固定版本 import、editable MACE/ELoRA 与 `pip check` 均通过。环境安装门控完成。
- Jobs 205–207：同一 venv 的 unit、CPU smoke、RTX 5090 GPU smoke 全部成功；GPU 证明 CC 12.0、sm_120 与实际 CUDA kernel。

## 2026-09-01 本地审计环境

- 路径：`.cache/local-readiness-venv`，由 `python -m venv --system-site-packages` 创建，仅用于 Windows 本地回归，验证后删除。
- Python 3.12.7；torch 2.11.0+cpu；e3nn 0.4.4；ASE 3.22.1；NumPy 1.26.4；SciPy 1.15.3；h5py 3.14.0。
- 新增代码没有引入新依赖；Guqq 的 canonical 依赖仍完整固定在 `docs/requirements.txt`，无需改动该清单。
- 本地 venv 继承宿主 `site-packages`，因此 `pip check` 会报告 fairchem/manim/torchaudio 等与 Goal 0 无关的宿主包冲突；该环境只支持本地测试通过的证据，不作为可复现安装门控。正式安装门控是 Python 创建的 Guqq `.venv`，Job 204 已在隔离环境中证明 canonical requirements、editable ELoRA、imports 与 `pip check` 全部通过。

## 2026-09-02 benchmark 插件增量

- Job 219 证明仓库完整 suite 需要 `pytest-benchmark` 提供 `benchmark` fixture；canonical `docs/requirements.txt` 新增 `pytest-benchmark==5.2.3` 与其固定传递依赖 `py-cpuinfo==9.0.0`。
- 离线 wheel：`pytest_benchmark-5.2.3-py3-none-any.whl`，45,255 字节，SHA-256 `bc839726ad20e99aaa0d11a127445457b4219bdb9e80a1afc4b51da7f96b0803`；`py_cpuinfo-9.0.0-py3-none-any.whl`，22,335 字节，SHA-256 `859625bc251f64e21f077d099d4162689c762b5d6a4c3c97553d56241c9674d5`。
- setup offline gate 会单独校验这两个新增 wheel；Guqq Python venv 重建与 `pip check` 尚待 exact commit 的 setup Slurm 验收。
