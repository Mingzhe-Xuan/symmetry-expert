# ELoRA 论文实验逻辑与本地仓库审计

## 1. 文献与阅读范围

本文档依据本地论文 [ELoRA: Low-Rank Adaptation for Equivariant GNNs](../ELoRA/paper.pdf) 全文及附录整理，并对本地 ELoRA 仓库中的训练、参数冻结和 symmetric contraction 实现进行了对应检查。

论文要解决的问题不是“普通 LoRA 能否减少参数”，而是：普通全局矩阵低秩分解若跨越不相容的不可约表示路径，可能破坏等变性；能否在每条允许的 tensor-product path 内做低秩更新，从而同时保持 \(SO(3)\) 等变性和小样本适配能力。

## 2. 方法逻辑

### 2.1 路径依赖的低秩更新

对 tensor product 中由 \((l_1,l_2,l_3)\) 标记的允许耦合路径，论文将预训练权重写成

\[
W_{l_3l_2l_1}
=W^0_{l_3l_2l_1}
+B_{l_3l_2l_1}A_{l_3l_2l_1}.
\]

其中 \(A\) 将该路径的通道投影到秩 \(R\) 的低维空间，\(B\) 再投影到输出通道。不同角动量路径不混合，因此低秩增量仍满足原 tensor product 的表示约束。论文进一步把 self-interaction 和 residual connection 表示为 fully connected tensor product 的特例，从而给出统一的等变性论证（论文 §4、图 2、附录 C）。

ELoRA 的关键归纳偏置有两个：

1. 下游任务所需的权重变化在每条等变路径内具有较低内在秩；
2. 冻结预训练权重 \(W^0\)，只学习低秩增量，可以降低小数据下的有效容量和过拟合风险。

论文附录 D 用 stable-rank 上界解释第二点：低秩 \(\Delta W\) 可收紧与模型复杂度有关的泛化上界，但 rank 太低会欠拟合，太高又可能过拟合。因此 rank 是偏差—方差旋钮，而不是越大越好。

## 3. 论文的证据链

论文实验不是单一 benchmark，而是按以下逻辑逐层排除替代解释。

### 3.1 先证明预训练微调优于从头训练

- 有机体系使用 MACE-OFF，预训练数据为 SPICE；
- 无机体系使用 MACE-MP，预训练数据为 MPTrj；
- 比较 from-scratch 模型、未微调预训练模型、全参数微调和 ELoRA 微调。

该组对比用于说明收益来自“预训练表示 + 下游适配”，而不是只来自 MACE 架构。

### 3.2 用极小训练集比较 ELoRA 与全参数微调

rMD17 对 10 个分子各只使用 50 个训练构型，评价能量和力 MAE。ELoRA 在表 1 的所有分子上优于全参数微调；论文报告平均能量误差改善 25.5%，力误差改善 23.7%。

这一实验主要检验：当标注极少时，低秩约束能否比全参数更新有更好的泛化。

### 3.3 用温度外推检验 OOD 泛化

- 3BPA：使用 300 K 数据训练，在 300、600、1200 K 以及二面角切片上测试；
- AcAc：使用 300 K 数据训练，在 300 K 和 600 K 测试；
- 每个条件报告三次运行的均值和标准差。

随着测试温度升高，ELoRA 相对全参数微调的优势增大。该实验支持的不是普通 IID 精度，而是低秩更新可能更好地保留预训练模型的外推能力。

### 3.4 用 10 个无机数据集验证跨体系稳定性

论文使用 SSE-PBE、H2O-PD、AgAu-PBE、AlMgCu、Cu、Sn、Ti、V、W 和 HfO2，覆盖金属、水/冰、氧化物和固态电解质。表 4 使用 energy RMSE（meV/atom）与 force RMSE（meV/Å）。

相对全参数微调，论文报告 10 个无机数据集上平均能量改善 12.3%、力改善 14.4%；力预测在 9/10 个数据集上优于所列其他模型。该部分说明 ELoRA 的趋势不局限于单个有机分子。

### 3.5 直接绘制数据量曲线

论文在 rMD17-aspirin 上把训练量从 50 增至 1000，比较 ELoRA 与全参数微调。图 4 的核心结论是：

- 小数据时 ELoRA 优势明显；
- 达到相同误差时，能量与力所需数据分别减少约 42% 和 44%；
- 数据增加后，两种方法的误差差距缩小。

这正是 plan_v2 中 \(P_{\text{update}}\times N_{\text{data}}\) 的直接依据。

### 3.6 rank 扫描解释容量规律

论文在 rMD17-aspirin 和 Cu 上扫描

\[
R\in\{2,4,8,16,32,64,128\}.
\]

验证损失先下降后上升，最终选择 \(R=16\)。此时论文报告 ELoRA 参数约占原模型参数的 22.5% 和 24.0%。该实验说明比较 ELoRA 时必须固定或扫描 rank；只报告单一 rank 无法区分方法收益与容量选择收益。

### 3.7 用其他微调方式做结构消融

附录表 7 比较 MACE 的：

- full-parameter fine-tuning；
- 普通 adapter；
- readout-only；
- ELoRA。

在 Cu 和 Sn 上，ELoRA 优于其余三种方法。普通 adapter 被认为会破坏等变性；readout-only 参数最少，但灵活性不足。该实验与 plan_v2 的 \(S_{\text{scope}}\) 轴直接相关，但论文只给出了 readout 与 full 两端，没有系统扫描 last-\(L\) 和 except-first。

### 3.8 排除学习率和普通正则化解释

附录图 7 分别扫描 ELoRA 与全参数微调的学习率，并对全参数微调扫描 weight decay。最优 ELoRA 仍优于最优全参数设置，而单纯增大 weight decay 没有复现 ELoRA 收益。

这意味着我们的比较不能只共用一个未经调优的学习率；至少应为 Dense 和 ELoRA 各自选择一次学习率，再在所有 router 条件下固定。

## 4. 论文训练设置

论文表 6 给出的 MACE-MP fine-tune 主要设置为：

| 项目 | 设置 |
|---|---:|
| 模型 | ScaleShiftMACE |
| \(r_{\max}\) | 6.0 Å |
| radial basis | 10 |
| channels | 128 |
| \(l_{\max}\) | 2 |
| loss | energy + force |
| energy weight | 1 |
| force weight | 1000 |
| learning rate | 0.005 |
| weight decay | \(10^{-8}\) |
| scheduler patience | 5 |
| EMA decay | 0.995 |
| gradient clipping | 100 |
| ELoRA rank | 16（由 rank 扫描选择） |

论文称全参数与 ELoRA 使用相同微调超参数；附录又提供了学习率敏感性分析。复现实验应先遵循表 6，再进行各更新方式独立的学习率小网格。

## 5. 本地仓库与论文逻辑的对应

### 5.1 已实现的部分

- MACE/MACE-MP foundation model 加载位于 [run_train.py](../ELoRA/mace/cli/run_train.py)；
- ELoRA 增量位于 [symmetric_contraction.py](../ELoRA/mace/modules/symmetric_contraction.py) 的最高阶 symmetric-contraction 权重；
- 初始化采用一个随机因子和一个零因子，所以训练开始时 \(\Delta W=0\)，应与预训练输出一致；
- 训练时通过参数名动态控制 requires-grad；
- 保存时尝试生成 LoRA 合并版本。

### 5.2 正式预实验前必须修正或验证的问题

1. **rank 与 scale 被硬编码。** symmetric_contraction.py 约第 155–159 行把 \(r=16\)、\(\alpha=16\) 写死，无法执行论文的 rank 消融或专家预算匹配。
2. **当前“ELoRA”并非纯低秩更新。** train.py 约第 319 行除 LoRA 参数外，还解冻 radial_embedding 和 symmetric_contractions 中除 weights_max 外的参数。因此论文式配置同时包含低秩更新和一部分 dense 更新。
3. **训练范围不可配置。** 当前逻辑没有统一的 decoder-only、last-\(L\)、except-first、full scope 开关，无法直接形成 \(S_{\text{scope}}\) 轴。
4. **没有专家参数 bank 和 router。** 当前每层只有一组 LoRA 因子，尚不支持共享 \(W_0\) 下按类别选择 \(\Delta W_g\)。
5. **合并路径需要单元测试。** merge_LoRA 删除 LoRA_weight、alpha 和 r，但 forward 仍直接引用这些成员；合并模型可能无法再次前向。
6. **参数统计口径不完整。** CLI 统计了 LoRA、radial embedding 和部分 symmetric contraction 参数，但 optimizer 仍注册更广的参数组；必须分别报告 optimizer 中参数、实际 requires-grad 参数和产生非零梯度的参数。
7. **依赖环境不匹配。** 仓库 setup.cfg 固定 e3nn 0.4.4；当前工作环境为 Python 3.12、CPU PyTorch 2.11 和 e3nn 0.5.6，不能把当前环境中的失败或成功当作论文复现结果。

## 6. 对本课题可继承与不可直接继承的部分

### 可继承

- 同一预训练 checkpoint 下比较 Dense 与 ELoRA；
- 以小数据学习曲线作为核心证据；
- rank 扫描和学习率控制；
- zero-shot、from-scratch、readout-only 与 full fine-tune 基线；
- energy/force 同时报错，并做 OOD 测试；
- 用 stable rank、可训练参数和训练成本解释结果。

### 不可直接继承

- 原论文没有晶系、点群、空间群或 learned router；
- 10 个无机数据集是材料体系/工况 benchmark，不是平衡的多空间群 benchmark；
- 多专家会增加总差分参数，必须加入随机等容量专家和预算匹配实验；
- readout-only × ELoRA 在当前实现中没有可低秩化的 symmetric-contraction 路径，属于退化/不可比单元；
- 论文主要证明单一 adapter 的 PEFT 效果，不能直接推出“对称专家优于共享 ELoRA”。

## 7. 对预实验的直接结论

预实验应分两步：

1. 先用论文原任务复现 Shared-ELoRA 相对 Shared-Dense 的基本趋势，确认代码和环境可信；
2. 再在平衡的多空间群数据上加入 router，并在同一 scope、update、data split 和优化预算下比较 Shared 与 Expert。

只有第二步中真实对称路由优于 Shared 和等容量随机路由，才能把增益归因于对称专家；仅仅优于单一全参数模型还不足够。

## 参考

Wang, H. et al. ELoRA: Low-Rank Adaptation for Equivariant GNNs. ICML 2025, PMLR 267:63113–63135. https://proceedings.mlr.press/v267/wang25al.html

