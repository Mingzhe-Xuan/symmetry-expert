# 面向有限数据晶体张量预测的对称性条件化等变微调

## 背景与问题定义

周期晶体由晶格、原子种类和胞内原子位置共同确定；组分与空间群并不能唯一确定结构，因为同一组分和空间群仍可具有不同的晶格参数、Wyckoff 占位、自由位置参数和多晶型 [1,2]。空间群描述把具体晶体映回自身的等距操作，Wyckoff position 则描述给定空间群下一组对称等价的位置轨道及其 site symmetry。

Neumann 原理要求宏观物性张量在晶体点群的全部操作下保持不变，即物性张量的对称群包含晶体点群 [3]。另一方面，E(3)、O(3) 或 SO(3) 等变描述整体改变坐标系或晶体姿态时，标量、向量和高阶张量如何变换 [4,5]。两者不是互相替代的对称性：前者是具体结构的内禀稳定子群，后者是模型对任意整体欧氏变换的协变规律。

设严格等变模型满足

\[
f(gx)=\rho(g)f(x),\qquad g\in O(3).
\]

若 \(h\) 是晶体 \(x\) 的内禀对称操作，并且周期边界、晶格变换及等价原子置换处理严格，使 \(hx=x\)，则

\[
f(x)=f(hx)=\rho(h)f(x).
\]

因此，理想的严格周期等变模型已经隐式满足晶体的 Neumann 约束 [6]。同样，在完整、无噪声结构给定时，点群、空间群和 Wyckoff 信息均可由结构确定，原则上不增加新的物理信息。显式对称信息的主要作用不是修复等变性的理论缺失，而是在**有限标注数据、有限模型容量、有限截断半径和有限优化预算**下提供压缩后的全局归纳偏置。

本课题研究：**在保持底座严格 O(3) 等变的前提下，点群/Wyckoff 条件化的参数高效 adapter 与 decoder 能否提高小样本晶体张量预测的样本效率，同时避免不同对称类别之间的负迁移。**

## 相关工作与研究空缺

球谐函数、不可约表示和 Clebsch–Gordan（CG）张量积是三维等变模型的主流实现 [4,5]。完整 SO(3) 张量积随最高角动量阶数增长较快，eSCN 将 SO(3) 卷积约化为 SO(2)，把其分析的复杂度从 \(O(L^6)\) 降至 \(O(L^3)\) [7]。局部 frame 与 frame averaging 是另一类实现路线 [8,9]，但单一规范姿态在退化或高度对称输入上可能不唯一或不连续 [10]。

已有模型从不同角度加入晶体对称先验：SEN 学习等价原子和多尺度对称模式 [11]；CrysMMNet 把空间群等全局信息作为额外模态 [12]；GMTNet 同时满足 O(3) 变换规律和晶体空间群约束，用于介电、压电和弹性张量预测 [6]；CLOUD 用空间群生成元、Wyckoff 位和组分建立紧凑的无坐标表示 [13]；PRISM 已将多种周期和多尺度表征组织为 mixture-of-experts，但专家按表征类型而非对称类别划分 [16]。

这些工作没有充分回答：在同一个严格周期等变底座、相同训练预算和相同参数量下，显式点群/Wyckoff 信息是否仍能带来增益；这种增益是否主要集中在小样本区间；以及增益来自对称先验、额外参数，还是不同类别 decoder 的容量扩张。

普通 LoRA 或 adapter 直接插入等变层可能混合不兼容的 irrep 并破坏等变性。LoRA 的低秩适配思想来自通用预训练模型 [14]，本课题采用按 \((\ell,\mathrm{parity})\) 分块的等变 adapter 或标量幅值门控 [15]。

## 核心假设

1. **有限数据假设：** 对称性条件化模型相对共享模型的主要优势出现在低数据比例、高成本张量标签和长尾点群中；随着训练数据充分，增益应减小。
2. **decoder 负迁移假设：** 不同点群的合法张量子空间和独立分量数不同，共享 decoder 可能发生梯度冲突；点群专用的低秩残差 decoder 能缓解冲突，同时保留跨类别共享。
3. **输出参数化假设：** 对张量性质，按点群合法基底预测独立系数，比在统一笛卡尔空间直接回归全部分量更具样本效率；对标量性质不存在这种输出维数优势。
4. **Wyckoff 有限模型假设：** Wyckoff/site-symmetry 信息对完整无限容量模型是冗余的，但可帮助有限感受野模型识别非局部等价原子及位点稳定子群；其增益必须超过随机标签对照才可归因于对称性。
5. **路由粒度假设：** 宏观张量首先由 32 个晶体点群约束。按 230 个空间群完全拆分会造成更严重的数据碎片化，空间群和 Wyckoff 信息更适合作为条件特征或层级残差，而非独立大模型路由。

## 方法

### 1. 严格周期等变底座

选择支持周期边界的 O(3)/E(3) 等变模型，正确构建周期邻居、晶格向量和等价原子置换。预训练或统一训练 backbone 后冻结前 \(L\) 层，只微调后层和 decoder。所有专家输出必须具有相同 irrep schema，不能用参数距离代替表示空间一致性的验证。

### 2. 对称信息表示

主路由使用晶体点群，另将晶系、空间群及 Wyckoff/site-symmetry 作为消融。Wyckoff 不只编码诸如 `4a`、`8c` 的字符串，因为字母依赖具体空间群和标准 setting；优先编码

\[
(\text{space group},\ \text{multiplicity},\ \text{site-symmetry group},\ \text{orbit ID}).
\]

也可在晶体图中增加“同一 Wyckoff orbit”关系或 orbit pooling。对于含缺陷、热扰动或数值畸变的结构，需记录空间群识别容差，并分别测试理想结构标签与扰动后标签。

### 3. 共享 decoder 与点群残差 decoder

采用“共享主干 + 点群专用小残差”而非为每个类别训练完全独立的大 decoder：

\[
\widetilde{T}=D_0(z)+\alpha_G\Delta D_G(z),
\]

其中 \(D_0\) 是所有晶体共享的严格等变 decoder，\(\Delta D_G\) 是点群 \(G\) 的低秩等变 adapter，\(\alpha_G\) 为标量门控。\(D_0\) 与 \(\Delta D_G\) 均只能在兼容的 irrep 重数通道之间进行映射 [15]。稀有点群采用晶系级共享残差或直接回退到 \(D_0\)。

完全独立 decoder 仅作为消融。若训练不足来自步数或样本不足，完全拆分会进一步减少每个 decoder 的更新次数；只有当共享 decoder 的主要问题是容量瓶颈或类别间负迁移时，专用 decoder 才应产生稳定收益。

### 4. 点群约束的张量输出

对点群 \(G\)，合法张量位于固定子空间

\[
\operatorname{Fix}_G=\{T\mid \rho(g)T=T,\ \forall g\in G\}.
\]

研究两种输出方式：

1. **投影输出：**

   \[
   \widehat{T}=P_G\widetilde{T},\qquad
   P_G=\frac{1}{|G|}\sum_{g\in G}\rho(g).
   \]

2. **独立系数输出：** 为 \(\operatorname{Fix}_G\) 构造基底 \(B_G\)，decoder 只预测独立系数

   \[
   c_G=D_G(z),\qquad \widehat{T}=B_Gc_G.
   \]

基底必须与晶格取向共同旋转，保证整体 O(3) 等变；不能在固定笛卡尔坐标中使用任意点群 MLP。投影输出作为硬约束基线，独立系数输出用于检验降低输出自由度是否改善小样本学习。GMTNet 已表明简单输出投影可能牺牲数值性能，因此两者必须实证比较 [6]。

标量任务采用共享不变 decoder，并只加入小型条件 adapter。由于不同点群的标量输出空间均为一维，标量任务不使用“点群减少独立分量”作为理论依据。

### 5. 训练目标与负迁移诊断

总损失包括性质损失、随机 O(3) 变换前后的等变误差、张量的点群残差，以及专家参数正则。额外记录不同点群在共享 decoder 上的梯度余弦：

\[
s_{ij}=\cos\left(\nabla_{\theta_D}L_{G_i},\nabla_{\theta_D}L_{G_j}\right).
\]

若不同点群之间持续出现负梯度余弦，且点群残差 decoder 能改善对应类别，才支持“decoder 负迁移”解释。若增加共享 decoder 宽度即可取得相同收益，则增益应归因于容量而非对称专家。

## 实验设计

### 数据规模曲线

在固定测试集上使用

\[
N\in\{1\%,5\%,10\%,25\%,50\%,100\%\}
\]

的训练数据绘制学习曲线。每个比例使用相同数据子集、多随机种子和相同优化预算。主任务至少包含一个张量性质，优先介电、压电或弹性张量；标量任务用于检验结论能否外推。

### 参数量匹配基线

1. 严格等变 backbone + 共享 decoder；
2. 加宽的共享 decoder；
3. 共享 decoder + symmetry embedding；
4. 共享 decoder + Wyckoff/site-symmetry embedding；
5. 共享 decoder + 点群低秩残差；
6. 完全独立的点群 decoder；
7. 随机等规模分组 decoder；
8. 点群投影输出；
9. 点群独立系数输出；
10. GMTNet（张量任务适用）[6]。

所有可比较模型匹配总可训练参数、训练步数和采样策略。对于独立 decoder，同时报告每类实际更新次数，避免把更高训练预算误判为结构优势。

### 数据审计与指标

- 公布各晶系、点群、空间群和 Wyckoff orbit 的样本数及目标分布；采用层级回退处理长尾类别。
- 去除重复结构，使用组成或结构原型感知划分，防止近重复晶体跨集合泄漏。
- 报告 micro MAE、点群 macro MAE、最差类别误差、每类置信区间和多随机种子统计检验。
- 报告 O(3) 等变误差、Neumann 约束违反率、参数量、训练显存、吞吐和每类 decoder 更新次数。
- 比较真实 Wyckoff 标签、随机置换标签和仅使用空间群编号，识别 Wyckoff 信息是否提供了超出额外参数的有效先验。
- 比较真值路由与预测路由；若部署时点群未知，必须计入对称分类错误对性质预测的影响。

## 结果解释与判定标准

- 若对称模型主要在 1%–25% 数据区间领先，且增益随数据增加而缩小，则支持“显式对称信息是有限数据归纳偏置”。
- 若点群残差 decoder 在全数据区间仍领先，同时共享 decoder 上存在稳定负梯度余弦，则支持“类别间负迁移”解释。
- 若加宽共享 decoder 与点群 decoder 表现相同，则不能宣称对称专家有效，收益主要来自容量。
- 若真实点群/Wyckoff 分组不优于随机分组，则不能把收益归因于晶体对称性。
- 若独立系数输出只改善张量任务而不改善标量任务，应把结论限定为“对称约束张量解码”，不外推到一般晶体表示学习。
- 任一精度收益都必须以等变误差不劣于严格底座为前提；普通 LoRA 导致的等变破坏不能作为可接受的精度权衡。

## 风险与预期贡献

主要风险包括：长尾点群使独立 decoder 欠拟合；空间群/Wyckoff 判定依赖数值容差；标准 setting 与坐标基底处理错误会破坏等变性；专家增益可能仅来自额外参数；理想对称标签可能对缺陷和热扰动结构形成错误先验。

若假设成立，预期贡献是：提出保持 O(3) 等变的层级点群条件化 PEFT decoder；系统区分有限数据增益、decoder 负迁移和参数容量效应；给出点群合法张量基底与 Wyckoff 条件信息的适用边界。若没有稳定增益，学习曲线、梯度冲突和对称标签消融仍可说明严格等变模型在何种数据规模下已足以隐式学习晶体对称性。

## 参考文献

[1] O. Anosova, V. Kurlin, and M. Senechal, “The importance of definitions in crystallography,” *IUCrJ*, 2024. https://doi.org/10.1107/S2052252524004056

[2] International Union of Crystallography, “Matrices, mappings, and crystallographic symmetry,” *IUCr Teaching Pamphlet 22*. https://www.iucr.org/education/pamphlets/22/full-text

[3] International Union of Crystallography, “Neumann's principle,” *Online Dictionary of Crystallography*. https://dictionary.iucr.org/Neumann%27s_principle

[4] N. Thomas et al., “Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds,” 2018. https://arxiv.org/abs/1802.08219

[5] S. Batzner et al., “E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials,” *Nature Communications*, 2022. https://doi.org/10.1038/s41467-022-29939-5

[6] K. Yan et al., “A Space Group Symmetry Informed Network for O(3) Equivariant Crystal Tensor Prediction,” *ICML*, 2024. https://proceedings.mlr.press/v235/yan24d.html

[7] S. Passaro and C. L. Zitnick, “Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs,” *ICML*, 2023. https://proceedings.mlr.press/v202/passaro23a.html

[8] W. Du et al., “SE(3) Equivariant Graph Neural Networks with Complete Local Frames,” *ICML*, 2022. https://proceedings.mlr.press/v162/du22e.html

[9] A. A. Duval et al., “FAENet: Frame Averaging Equivariant GNN for Materials Modeling,” *ICML*, 2023. https://proceedings.mlr.press/v202/duval23a.html

[10] N. Dym, H. Lawrence, and J. W. Siegel, “Equivariant Frames and the Impossibility of Continuous Canonicalization,” *ICML*, 2024. https://proceedings.mlr.press/v235/dym24a.html

[11] C. Liang et al., “Material symmetry recognition and property prediction accomplished by crystal capsule representation,” *Nature Communications*, 2023. https://doi.org/10.1038/s41467-023-40756-2

[12] K. Das et al., “CrysMMNet: Multimodal Representation for Crystal Property Prediction,” *UAI*, 2023. https://proceedings.mlr.press/v216/das23a.html

[13] C. Xu, S. Zhu, and V. Viswanathan, “CLOUD: A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning,” *Nature Communications*, 2026. https://doi.org/10.1038/s41467-026-70467-3

[14] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” *ICLR*, 2022. https://arxiv.org/abs/2106.09685

[15] D. Jin, Y. Yuan, and X. Tao, “Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks,” *AAAI*, 2026. https://doi.org/10.1609/aaai.v40i1.37013

[16] À. Solé et al., “PRISM: periodic representation with multiscale and similarity graph modelling for enhanced crystal structure property prediction,” *npj Computational Materials*, 2026. https://doi.org/10.1038/s41524-026-02074-1
