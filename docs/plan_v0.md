# 面向晶体对称性条件化的等变模型专家微调

## 背景与问题定义

周期晶体通常表示为化学种类、晶格矩阵和胞内原子分数坐标的组合；空间群描述把该结构映回自身的等距变换。因而，“组分 + 空间群（或点群、晶系）唯一确定晶体结构”并不成立：同一组分和空间群仍可具有不同晶格参数、Wyckoff 位置、自由位置参数、占位或多晶型。晶体结构需要由晶格、原子种类及位置（必要时还包括占位）共同确定 [1,2]。

Neumann 原理更准确的表述是：描述晶体宏观物性的张量必须在晶体点群的全部操作下保持不变，即物性张量的对称群包含晶体点群 [3]。对于能带等依赖波矢的量，还需考虑空间群在倒空间中的作用、简并和选择定则，不能笼统表述为“所有物性都与空间群具有完全相同的对称性”。

现有原子模型常施加 E(3)、O(3) 或 SO(3) 等变性，使整体平移、旋转或反射输入时，标量、向量和高阶张量输出按正确的群表示变换 [4,5]。这与晶体的内禀空间群不是同一概念：前者是坐标系/整体姿态改变下的协变规律，后者是某个具体结构的稳定子群。O(3) 等变并不强迫晶体物性各向同性；如果空间群操作把结构映回自身（至多发生周期平移和等价原子重标号），严格等变模型的晶体级张量输出会自动满足相应的 Neumann 约束 [6]。因此，本课题不把“用较小的空间群替代 O(3)”作为目标，而研究：**在保持底座 O(3) 等变性的前提下，对称性条件化的参数高效专家能否改善数据效率、预测精度和对稀有对称类别的泛化。**

## 相关工作

球谐函数与不可约表示是三维等变原子模型的主流实现。Tensor Field Networks、NequIP 等使用球谐特征和 Clebsch–Gordan（CG）张量积传递角向信息 [4,5]。完整 SO(3) 张量积随最高角动量阶数增长较快；eSCN 通过将 SO(3) 卷积约化为 SO(2) 卷积，把文中分析的复杂度从 \(O(L^6)\) 降至 \(O(L^3)\) [7]。因此“高阶表示和 CG 路径可能昂贵”是成立的，但并非所有现代等变模型都必须显式维护同样数量的路径。

另一类方法利用局部参考标架或规范化。ClofNet 构造完备局部标架以获得 SE(3) 等变性 [8]；FAENet 通过 frame averaging 为材料模型施加 E(3) 对称性 [9]。在高度对称或特征值退化的输入上，单一规范姿态可能不唯一或不连续；该问题已有一般性的连续规范化不可能性分析 [10]。因此需要区分“单一局部 frame”“多 frame 平均”和球谐/笛卡尔张量方法，不能把局部 frame 简单归因于蛋白模型。

已有工作已显式利用晶体对称性。SEN 学习等价原子和多尺度晶体对称模式 [11]；CrysMMNet 将空间群等全局信息作为额外模态 [12]；GMTNet 同时保证 O(3) 等变和晶体空间群约束，用于介电、压电和弹性张量预测 [6]。最新的 CLOUD 则把空间群生成元、Wyckoff 位和组分编码为对称性一致的晶体表示 [13]。PRISM 已将多种局部、周期和多尺度晶体表征融合为 mixture-of-experts，但其专家按表征类型而非晶体对称类别划分 [16]。这些工作说明“显式加入晶体对称信息”有研究依据，同时也意味着仅加入空间群标签、对称性模块或晶体 MoE 本身不足以构成创新点。

LoRA 通过低秩权重增量降低微调参数量 [14]，但普通 LoRA/adapter 直接插入等变层可能破坏不可约表示的变换规律。面向等变图网络的研究已指出这一风险，并提出按张量阶数和通道调制的严格等变 adapter [15]。因此本课题必须使用 irrep 保持的线性映射、标量门控或等变 LoRA，而不能直接套用普通 LoRA。

## 研究假设

1. 在参数量和训练预算匹配时，对称性条件化的等变 adapter 比单一共享 adapter、仅拼接对称性标签或随机分组专家具有更低的测试误差，尤其有利于数据充分且物性受对称性显著约束的类别。
2. 对宏观张量性质，按 32 个晶体点群路由通常比按 230 个空间群更符合 Neumann 原理，也更缓解数据碎片化；空间群路由是否额外有效作为消融实验检验。
3. 对形成能、带隙等标量性质，对称性不会对输出分量施加额外的张量零元约束，专家的收益只能来自统计先验而非“更严格的等变性”，其效果需要实验而不能预设。

## 方法

1. **任务与底座。** 选择一个支持周期边界的 O(3)/E(3) 等变底座，分别研究至少一个标量任务和一个张量任务。先训练或使用统一预训练底座，冻结前 \(L\) 层；晶格、周期邻居和等价原子置换必须在数据管线中正确处理。
2. **路由粒度。** 主实验按晶体点群路由；另比较晶系（7 类）、空间群（230 类）、按训练集频数合并的层级类别及随机等规模分组。路由标签由输入结构确定并在整体 O(3) 变换下保持不变。若部署时结构未经标准化或标签未知，需单独报告“真值路由”和“预测路由”的差距。
3. **专家结构。** 所有样本共享底座和 decoder，每类仅增加后若干层的残差 adapter。adapter 对每种 irrep 分块，仅在相同 \((\ell,\text{parity})\) 的重数通道间混合，或采用标量幅值门控，以保证专家分支仍严格 O(3) 等变 [15]。专家数较多时使用层级共享（共享 adapter + 类别残差），避免小类完全独立训练。
4. **训练约束。** 优化性质损失，并按任务加入：(a) 随机 O(3) 变换前后的等变误差；(b) 张量在点群操作下的 Neumann 残差；(c) 专家残差范数或相对共享 adapter 的正则。参数接近并不能证明表示位于“同一空间”；共同的 irrep 类型、共同 decoder 和显式变换测试才是可检验条件。
5. **输出。** 标量用不变 pooling 与共享标量 decoder；张量用对应阶数/奇偶性的不可约表示 decoder。可选地在输出端施加群平均投影作为“硬约束”基线，但需与结构内生的对称模型分别比较，因为 GMTNet 已观察到简单输出投影可能牺牲预测性能 [6]。

## 实验设计与判定标准

- **数据审计：** 公布每个晶系、点群和空间群的训练/验证/测试样本数及目标分布；为长尾类别设最低样本阈值或层级回退。划分时去除重复结构，并使用组成或原型感知划分检查泛化，避免近重复晶体跨集合泄漏。
- **必要基线：** 冻结底座 + 线性头、全量微调、单一共享 adapter、仅加入 symmetry embedding、参数量匹配的普通多专家、随机类别专家、输出群平均、GMTNet（张量任务适用）[6]。
- **消融：** 冻结层数 \(L\)、专家粒度、hard/soft routing、共享强度、adapter 秩、是否使用等变 adapter、真值/预测路由。
- **指标：** 总体 micro MAE、类别 macro MAE、每类置信区间、最差类别误差、旋转/反射等变误差、点群约束违反率、参数量、训练显存和推理吞吐。主结论必须来自多随机种子及配对统计检验。
- **支持命题的最低证据：** 对称性专家需在参数量和算力匹配条件下稳定优于单 adapter、symmetry-token 和随机路由；同时等变误差不劣于底座。若仅张量任务受益，应将结论限定为对称性约束张量预测，不外推到一般晶体表示学习。

## 风险与预期贡献

主要风险是类别长尾导致专家欠拟合、空间群判定对数值容差和结构扰动敏感、普通 LoRA 破坏等变性，以及专家增益仅来自额外参数。若严格对照后仍有收益，贡献可表述为“保持 O(3) 等变的层级对称性条件化 PEFT”及其对点群/空间群粒度的系统实证；若无稳定精度收益，类别间 adapter 差异、长尾效应和对称性违反率仍可形成有价值的负结果与诊断。

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
