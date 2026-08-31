# 基于 MACE 基础势的对称性专家参数高效微调

## 1. 研究问题

本课题研究：在保持 MACE 势能模型严格 \(E(3)\) 等变的前提下，按晶体对称性划分或学习专家，能否提高有限标注数据下的能量—力联合微调效率；这种增益如何随数据量、可训练层范围和更新参数化方式变化。

严格等变性保证整体旋转、反射和平移下输出具有正确变换规律，但并不保证有限容量、有限截断半径和有限优化预算下，模型能高效学习不同晶体结构域的统计差异。因此，晶系、点群和空间群在这里是**路由/条件变量**，不是用来修复 MACE 的等变性，也不应把受热扰动后的力硬投影到母相对称子空间。

核心问题为：

1. 只更新能量 readout、更新尾部若干层、更新除首层外全部层和全模型更新，何者具有最佳样本效率？
2. 在相同更新范围内，ELoRA 是否优于直接梯度更新参数？
3. 基于晶系、点群、空间群的硬路由，是否优于共享模型与可学习 router？
4. 专家化的收益是否主要出现在中小数据区间，并在数据充分后饱和或缩小？

## 2. 首个 benchmark 与具体任务

### 2.1 主任务：预训练 MACE 势的能量—力联合微调

首个任务固定为**周期性无机晶体构型的总能量和原子力预测**：

- 输入：元素种类、原子坐标、周期性晶胞和周期边界条件；
- 输出：构型总能量 \(E\) 与保守力
  \[
  \mathbf F_i=-\frac{\partial E}{\partial \mathbf r_i};
  \]
- 初始模型：MACE-MP 系列公开预训练 checkpoint；
- 训练目标：
  \[
  \mathcal L=\lambda_E\mathcal L_E+\lambda_F\mathcal L_F,
  \]
  能量按原子数归一化，力按笛卡尔分量计算；
- 主指标：energy MAE（meV/atom）与 force MAE（meV/Å）；
- 次指标：energy/force RMSE、每个对称类别的宏平均与最差类别误差、训练时间、峰值显存和可训练参数量；
- 后续任务：应力预测、分子动力学稳定性与材料张量预测，但不纳入第一阶段的主结论。

选择该任务的原因是：MACE 通过等变消息传递和高阶相关构造预测标量势能，力由能量梯度获得，已有成熟的能量—力训练流程 [1]；MACE-MP-0 又提供了在大规模材料数据上预训练并可下游微调的基础势 [2]。

### 2.2 两条 benchmark 轨道

为避免把“复现 ELoRA”与“验证对称专家”混成同一结论，实验分成两条轨道：

**轨道 A：ELoRA 实现复现。** 按 ELoRA 论文在 rMD17 和其无机材料数据上的能量—力设置，先复现共享 ELoRA 相对于冻结模型和常规微调的趋势 [3]。rMD17 是分子数据，不含可用的晶体空间群，因而只验证实现，不用于证明晶体对称路由有效。

**轨道 B：对称路由主实验。** 使用覆盖多个母相结构、组成和晶体学类别的周期性无机数据。优先审计并复用 ELoRA/MACE benchmark 的无机部分；只有它满足下列条件时，才直接作为主 benchmark：

- 每个进入比较的类别包含多个独立母结构，而不是一个类别对应一个材料；
- 训练、验证和测试能按母结构/轨迹分组切分；
- 各晶系和主要点群有足够样本，空间群长尾可被显式处理；
- 能核查其与 MACE-MP 预训练数据 MPtrj 的重合，区分真正迁移与预训练记忆。

如果原 benchmark 不满足这些条件，则保留轨道 A 作为复现，并另建一个满足上述条件的 `MACE-SymFT` 多晶体下游集合。不能仅把少数材料的 MD 帧混合后随机拆分，否则空间群专家可能只是在识别材料身份，测试帧也会与训练轨迹高度相关。

### 2.3 对称标签口径

热扰动、声子位移和弛豫中间构型的瞬时空间群通常显著降级，甚至被识别为 \(P1\)。因此主实验采用：

- 以弛豫参考结构或轨迹母相的晶系、点群和空间群作为该轨迹全部构型的**结构域标签**；
- 用统一的标准化流程和固定容差生成标签，并保存国际空间群号、点群和晶系；
- 另做“逐帧瞬时标签 × 多种容差”的敏感性分析；
- 标签只选择专家，不硬约束瞬时能量或力满足母相的内部对称性。

这一区分很重要：全局 \(E(3)\) 等变性必须对每个构型严格保持，而母相空间群在热扰动构型上只是条件信息。

## 3. 笛卡尔积实验空间

完整实验定义为

\[
\mathcal E=\mathcal S_{\text{scope}}
\times\mathcal P_{\text{update}}
\times\mathcal R_{\text{router}}
\times\mathcal N_{\text{data}}.
\]

### 3.1 可训练范围 \(\mathcal S_{\text{scope}}\)

1. **Decoder/readout only（D）**：只更新 MACE 的全部逐层 energy readout，以及预先规定的 scale/shift；原子参考能 \(E_0\) 采用统一策略重新估计或固定，不能让不同方法使用不同的 \(E_0\) 处理。
2. **Last-\(L\)（T\(_L\)）**：更新最后 \(L\) 个 interaction/product block 及其 readout，\(L\in\{1,2,\ldots,B\}\)，其中 \(B\) 是所用 checkpoint 的实际 block 数。
3. **Except-first（NF）**：冻结元素嵌入、径向基与第一个 interaction/product block，更新其余层及 readout。
4. **Full coverage（F）**：允许在整个模型的所有合格层中更新参数。

MACE 并非只有一个普通神经网络“最后 decoder”，而是可从多个交互层产生能量贡献。因此，D 必须按实际模块名定义为全部 readout，而不是含糊地仅解冻最后一个线性层。若某个浅层 checkpoint 中 T\(_L\) 与 NF/F 实际覆盖相同，则合并重复条件，不重复宣称为独立实验。

### 3.2 更新参数化 \(\mathcal P_{\text{update}}\)

每个范围分别比较：

1. **Dense**：对该范围内的原始参数直接进行梯度更新；
2. **ELoRA**：冻结原始等变权重，仅在该范围内合格的等变路径上加入低秩更新。

ELoRA 将低秩分解放在等变 tensor-product 路径上，目的在于参数高效适配同时保持 \(SO(3)\) 等变性 [3]。本课题使用其 MACE 实现作为起点，但需明确：

- “F × ELoRA”表示 ELoRA 覆盖全模型合格层，不等于全参数 dense fine-tuning；
- 非等变标量模块、径向模块、readout、scale/shift 是否训练必须形成固定白名单，并在所有实验中报告；
- rank \(r\)、缩放系数、初始化和可合并性要由单元测试确认，不依赖仓库中的隐式硬编码；
- 对同一路由方案同时报告总存储参数、可训练参数和单次前向激活参数，避免仅以“可训练参数少”掩盖多专家存储成本。

### 3.3 路由与专家 \(\mathcal R_{\text{router}}\)

至少包含下列五类：

1. **Shared**：无专家、所有样本共享同一套更新，是必要基线；
2. **Crystal-system experts**：按 7 个晶系硬路由；
3. **Point-group experts**：按 32 个晶体学点群硬路由；
4. **Space-group experts**：按最多 230 个空间群硬路由；
5. **Learned-\(K\) experts**：可学习 router 将构型路由到 \(K\) 个专家。

所有专家共享冻结的基础权重 \(W_0\)，只把所选范围内的增量设为专家特异：

\[
W_g=W_0+\Delta W_g.
\]

Dense 条件下，\(\Delta W_g\) 是对应范围的直接参数副本/残差；ELoRA 条件下，\(\Delta W_g\) 由该专家的低秩路径参数产生。这样“空间群专家”不意味着复制 230 个完整 MACE。

#### 共享参数的加载与计算约束

实现必须采用“一份共享骨干 + 一个专家差分参数库”，而不是为每个类别构造一份完整模型：

\[
\Theta=\Theta_{\mathrm{shared}}+\{\Delta\Theta_1,\ldots,\Delta\Theta_K\}.
\]

- 初始化时，MACE 的 embedding、interaction/product blocks、readout 等共享参数只从 checkpoint **加载并驻留一份**；
- 专家差异保存为按 expert id 索引的参数 bank。Dense 方案保存所选 scope 的 \(\Delta\Theta_g\)，ELoRA 方案只保存各专家的低秩因子；
- 前向计算时，router 产生 expert id 或 top-\(k\) 权重，逐层选取相应差分。对第 \(\ell\) 层采用
  \[
  W_{\ell,g}=W_{\ell,0}+\Delta W_{\ell,g},
  \qquad
  h_{\ell+1}=\Phi_\ell(h_\ell;W_{\ell,g}),
  \]
  或 ELoRA 在各等变路径上的对应低秩形式。不能把含非线性的整个深层网络误写成两个完整模型输出的简单相加；不得在路由时重新加载 checkpoint，也不得临时实例化 \(K\) 个完整 MACE；
- 一个 batch 含多个 expert id 时，按路由结果分组或使用 batched gather/einsum 读取参数 bank，再恢复原样本顺序；共享模块保持同一实例；
- checkpoint 分开保存 shared state、router state 和 expert-delta state，加载后用参数对象标识和显存占用检查共享权重没有被复制；
- 参数成本应满足
  \[
  P_{\mathrm{stored}}
  =P_{\mathrm{shared}}+\sum_{g=1}^{K}P_{\Delta g}
  +P_{\mathrm{router}},
  \]
  而不是 \(K P_{\mathrm{shared}}\)。推理时另报告被激活的专家参数量。

如果在较深层才开始专家化，应先完成所有样本的共享前缀计算，再对尾部特征进行分组路由，以避免重复执行共享前缀。若每层都带专家增量，则每层仍复用同一个共享算子，只对差分分支做索引。

对空间群长尾采用预先规定的层级回退：

\[
\text{space group}\rightarrow\text{point group}
\rightarrow\text{crystal system}\rightarrow\text{shared},
\]

仅当某一叶节点的独立母结构数和训练构型数均超过阈值时才建立独立专家。阈值只根据训练集确定。

可学习 router 必须满足：

- 在**构型级**而非原子级选择专家；
- 输入只使用旋转不变量，如池化后的 \(0e\) 特征、组成与晶胞不变量；
- 比较 top-1 与 top-2 路由，并加入负载均衡正则；
- 报告各专家利用率、router 熵、塌缩率，以及其与已知晶体标签/组成的互信息；
- \(K\in\{4,8,16\}\) 先在验证集筛选，不能根据测试集选择。

### 3.4 数据量 \(\mathcal N_{\text{data}}\)

固定验证集和测试集，仅对训练母结构分层抽样：

\[
\mathcal N=\{1\%,5\%,10\%,25\%,50\%,100\%\}.
\]

若 1% 无法让主要类别达到最低母结构数，则改用按类给定的绝对样本档位。抽样单位优先为母结构/轨迹，而不是独立帧；同时报告全局训练量与每个专家的有效样本量。

## 4. 可检验假设

### H1：冻结范围存在任务相关的最优点

小数据下 D 或 T\(_1\) 预期优于大范围 dense 更新，因为后者更易过拟合或遗忘；随着数据增加，NF/F 的上限可能更高。该结论不能预设，须由学习曲线判断。

### H2：ELoRA 的优势取决于更新范围和数据量

在参数预算受限和小数据条件下，ELoRA 预期以更少参数接近或超过 dense 更新；在数据充分且允许全模型更新时，dense 的性能上限可能更高。比较同时提供原生配置和预算匹配配置。

### H3：专家增益不是简单单调增长

将专家方法相对 Shared 的增益记为

\[
G(N)=\mathrm{MAE}_{\mathrm{shared}}(N)-
\mathrm{MAE}_{\mathrm{expert}}(N).
\]

更稳健的预期是“阈值—增长—饱和”：极小数据时，切分样本可能使每个专家欠训练，\(G(N)\le 0\)；达到最低每类样本量后，专业化减少负迁移，\(G(N)>0\)；数据继续增长后，两者都逐渐饱和，绝对增益平台化，相对增益可能缩小。用户提出的“随数据量增长先变好后饱和”主要适用于专家的绝对性能；其**相对共享模型的增益**未必单调，这是本实验需验证的重点。

### H4：对称标签应提供超越材料身份的贡献

若真实对称标签只在普通随机切分中有效，却在组成控制或母结构不相交切分中消失，则不能声称模型学到了可迁移的对称结构域。

## 5. 数据切分与反混淆设计

1. **Group split**：同一母结构、同一 MD/位移轨迹的所有帧只能进入同一个 split。
2. **主要 IID 测试**：按母结构分组且按晶系/点群分层。
3. **化学 OOD 测试**：组成或化学体系不相交，用于检测空间群是否只是成分代理。
4. **预训练重合审计**：依据材料标识、组成、标准化结构指纹检查与 MPtrj 的重合；分别报告 seen-like 与 novel subsets。MACE-MP-0 的训练数据来自 MPtrj，且其能量约定需要与下游标签和原子参考能对齐 [2,4]。
5. **标签对照**：增加与真实类别频率相同的随机分组、全局打乱标签、组成内打乱标签。若真实对称路由未显著优于这些对照，则证据不足。
6. **类别充分性**：每个主报告类别至少包含多个组成、多个母结构；否则合并到父级或只作案例分析。

## 6. 评价协议

### 6.1 精度与样本效率

- Energy MAE/RMSE（meV/atom）；
- Force component MAE/RMSE（meV/Å）；
- micro average、按类别 macro average、最差类别和类别间方差；
- 达到给定误差阈值所需训练样本量；
- 3–5 个随机种子的均值、标准差与配对置信区间。

### 6.2 对称性与物理一致性

对随机旋转/反射 \(R\)、平移和等价晶胞表示测试：

\[
\epsilon_E=|E(Rx)-E(x)|,
\qquad
\epsilon_F=\|F(Rx)-RF(x)\|.
\]

ELoRA、dense 更新和 router 都必须在数值精度内保持能量不变性与力等变性。可学习 router 若依赖非不变量，会直接破坏这一要求。

### 6.3 效率与稳定性

- 总参数、可训练参数、激活参数、checkpoint 大小；
- 验证共享骨干仅加载一份，并分别报告 \(P_{\mathrm{shared}}\)、专家差分参数总量与单样本激活的差分参数量；
- 单步训练时间、推理吞吐、峰值显存；
- router 负载、专家塌缩率；
- 在预训练域保留集上的性能变化，用于衡量灾难性遗忘；
- 对最终候选做短程 NVE/NVT 稳定性测试，但不以单条轨迹替代统计测试。

## 7. 实际执行顺序

完整笛卡尔积用于定义问题，但不一次性穷举。若按 4 类 scope、2 类参数化、5 类路由、6 个数据量和 3 个种子计算，至少已有 720 次训练，尚未包含 \(L\)、rank 和 \(K\)。因此采用预注册的分阶段设计：

### 阶段 0：实现与 benchmark 对齐

1. 锁定 MACE checkpoint、数据版本、单位、邻居截断、\(E_0\)、损失权重和 split；
2. 在 ELoRA benchmark 的小规模设置复现 Shared-Dense、Shared-ELoRA；
3. 对 ELoRA 各目标层做参数清单、rank/scale 检查、合并前后输出检查和随机旋转等变测试；
4. 对本地 ELoRA 分支做版本固定，因为其 MACE 分支依赖特定修改版 e3nn，不能默认与当前标准 e3nn 互换。

### 阶段 1：筛选更新范围与参数化

只使用 Shared 路由，在 

\[
\mathcal S_{\text{scope}}\times
\mathcal P_{\text{update}}
\]

上比较 10%、50%、100% 三个数据量。选择 Pareto 前沿上的 2–3 个配置，标准为验证误差、可训练参数和训练成本，不查看测试集。

### 阶段 2：比较专家路由

在阶段 1 入选配置上比较 Shared、晶系、点群、空间群和 Learned-\(K\)。

### 阶段 3：完整数据缩放曲线

仅对阶段 2 的 Shared 基线和 2–3 个最佳专家方案运行全部六个数据量与 3–5 个随机种子，拟合饱和曲线并检验 H3。

### 阶段 4：跨模型与跨任务拓展

- foundation model：从 MACE-MP 扩展到 NequIP/NequIP-OAM 或 Allegro。NequIP 同样是严格 \(E(3)\) 等变原子势 [5]，但 ELoRA 的插入点需要按其 tensor-product 实现重新审计，不能直接复用 MACE patch；
- benchmark：扩展到其他无机轨迹、缺陷、表面或高温数据；
- task：在能量—力结论稳定后，再扩展到应力及 plan_v1 中的晶体张量预测。

## 8. 最小必要消融

最终论文至少报告：

1. zero-shot MACE、Shared-Dense、Shared-ELoRA；
2. 同一 scope 下 Dense vs ELoRA；
3. 同一参数化下 D、T\(_L\)、NF、F；
4. Shared vs 晶系 vs 点群 vs 空间群 vs Learned-\(K\)；
5. 真实标签 vs 等频随机标签 vs 组成内打乱标签；
6. 原生参数成本与参数预算匹配两种比较；
7. 母相标签 vs 瞬时标签；
8. group-disjoint 与 chemistry-disjoint 测试；
9. 有无层级回退和负载均衡正则；
10. 不同数据量下每类有效样本数与专家增益。

## 9. 成功、失败与结论边界

支持课题命题需要同时观察到：真实对称路由在至少一个中小数据区间稳定优于 Shared 和随机标签；增益在母结构不相交测试中存在；在预算匹配后仍成立；且等变误差没有恶化。

以下结果也具有明确科研价值：

- 若 Learned-\(K\) 优于显式对称标签，说明最优结构域不等同于晶体学分类；
- 若只有晶系有效而空间群无效，说明细粒度类别的数据碎片化超过专业化收益；
- 若所有专家均不优于 Shared，说明严格等变基础势的共享表示已足够，或当前下游数据不足以区分对称先验与材料身份；
- 若专家只在同轨迹随机切分中有效，则应判为数据泄漏/身份记忆，而非对称性增益；
- 若 Dense 在大数据下胜出而 ELoRA 在小数据下占优，则形成清晰的预算—数据量选择规律，而不是简单宣称某方法全面更好。

## 10. 预期贡献

1. 给出 MACE 基础势在不同解冻深度上的系统微调基线；
2. 明确 ELoRA 与直接更新在不同数据规模和参数预算下的适用区间；
3. 区分晶体学硬路由、学习路由和随机分区，验证对称标签是否具有超越材料身份的增益；
4. 建立专家收益随**每专家有效数据量**变化的学习曲线；
5. 形成可迁移到 NequIP 等其他严格等变基础势的实验协议。

## 参考文献

[1] Batatia, I. et al. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. *NeurIPS*, 2022. https://arxiv.org/abs/2206.07697

[2] Batatia, I. et al. A foundation model for atomistic materials chemistry. *The Journal of Chemical Physics*, 2025. https://doi.org/10.1063/5.0297006

[3] Wang, H. et al. ELoRA: Low-Rank Adaptation for Equivariant GNNs. *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267:63113–63135, 2025. https://proceedings.mlr.press/v267/wang25al.html

[4] MACE Developers. MACE documentation and pretrained foundation-model usage. https://github.com/ACEsuit/mace

[5] Batzner, S. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications* 13, 2453, 2022. https://doi.org/10.1038/s41467-022-29939-5

[6] NequIP Developers. NequIP foundation-model documentation. https://nequip.readthedocs.io/en/latest/guide/getting-started/foundation-models.html
