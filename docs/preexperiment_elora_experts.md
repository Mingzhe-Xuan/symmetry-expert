# ELoRA 对称专家预实验计划

## 1. 目标与结论边界

本预实验服务于 [plan_v2](plan_v2.md)，固定预测目标为周期性晶体的能量—力联合微调，只考察

\[
\mathcal S_{\mathrm{scope}}
\times\mathcal P_{\mathrm{update}}
\times\mathcal R_{\mathrm{router}}
\times\mathcal N_{\mathrm{data}}.
\]

预实验要回答：

1. 在同一预训练 MACE、同一数据 split 和同一训练预算下，ELoRA 是否优于直接梯度更新；
2. 在同一 update 与 scope 下，对称专家是否优于共享更新；
3. 专家收益是否超越“增加总参数量”与“任意切分数据”带来的收益；
4. scope、update、router 和每类数据量之间是否存在稳定交互规律。

本文中的“直接训练”指**对预训练 MACE 的所选原始参数进行 dense gradient update**，不是从随机初始化训练。From-scratch 只作为昂贵的参考基线，不是主要对照。

预实验的最小可信结论是：

> 在相同 \(S,P,N\) 下，真实对称路由相对 Shared 有稳定增益，并且优于等容量随机路由。

若只观察到 Expert-ELoRA 优于 Shared-Dense，不能判断收益来自 ELoRA、专家化还是额外参数，因而不构成支持证据。

## 2. 总体路线

预实验分为两条轨道。

### 轨道 A：论文逻辑复现

目的：验证本地 ELoRA 代码、环境和训练超参数能复现论文中“Shared-ELoRA 在小数据下优于 Shared-Dense”的趋势。

首选设置：

- MACE-OFF + rMD17 aspirin；
- 训练量 \(N\in\{50,200,1000\}\)；
- Shared-Dense full fine-tune；
- Shared-ELoRA，rank 16；
- 相同 train/validation/test 索引；
- 3 个随机种子；
- 报告 energy/force MAE，并与论文图 4 的趋势而非单个小数点对齐。

若有可直接获取的论文 Cu 数据，再增加 MACE-MP + Cu 的 full/readout/ELoRA 对比，用于核对无机设置。轨道 A 不含 symmetry router，因为 rMD17 和单一 Cu 体系不足以形成平衡的多空间群实验。

### 轨道 B：四轴专家预实验

目的：在多空间群、类别平衡的数据上验证

\[
S\times P\times R\times N.
\]

主数据优先使用未参与 MACE-MP 预训练、具有能量和力标签的周期无机数据子集。建议从 OMat24 中抽取小型平衡子集；其公开数据包含非平衡结构、结构弛豫以及能量、力和应力标签 [2,3]。如果下载、许可或 parent-id 元数据阻碍快速执行，可先使用 MPTrj 子集完成工程预实验，但必须把结果标为“预训练域内管线验证”，不能作为下游泛化证据。

## 3. 数据构建

### 3.1 强制性数据集统计描述

**任何训练开始前，必须先生成、审阅并冻结数据集统计报告。没有统计报告，或类别统计未通过检查时，不得启动 Stage 1–4。** 报告不能只给总结构数，必须以分类类别为核心，逐级描述晶系、点群和空间群。

#### 3.1.1 总体统计

至少报告：

- 原始记录数、有效标签记录数、去重后独立结构数和 parent 数；
- 重复结构、缺失 energy/force、异常晶胞和无法识别空间群的数量；
- 元素覆盖、元素共现、化学式数量、组成体系数量和结构原型数量；
- 每结构原子数、体积/原子、energy/atom、force norm 的均值、标准差、中位数、四分位数、最小值和最大值；
- 数据来源、计算设置、trajectory 类型和 relaxation stage 的数量分布；
- 与 MACE-MP 预训练数据已知或疑似重合的结构数。

#### 3.1.2 分类类别统计

分别对 crystal system、point group 和 space group 生成逐类表格。每一行至少包含：

| 字段 | 含义 |
|---|---|
| category level | 晶系 / 点群 / 空间群 |
| category id | 标准类别编号或名称 |
| international symbol | 国际符号；空间群同时记录国际编号 |
| parent category | 对应点群与晶系 |
| raw configurations | 筛选前构型数 |
| unique structures | 去重后的独立结构数，作为 \(n_g\) |
| unique parents | 独立 parent/reference structure 数 |
| valid labels | 同时具有 energy 和 force 标签的数量 |
| removed | 是否因 \(n_g<100\) 被整类删除 |
| retained configurations | 删除、去重和平衡后的数量 |
| train / valid / test | 三个 split 的独立结构数 |
| compositions / prototypes | 类内组成和原型数 |
| atoms median [IQR] | 类内原子数分布 |
| energy/force summary | 类内标签分布摘要 |

必须同时给出：

- 筛选前类别总数、被删除类别数、保留类别数；
- 所有 \(n_g<100\) 类别的完整删除清单及其结构数；
- 删除操作移除的结构总数和占原数据比例；
- 保留类别的最大/最小样本比、变异系数、类别熵和有效类别数；
- crystal-system × point-group × space-group 的层级对应表；
- space-group × 数据来源、space-group × 主要组成体系的交叉表；
- 各 split 的逐类计数，确认每个专家在 train/valid/test 均有样本；
- 平衡前与平衡后的逐类计数对照。

#### 3.1.3 必须绘制的图

- 晶系、点群、空间群样本数条形图，空间群另提供对数纵轴版本；
- 筛选前后类别计数对照图；
- train/valid/test 的逐类堆叠条形图；
- 各空间群的原子数、energy/atom 和 force norm 箱线图；
- space group × composition family 热图；
- parent 数与 configuration 数对照图，用于识别少数 parent 产生大量 frame 的伪大类。

#### 3.1.4 训练前验收条件

统计报告必须满足：

1. 最终主分类粒度的所有保留类别均有 \(n_g\ge100\) 个去重后的独立结构；
2. 空间群删除审计及最终主分类的 \(n_g<100\) 整类删除均已完成，删除清单可以从 manifest 重现；
3. 每个结构的晶系、点群和空间群层级映射一致；
4. train + valid + test 与保留总数逐类精确相等；
5. 同一 parent 不跨 split，结构指纹无跨 split 重复；
6. 平衡主实验的最大类/最小类结构数之比不超过预注册阈值；
7. 每个入选类别包含足够的组成或原型多样性，类别不是单一材料 id 的替代变量；
8. 类别集合、删除列表、平衡配额和 split hash 在训练前冻结。

若任一条件不满足，必须先修正数据或缩小类别集合，不能通过 loss weighting 掩盖数据问题。

### 3.2 空间群筛选

对当前目标确有 energy/force 标签的结构进行标准化和去重，然后统计空间群。严格执行 plan_v2 的删除规则：

\[
n_g<100\ \Longrightarrow\ \text{删除空间群 }g\text{ 的全部结构}.
\]

被删除类别不进入 train、validation 或 test，也不建立专家。

为得到可用的数据曲线，预实验从保留类别中再选择 \(N_{\mathrm{SG}}=4\) 个空间群，优先要求每类至少 300 个独立结构。选择只依据：

- 样本量；
- 晶系和点群覆盖；
- 元素、组成和结构原型多样性；
- 原子数分布是否相近；
- 标签完整性。

不得根据模型误差选择空间群。若没有 4 类达到 300，则降为 3 类；若连 3 类都不满足，则使用全部 \(n_g\ge100\) 的类别并采用自适应数据档位。

#### 3.2.1 空间群—晶系自动门控

空间群筛选完成后，计算因 \(n_g<100\) 规则删除的去重独立结构总数 \(N_{\mathrm{removed,SG}}\)：

- 若 \(N_{\mathrm{removed,SG}}\le2000\)，主分类保持为空间群；
- 若 \(N_{\mathrm{removed,SG}}>2000\)，主分类自动切换为晶系。此时回到应用空间群删除前的同一合格去重集合，按晶系重新统计，并删除所有 \(n_c<100\) 的晶系类别。

阈值使用严格的“大于 2000”，计数对象是去重独立结构，不是 trajectory frame/configuration。该决定只依赖训练前数据统计，写入 classification_decision.json 并在任何模型训练前冻结。若切换到晶系，后文 Stage 2–4、主指标和成功标准中的 Space-group/Space 均解释为 Crystal-system/Crystal；Learned-\(K\) 与 Random-\(K\) 的 \(K\) 改为保留晶系数，Random 保持相同类别频率。不得在看到模型结果后切回空间群。

### 3.3 对称标签

优先使用 parent/reference relaxed structure 的空间群作为该 parent 所有衍生构型的路由标签。若外部数据无法恢复 parent 结构，则：

1. 以固定 spglib 版本计算 instantaneous space group；
2. 同时用两组预注册容差检查标签稳定性；
3. 只保留两组容差下标签一致的构型；
4. 在报告中明确该轨道检验的是“瞬时对称类别路由”，不是母相结构域路由。

不能用目标能量、力或测试误差参与标签生成。

### 3.4 去重与切分

- 使用 source id、标准化结构指纹和组成联合去重；
- 若同一 parent 有多个 relaxation/MD frame，则按 parent id 分组；
- 70% parent 用于训练，15% 验证，15% 测试；
- 同一 parent 的所有 frame 必须位于同一 split；
- 最终主分类的每个类别在各 split 中使用相同配额；
- OMat24 与 MPtrj 做可行范围内的结构指纹重合审计；
- 不混用不同 DFT 约定的能量标签；对下游训练集单独估计 \(E_0\)，并把策略固定到所有方法。

若使用 trajectory 数据，每个 parent 固定抽取相同数量和相近阶段的非平衡构型，避免一个长轨迹在损失中占据过大权重。

### 3.5 数据量轴

若最终主分类的每个入选类别至少有 300 个独立结构，则固定每类：

- train pool：200；
- validation：50；
- test：50；
- 训练档位：\(n_{\mathrm{train/class}}\in\{25,50,100,200\}\)。

所有较小档位是 200-structure train pool 的嵌套子集，并在所有 \(S,P,R\) 方法间共享。这样 \(N=25\) 的样本完全包含于 \(N=50\)，可做配对学习曲线。

若类别只有 100–299 个结构，则先固定 validation/test 配额，再令

\[
n_{\max}=\min_g n_{g,\mathrm{train}},
\qquad
\mathcal N=\{0.125,0.25,0.5,1.0\}n_{\max},
\]

并取整为每类相同数量，最低档不少于 8 个结构/类。

## 4. 四个实验轴

### 4.1 Scope：\(\mathcal S_{\mathrm{scope}}\)

使用实际 MACE 模块名定义，不用模糊的“最后几层”：

| 编号 | Dense 可训练参数 | ELoRA 可训练参数 |
|---|---|---|
| D | 全部 energy readouts，必要的 scale/shift | 与 Dense 相同；当前仓库在此 scope 无可用 ELoRA 路径，因此是退化对照 |
| T1 | 最后 1 个 interaction/product block + readouts | 最后 1 个 product block 内的 ELoRA + dense readouts |
| NF | 除 embedding、radial embedding 和第一个 interaction/product block 外的层 + readouts | 对应后续 product blocks 的 ELoRA + dense readouts |
| F | 全部原始模型参数 | 所有合格 product blocks 的 ELoRA + dense readouts |

对于常见的两层 MACE-MP checkpoint，T1 与 NF 可能实际相同。必须先打印参数名集合；若集合相同则合并为一个条件，不重复计算。

D × ELoRA 不应被宣传为 ELoRA 优势实验，因为当前实现只在 symmetric contraction 上放置低秩因子。该单元只用于验证 scope 控制，主要 \(P\) 对比从 T1 开始。

### 4.2 Update：\(\mathcal P_{\mathrm{update}}\)

主实验只有两个清晰定义：

1. **Dense**：直接更新 scope 内原始参数；
2. **ELoRA-clean**：冻结原始权重，只训练 scope 内 ELoRA 因子和统一规定的 dense readout。

另外保留一个不进入完整笛卡尔积的 **ELoRA-paper-compatible** 配置：复现当前仓库对 radial embedding 和部分 symmetric-contraction 参数的额外解冻。它只用于轨道 A 与论文对齐。

ELoRA 默认 rank 16、\(\alpha=16\)。在正式四轴实验前，用 Shared-T1 在一个中等数据档位扫描

\[
r\in\{4,8,16,32\},
\]

选定一次后锁定。不同 router 不得分别选择最有利 rank。

### 4.3 Router：\(\mathcal R_{\mathrm{router}}\)

1. **Shared**：一组共享更新；
2. **Crystal system**：按入选子集实际出现的晶系路由；
3. **Point group**：按实际出现的点群路由；
4. **Primary symmetry**：由 3.2.1 冻结为空间群或晶系，对 \(N_{\mathrm{primary}}\) 个保留类别建立差分；
5. **Learned-\(K\)**：令 \(K=N_{\mathrm{primary}}\)，使用 top-1 可学习路由。

另加一个不计入主轴但不可缺少的 **Random-\(K\)** 对照：保持与最终主分类相同的类别频率，把 parent 随机分到 \(K\) 组。它与 Primary-symmetry experts 有相同参数量和每专家样本量。

Learned router 使用构型级旋转不变量。预实验采用冻结基础 MACE 提取并缓存的 pooled \(0e\) 特征，辅以原子数、体积/原子等晶胞不变量；router 不读取标签。先用 top-1 routing 和负载均衡损失，避免 top-\(k\) 混合引入额外计算变量。

### 4.4 Expert 参数存储

共享骨干只加载一次。对目标 \(t=\mathrm{EF}\) 和专家 \(g\)：

\[
W_{\ell,g}=W_{\ell,0}+\Delta W_{\ell,g}.
\]

- Dense expert 保存 scope 内的 \(\Delta W_{\ell,g}\)；
- ELoRA expert 保存 \(A_{\ell,g},B_{\ell,g}\)；
- 一个 batch 先按 expert id 分组，再对同一共享模块选择对应 delta；
- 不实例化 \(K\) 个完整 MACE；
- checkpoint 保存 shared state、readout、router 和 expert delta bank；
- 训练时验证某个 expert batch 只更新对应 delta slice。

### 4.5 参数预算对照

每项结果同时给出：

- shared 参数量；
- 总差分参数量；
- 单样本激活参数量；
- optimizer-state 内存；
- FLOPs、吞吐和峰值显存。

主比较采用相同**单样本激活 rank/参数量**。对最终优胜配置再增加：

1. 总参数匹配：调节 expert rank，使所有专家差分总量接近 Shared-ELoRA rank 16；
2. 宽共享 adapter：增加 Shared rank，使其总差分参数接近所有专家之和；
3. Random-\(K\)：保持专家总参数和样本碎片化完全一致。

只有 Space-group 优于这些容量对照，才支持“对称划分”而不只是“更多参数”。

## 5. 仓库改造清单

预实验前需在 ELoRA 仓库增加统一配置层。

### 5.1 建议参数

| 参数 | 取值 |
|---|---|
| update_mode | dense / elora_clean / elora_paper |
| scope | readout / tail_1 / no_first / full |
| router | shared / crystal / point / space / learned / random |
| num_experts | 从训练集映射表读取 |
| elora_rank | 4 / 8 / 16 / 32 |
| elora_alpha | 默认等于 rank |
| expert_map | 类别到连续 expert id 的冻结映射 |
| trainable_manifest | 输出实际可训练参数清单 |

### 5.2 必要代码修改

1. 将 symmetric_contraction.py 中硬编码的 rank/alpha 改为构造参数；
2. 把单组 LoRA 因子改成带 expert 维度的 parameter bank；
3. 把 train.py 中基于字符串的 requires-grad 逻辑替换为显式 scope/update policy；
4. 让 paper-compatible 与 clean ELoRA 分离；
5. 修正或禁用未经验证的 merge_LoRA 保存路径；
6. 在数据对象中加入 parent_id、crystal_system、point_group、space_group 和 expert_id；
7. 增加按 expert 分组的 batch sampler；
8. Learned router 先使用冻结缓存特征，避免路由与专家前向形成循环依赖；
9. 统一输出 config、split hash、参数 manifest、数据索引和环境版本。

### 5.3 必过单元测试

- **零增量一致性**：初始化时 ELoRA 输出与原 checkpoint 一致；
- **梯度白名单**：每个 \(S\times P\) 只产生允许参数的梯度；
- **专家隔离**：expert \(g\) 的 batch 不改变 \(g'\ne g\) 的 delta；
- **共享唯一性**：共享权重只存在一个 Parameter 对象和一份显存；
- **混合 batch 一致性**：混合路由 batch 与逐专家子 batch 拼接结果一致；
- **等变性**：随机旋转/反射后 energy invariant、force equivariant；
- **merge 一致性**：若启用 merge，合并前后能量和力在容差内相同；
- **保存恢复**：shared + router + delta bank 重新加载后输出完全一致；
- **参数统计**：requires-grad 数、非零梯度数和 optimizer 参数数相互一致。

## 6. 训练协议

### 6.1 基础超参数

先采用论文 MACE-MP fine-tune 设置：

| 项目 | 值 |
|---|---:|
| foundation | MACE-MP small/论文对应 checkpoint |
| \(r_{\max}\) | 6.0 Å |
| energy weight | 1 |
| force weight | 1000 |
| optimizer | Adam/仓库论文配置 |
| base learning rate | 0.005 |
| weight decay | \(10^{-8}\) |
| EMA decay | 0.995 |
| gradient clip | 100 |
| ELoRA rank | 16 |
| dtype | float32 |

Dense 与 ELoRA-clean 分别在 Shared-T1、中等数据档位用

\[
\eta\in\{10^{-3},5\times10^{-3},10^{-2}\}
\]

选择学习率；选定后，同一 update 的所有 router 共用该学习率。不得为每个 expert 方法单独扩大调参预算。

### 6.2 训练预算

- batch size 固定；
- 各数据量使用相同最大 optimizer steps，而不是相同 epochs；
- 每 500 steps 验证一次；
- early stopping 规则对全部方法一致；
- 以固定 validation objective 选择 checkpoint；
- 3 个随机种子使用相同初始化种子表、split 和嵌套样本索引；
- router 额外 warm-up 若必要，必须同样计入总优化步数。

建议预实验上限为 50,000 optimizer steps；若轨道 A 表明 20,000 steps 已稳定收敛，可统一缩短。

## 7. 分阶段实验矩阵

完整笛卡尔积过大，采用门控式执行。

### Stage 0：环境与论文复现

1. 创建独立环境，满足仓库声明的 e3nn 0.4.4；使用 GPU PyTorch；
2. 运行 zero-delta、梯度和等变测试；
3. 在 rMD17-aspirin 上运行 Shared-Dense 与 ELoRA-paper-compatible；
4. 检查 \(N=50,200,1000\) 时 ELoRA 优势是否随数据增加而缩小；
5. 若趋势与论文相反，先停止专家实验并排查数据 split、单位、\(E_0\)、checkpoint 和解冻参数。

### Stage 1：clean ELoRA 与 scope 单种子筛选

固定 Shared router 和中等数据档 \(n_{\mathrm{train/SG}}=100\)：

- D-Dense；
- T1/NF/F × Dense；
- T1/NF/F × ELoRA-clean。

先用一个种子排除明显欠拟合、爆炸或重复 scope。保留 Dense 和 ELoRA 各自验证集 Pareto 前沿的 1–2 个 scope。

### Stage 2：router 单种子筛选

在 Stage 1 入选 scope 上固定 \(n_{\mathrm{train/class}}=100\)，比较：

- Shared；
- Crystal-system；
- Point-group；
- Primary-symmetry（Space-group 或 Crystal-system，由数据门控冻结）；
- Learned-\(K\)；
- Random-\(K\)。

主要观察 macro force RMSE、最差主分类类别误差和专家负载。若 Primary-symmetry 连单种子都不优于 Shared/Random，则不立即扩大到完整种子，而先检查每类数据、router 和参数更新隔离。

### Stage 3：核心确认实验

对 Dense 与 ELoRA-clean 各保留一个 scope，运行：

\[
P\in\{\mathrm{Dense},\mathrm{ELoRA}\},
\quad
R\in\{\mathrm{Shared},\mathrm{Primary},\mathrm{Learned},\mathrm{Random}\},
\]

\[
N_{\mathrm{train/class}}\in\{25,50,100,200\},
\quad
\mathrm{seed}\in\{1,2,3\}.
\]

最大为 \(2\times4\times4\times3=96\) 次训练。非主分类的其他对称粒度只在 Stage 2 有信号且预算允许时进入额外完整曲线。

### Stage 4：容量与机制消融

只对 Stage 3 中表现最好的专家方案执行：

- rank \(\{4,8,16,32\}\)；
- 总参数匹配；
- 宽 Shared adapter；
- router 去负载均衡；
- parent symmetry 与 instantaneous symmetry；
- composition 内随机标签；
- chemistry-disjoint test；
- stable rank 分析。

## 8. 指标与统计

### 8.1 主指标

主指标为按最终主分类类别（空间群或晶系）宏平均的 force RMSE：

\[
\mathrm{MacroF}
=\frac{1}{N_{\mathrm{SG}}}
\sum_g\mathrm{RMSE}_{F,g}.
\]

次指标包括：

- energy RMSE（meV/atom）；
- force MAE；
- micro average；
- 最差主分类类别误差；
- 各主分类类别误差方差；
- 训练时间、吞吐、峰值显存和参数量；
- 随机旋转下 energy/force 等变误差；
- learned router 利用率、熵和塌缩率。

### 8.2 核心对比

对每个固定的 \(S,P,N\) 计算：

\[
G_{\mathrm{sym}}
=\mathrm{MacroF}_{\mathrm{Shared}}
-\mathrm{MacroF}_{\mathrm{Space}},
\]

\[
G_{\mathrm{label}}
=\mathrm{MacroF}_{\mathrm{Random}}
-\mathrm{MacroF}_{\mathrm{Space}}.
\]

对每个固定的 \(S,R,N\) 计算：

\[
G_{\mathrm{ELoRA}}
=\mathrm{MacroF}_{\mathrm{Dense}}
-\mathrm{MacroF}_{\mathrm{ELoRA}}.
\]

所有差值使用同一 split、相同嵌套数据和相同 seed 配对。对 test parent 做分层 bootstrap，给出 95% 置信区间。

### 8.3 四轴规律

用学习曲线

\[
\mathrm{error}(N)=aN^{-b}+c
\]

分别拟合 Shared、Space、Learned 和 Random。重点报告：

- 小数据区的 expert 增益是否为负；
- 从何种每类样本量开始转正；
- 增益何时平台化；
- ELoRA 是否把转正阈值推向更小的 \(N\)；
- 更深 scope 是否只在大数据区受益；
- Learned router 是否收敛到与晶体学标签相关但不完全相同的分区。

可用含 \(S,P,R,\log N\) 及交互项的回归/混合效应模型做探索性汇总，但不以单个 \(p\)-value 替代配对学习曲线。

## 9. 预注册成功标准

称为“专家训练相对直接训练有初步提升”至少需要：

1. Primary-symmetry expert（空间群或晶系）在相同 \(S,P,N\) 下相对 Shared 的 MacroF 改善至少 5%；
2. 改善出现在至少两个相邻数据档位；
3. 三种子配对均值方向一致，bootstrap 95% CI 不跨 0；
4. Primary-symmetry 优于 Random-\(K\)，否则只能归因于数据分治/额外容量；
5. 等变误差不高于 Shared 的数值容差；
6. 改善不是由某一个主分类类别独占，最差类别误差不显著恶化。

若只满足 1–2 条，记为“值得扩大实验的信号”，不写成肯定结论。

## 10. 失败模式与处理

| 现象 | 可能原因 | 处理 |
|---|---|---|
| ELoRA 复现失败 | 环境、split、单位、非纯参数冻结 | 先完成轨道 A 审计，不进入 router |
| 所有构型被判为 \(P1\) | 非平衡构型瞬时对称性破缺 | 恢复 parent label，或改用标签稳定子集 |
| 专家优于 Shared 但不优于 Random | 总容量或样本分治效应 | 不归因于晶体对称性 |
| 小数据专家更差 | 每专家样本不足 | 保留为 H3 的阈值区，不人为删除 |
| Learned router 塌缩 | router 输入弱或负载失衡 | warm-up、均衡损失，报告而非隐藏 |
| Dense full 明显过拟合 | scope 太深、数据小 | 由学习曲线解释，不单独加大正则偏袒 Dense |
| ELoRA full 与 T1 无差异 | 当前实现只适配少数 product 权重 | 核查 trainable manifest 和插入覆盖 |
| 合并模型前后不一致 | merge_LoRA 实现错误 | 正式评测使用未合并模型，修复后再测 |

## 11. 输出文件与记录

每次训练至少保存：

- 完整配置；
- Git commit 和环境锁文件；
- 数据源版本、结构 id、split hash、空间群映射；
- trainable parameter manifest；
- 每 epoch/step 的 train/validation 指标；
- test 的逐结构预测；
- router 分配与专家利用率；
- 参数量、运行时间和显存；
- 最佳 checkpoint 与未合并 delta checkpoint。

最终预实验报告应包含：

1. 论文复现表；
2. scope × update 热图；
3. router 对比表；
4. 四条数据学习曲线；
5. Space vs Random 的配对误差图；
6. 参数—精度 Pareto 图；
7. 失败实验和停止条件记录。

## 12. Codex Goal 执行入口

本文件是科学问题、实验矩阵和预注册标准的总规范，不建议原样作为单个 Codex Goal。实际执行拆分为两个有独立验收与停止条件的目标：

1. [Goal 0：ELoRA 对称专家实现与就绪性验证](goal_0_elora_readiness.md)：完成配置、专家差分 bank、强制数据统计、测试与 CPU smoke，不运行正式矩阵。
2. [Goal 1：ELoRA 对称专家正式预实验](goal_1_elora_preexperiment.md)：在数据、模型、计算资源和 Goal 0 就绪后，冻结统计与 split，执行预注册矩阵并形成结果报告。

统一入口与启动前必填项见 [ELoRA 对称专家预实验：Codex Goal 入口](codex_goals.md)。Goal 1 的完成不以获得正向结果为条件；完整、可复现的阴性结果同样视为科学任务完成。

## 参考

[1] Wang, H. et al. ELoRA: Low-Rank Adaptation for Equivariant GNNs. ICML 2025. [本地论文](../ELoRA/paper.pdf)

[2] Barroso-Luque, L. et al. Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models. 2024. https://arxiv.org/abs/2410.12771

[3] FAIR Chemistry. OMat24 official dataset card. https://huggingface.co/datasets/facebook/OMAT24
