# 第 4 部分 / 第 6 部分：模型架构设计 (二)：混合 CNN-LSTM、DLinear 移动平均时序分解与 PatchTST 通道独立 Patch Transformer 深度解析
## Section 4.4--4.5 / 6.0: Model Architectures Part 2: Hybrid CNN-LSTM, DLinear Trend-Seasonal Decomposition, and PatchTST Channel-Independent Patch Transformer Detailed Mechanics

> **编写负责人**：学生 1 (Student 1, Student ID: 100001)  
> **对应代码实现**：  
> - `my_report/src/models/hybrid_models.py` (`CNNLSTMModel`)  
> - `my_report/src/models/modern_models.py` (`DLinearModel`, `SeriesDecomp`, `MovingAvgBlock`)  
> - `my_report/src/models/modern_models.py` (`PatchTSTModel`)  
> **实验元数据依据**：`my_report/experiments.db` (SQLite 数据库实测记录)

---

### 4.4 / 6.1 架构演进背景与归纳偏置互补性分析 (Architectural Evolution & Inductive Bias Complementarity)

在时间序列多步预测领域，传统的单一模型架构往往受制于其内生归纳偏置（Inductive Bias）的结构性短板：
1. **循环神经网络（LSTM / GRU）的表达瓶颈**：通过循环隐藏状态迭代维持历史记忆，擅长捕捉全局动力学演化与宏观趋势；然而在处理高采样频率（如 10 分钟级光伏）或强噪声序列时，高频突变与随机抖动会持续冲击门控单元，极易引发隐藏状态的扰动与长程梯度衰减。此外，将全部历史压缩至唯独 64 维的终态向量形成了显著的信息单点瓶颈。
2. **纯一维卷积神经网络（1D-CNN）的结构局限**：通过局部时间卷积核提取多尺度波形模式，具备出色的局部特征抽取与抗噪平滑能力；但要捕获覆盖数十乃至上百时间步的宏观长程依赖，必须依赖多层卷积深堆叠或扩张卷积（Dilated Convolution），这会导致参数量急剧膨胀，且扁平化全连接投影头难以适应可变历史长度。
3. **传统注意力机制（Vanilla Transformer）的三大失效机理**：直接将自然语言处理中的单点自注意力机制套用于时序数据，存在三大本质缺陷：
   - **单点语义缺失**：单点标量缺乏明确时序语义，且易受孤立异常点与高频白噪声污染；
   - **时间顺序破坏**：自注意力的置换不变性（Permutation Invariance）削弱了时间因果单调性；
   - **计算与显存开销沉重**：全点级注意力计算复杂度随历史序列长度呈现平方级增长 $\mathcal{O}(L^2)$，在长序列（如光伏 $L=288$）下计算与内存开销剧增。

针对上述问题，本研究在基准网络之上引入了两大进阶设计范式：
- **范式一：特征分工与级联协同（CNN–LSTM）**  
  通过前置一维卷积实现“高频局部滤波与波形特征提取”，后置 LSTM 实现“宏观时序演变状态追踪”，形成归纳偏置的互补闭环。
- **范式二：前沿现代时序解构（DLinear 与 PatchTST）**  
  - **DLinear (AAAI 2023)**：回归经典时序分析的本质，通过移动平均将序列显式解耦为趋势项（Trend）与季节残差项（Seasonal），并利用极简单层线性映射替代复杂的非线性变换，以反直觉的极轻量参数确立强周期性时序的精度与效率基准。
  - **PatchTST (ICLR 2023)**：引入“通道独立性（Channel Independence, CI）”与“时间切块（Patching）”，将时序序列重构为富含局部模式的切块 Token，将自注意力复杂度从 $\mathcal{O}(L^2)$ 降维至 $\mathcal{O}((L/S)^2)$，并在批次维度实现跨通道参数共享与数据增强。

---

### 4.5 / 6.2 级联混合网络 CNN–LSTM 架构设计与数学机理 (Hybrid Architecture: CNN-LSTM)

#### 4.5.1 / 6.2.1 设计假设与协同机制
CNN–LSTM 级联网络的核心设计假设在于**频域与时域特征的分层解耦**。在实际工业电力系统中：
- 电力用电负荷受到瞬态大功率用电设备启停的局部冲击，光伏发电受到大气中随机云团短暂遮挡的局部突变。这类微观波动在时域上持续时间短（1~3 个时间步）、局部导数变化剧烈。
- 同时，两者均受到昼夜 24 小时人类作息与地球自转天文几何的强宏观日周期控制。

因此，CNN–LSTM 将预测任务解耦为两个阶段：
1. **阶段 1（局部感受野滤波）**：利用核尺寸较小（$k=3$）的时序卷积核在局部滑动，充当可学习的带通/低通滤波器与局部特征提取器，消除高频白噪声，提炼出局部波形斜率与峰谷特征；
2. **阶段 2（长程因果状态记忆）**：LSTM 不再直接接收高噪声的原始标量时序，而是接收经由卷积提炼后的高级局部特征向量序列，进而专注建模跨时间步的长程宏观趋势。

```
CNN-LSTM 数据流架构示意图：
输入张量 X: [B, L, C] 
     │
     ▼ Transpose
[B, C, L]
     │
     ▼ Conv1D (k=3, pad=1, out=64) ──► 局部波形模式提取与滤波
     ▼ BatchNorm1d & ReLU
[B, 64, L]
     │
     ▼ Transpose
[B, L, 64]
     │
     ▼ LSTM (hidden=64, 1-layer)   ──► 宏观长程时序演变状态追踪
     │
   h_L: [B, 64] (最终时间步隐状态)
     │
     ▼ Linear (64 -> H * C)        ──► 直接多步非自回归全连接投影头
[B, H * C]
     │
     ▼ Reshape
输出预测 Y_hat: [B, H, C]
```

#### 4.5.2 / 6.2.2 严密数学形式化推导
设输入多通道时序张量为 $\mathbf{X} \in \mathbb{R}^{B \times L \times C}$，其中 $B$ 为 Batch 大小，$L$ 为历史观测步长（Electricity 为 48，Solar 为 288），$C=16$ 为通道数。

**第一步：通道置换与时序一维卷积**  
将输入张量转置为适合一维卷积的时序在后维度：
$$\mathbf{X}_{\text{conv}} = \mathbf{X}^\top \in \mathbb{R}^{B \times C \times L}$$
应用一维卷积层提取局部特征，卷积核尺寸设为 $k=3$，填充大小设为 $p = \lfloor k/2 \rfloor = 1$ 以保持时序长度严格对齐：
$$\mathbf{Z}_{\text{conv}}^{(i, \tau)} = \sum_{c=1}^C \sum_{j=-1}^1 \mathbf{W}_c^{(i, c, j+1)} \mathbf{X}_{\text{conv}}^{(b, c, \tau + j)} + \mathbf{b}_c^{(i)}, \quad i \in \{1, \dots, d_{\text{conv}}\}$$
随后通过批归一化（Batch Normalization）稳定中间特征分布，并通过 ReLU 引入非线性表达能力：
$$\mathbf{Z} = \text{ReLU}\left(\text{BatchNorm1d}\left(\mathbf{Z}_{\text{conv}}\right)\right) \in \mathbb{R}^{B \times d_{\text{conv}} \times L}$$
其中卷积特征维度 $d_{\text{conv}} = 64$。

**第二步：时序转置与 LSTM 编码**  
将提取的局部特征映射转置回时序在前格式：
$$\mathbf{Z}_{\text{lstm}} = \mathbf{Z}^\top \in \mathbb{R}^{B \times L \times d_{\text{conv}}}$$
将其输入单层 LSTM 循环单元（$d_{\text{lstm}} = 64$）。在每个时间步 $\tau \in \{1, \dots, L\}$，执行标准门控更新方程：
$$\begin{aligned}
\mathbf{f}_\tau &= \sigma\left(\mathbf{W}_f \mathbf{z}_\tau + \mathbf{U}_f \mathbf{h}_{\tau-1} + \mathbf{b}_f\right) \quad &\text{(遗忘门：决定舍弃的旧信息)} \\
\mathbf{i}_\tau &= \sigma\left(\mathbf{W}_i \mathbf{z}_\tau + \mathbf{U}_i \mathbf{h}_{\tau-1} + \mathbf{b}_i\right) \quad &\text{(输入门：决定写入的新信息)} \\
\tilde{\mathbf{c}}_\tau &= \tanh\left(\mathbf{W}_c \mathbf{z}_\tau + \mathbf{U}_c \mathbf{h}_{\tau-1} + \mathbf{b}_c\right) \quad &\text{(候选细胞状态)} \\
\mathbf{c}_\tau &= \mathbf{f}_\tau \odot \mathbf{c}_{\tau-1} + \mathbf{i}_\tau \odot \tilde{\mathbf{c}}_\tau \quad &\text{(细胞状态长程累积更新)} \\
\mathbf{o}_\tau &= \sigma\left(\mathbf{W}_o \mathbf{z}_\tau + \mathbf{U}_o \mathbf{h}_{\tau-1} + \mathbf{b}_o\right) \quad &\text{(输出门：决定隐状态输出)} \\
\mathbf{h}_\tau &= \mathbf{o}_\tau \odot \tanh\left(\mathbf{c}_\tau\right) \quad &\text{(当前时间步隐状态向量)}
\end{aligned}$$
其中 $\mathbf{h}_\tau \in \mathbb{R}^{d_{\text{lstm}}}$。模型读取最后一个历史时间步的隐藏状态 $\mathbf{h}_L \in \mathbb{R}^{B \times d_{\text{lstm}}}$，该向量高度凝练了整个 $L$ 步历史时序的上下文动态。

**第三步：多步直接预测投影头 (Direct Multi-Step Linear Head)**  
区别于自回归（Auto-Regressive）模型在测试阶段需逐步生成并回填导致的误差指数级累积，本模型采用直接多步映射机制（Direct Multi-Step Forecasting）。将 $\mathbf{h}_L$ 输入全连接投影矩阵：
$$\hat{\mathbf{Y}}_{\text{flat}} = \mathbf{h}_L \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \in \mathbb{R}^{B \times (H \cdot C)}$$
其中 $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{lstm}} \times (H \cdot C)}$，$\mathbf{b}_{\text{out}} \in \mathbb{R}^{H \cdot C}$。最终重构恢复物理输出张量：
$$\hat{\mathbf{Y}} = \text{Reshape}\left(\hat{\mathbf{Y}}_{\text{flat}}, [B, H, C]\right) \in \mathbb{R}^{B \times H \times C}$$

#### 4.5.3 / 6.2.3 代码映射与参数复杂度剖析
在代码库 `my_report/src/models/hybrid_models.py` 中，`CNNLSTMModel` 完整实现了上述逻辑。其可训练参数量分布如下：
- **Conv1D 层**：权重参数 $64 \times 16 \times 3 = 3,072$，偏置参数 $64$，共计 $3,136$。
- **BatchNorm1d 层**：缩放与平移可学习参数 $64 \times 2 = 128$。
- **LSTM 层**：包含输入到隐层权重 $\mathbf{W}$ 与隐层到隐层权重 $\mathbf{U}$ 及偏置，单层参数量为 $4 \times (64 \times 64 + 64 \times 64 + 64 + 64) = 33,280$。
- **线性输出投影头**：
  - 在 Electricity 任务（$H=24, C=16$）：输入 64，输出 $384$，权重参数 $64 \times 384 = 24,576$，偏置 $384$，输出头参数量为 $24,960$。**全模型总参数量为 61,504**。
  - 在 Solar 任务（$H=144, C=16$）：输出维度扩展至 $144 \times 16 = 2,304$，输出头参数量增加至 $64 \times 2304 + 2304 = 149,760$。**全模型总参数量为 186,304**。

**架构对比优势**：
与纯一维卷积网络 1D-CNN 相比（Electricity 参数量高达 401,568，Solar 高达 2,369,568），CNN–LSTM 借助 LSTM 对时序长度 $L$ 的时间压缩特性，通过读取定长终态向量 $\mathbf{h}_L$ 进行多步映射，避免了 1D-CNN 中随历史序列铺开的庞大 Flatten 全连接层，**在 Electricity 任务中参数量缩减了 84.7%，在 Solar 任务中参数量缩减了 92.1%**！同时在晚间负荷尖峰切片（Peak Slice）实测中，CNN–LSTM 取得 2.7072 MAE，优于纯 LSTM（2.8508）与纯 1D-CNN（2.7218），充分证实了前端波形滤波的抗冲击价值。

---

### 4.6 / 6.3 经典时序分解线性模型 DLinear (AAAI 2023)

#### 4.6.1 / 6.3.1 理论动机：挑战复杂注意力的极简线性基准
在 AAAI 2023 的开创性论文《Are Transformers Effective for Time Series?》中，Zeng 等人深刻反思了深度学习时序领域的复杂化倾向。论文指出：
1. **时序序列的时间连续性与单向不可逆性**：自然语言中的词语语义高度依赖双向上下文与动态句法结构；而物理时间序列具有严格单向流逝的时间因果关系与强规律的物理周期动力学。
2. **多头注意力的盲目套用**：在强周期性时序上，注意力矩阵计算出的点对点权重往往退化为简单的周期性距离衰减；反之，若时序中存在非平稳趋势漂移（Temporal Drift），自注意力机制甚至会产生注意力发散。
3. **经典时序分解（Seasonal-Trend Decomposition）的有效性**：时序预测的核心挑战在于分离“长期平滑演变的大趋势”与“均值为零、围绕趋势周期震荡的季节性微观波动”。

DLinear 继承了经典 STL 分解（Seasonal and Trend decomposition using Loess）的核心思想，但将其极致精简为：**一层可导移动平均池化 + 两组独立的一维时间全连接层**。

```
DLinear 架构数据流示意图：
输入张量 X: [B, L, C]
     │
     ├───► Moving Average (k=25, 端点对称复制填充) ──► 趋势分量 X_trend: [B, L, C]
     │                                                     │
     └───► 残差减法 (X - X_trend) ─────────────────► 季节分量 X_seasonal: [B, L, C]
                                                           │
          ┌────────────────────────────────────────────────┘
          │
          ├── X_trend    ──► Transpose [B, C, L] ──► Linear_Trend (L -> H)    ──► Y_trend: [B, C, H]
          │
          └── X_seasonal ──► Transpose [B, C, L] ──► Linear_Seasonal (L -> H) ──► Y_seasonal: [B, C, H]
                                                                                      │
                                                                   相加重构 ◄─────────┘
                                                                       │
                                                       Y_hat = Y_trend + Y_seasonal
                                                                       │
                                                       Transpose [B, H, C]
```

#### 4.6.2 / 6.3.2 移动平均分解与边界闭式加权及防泄漏证明
设输入序列为 $\mathbf{X} \in \mathbb{R}^{B \times L \times C}$。

**第一步：移动平均趋势项提取 (Moving Average Trend Extraction)**  
采用窗口尺寸为 $k_{\text{ma}} = 25$（奇数）的一维平均池化（AvgPool1d）提取长期趋势。为了使卷积后的趋势项长度与输入历史序列 $L$ 严格保持一致，且杜绝任何未来信息泄漏，必须在历史序列两端执行端点对称复制填充（Replication Padding）：
- 前端填充长度：$p_{\text{front}} = \lfloor (k_{\text{ma}} - 1) / 2 \rfloor = 12$；填充值为历史窗口首个采样点 $\mathbf{X}[:, 0, :]$ 的静态复制；
- 尾端填充长度：$p_{\text{end}} = k_{\text{ma}} - 1 - p_{\text{front}} = 12$；填充值为历史窗口最后一个采样点 $\mathbf{X}[:, L-1, :]$ 的静态复制。

在填充后的张量 $\mathbf{X}_{\text{padded}} \in \mathbb{R}^{B \times C \times (L + 24)}$ 上执行步长为 1 的无偏滑动平均：
$$\mathbf{X}_{\text{trend}}(b, c, t) = \frac{1}{k_{\text{ma}}} \sum_{j=-12}^{12} \mathbf{X}_{\text{padded}}(b, c, t + 12 + j)$$

**边界闭式表达与抗泄漏数学证明**：  
在历史序列的最末时间步 $t = L-1$（即最接近预测起点的关键点），由于未来 $j \in \{1, \dots, 12\}$ 的时步被严格对称复制填充为已观测值 $\mathbf{X}[:, L-1, :]$，该点提取出的趋势项具有严格闭式加权形式：
$$\mathbf{X}_{\text{trend}}(b, c, L-1) = \frac{1}{25} \left( \sum_{j=-12}^{0} \mathbf{X}(b, c, L-1 + j) + 12 \cdot \mathbf{X}(b, c, L-1) \right)$$
这一机制赋予了末端趋势 $48\%$ 的瞬态观测权重与 $52\%$ 的历史 12 步移动平滑权重，不仅实现了指数平滑的动态跟踪效果，且整个池化滤波器的感受野范围严格受限于历史闭区间 $(-\infty, L-1]$，**绝无任何来自预测区间 $[L, L+H)$ 的未来数据参与运算**，彻底符合时间因果物理约束。

**第二步：季节残差项解耦 (Seasonal Residual Extraction)**  
季节残差项代表去除低频基线后的零均值周期性波动信号：
$$\mathbf{X}_{\text{seasonal}} = \mathbf{X} - \mathbf{X}_{\text{trend}} \in \mathbb{R}^{B \times L \times C}$$

**第三步：双分支独立线性时序映射**  
将趋势项与季节项沿通道维度独立处理。将两分量转置为 $[B, C, L]$ 形式，由两个专用的全连接层在时间步维度分别进行 $L \to H$ 的投影映射：
$$\hat{\mathbf{Y}}_{\text{trend}} = \mathbf{X}_{\text{trend}} \mathbf{W}_{\text{trend}}^\top + \mathbf{b}_{\text{trend}} \in \mathbb{R}^{B \times C \times H}$$
$$\hat{\mathbf{Y}}_{\text{seasonal}} = \mathbf{X}_{\text{seasonal}} \mathbf{W}_{\text{seasonal}}^\top + \mathbf{b}_{\text{seasonal}} \in \mathbb{R}^{B \times C \times H}$$
其中：
- $\mathbf{W}_{\text{trend}} \in \mathbb{R}^{H \times L}$ 学习宏观趋势的外推与衰减规律；
- $\mathbf{W}_{\text{seasonal}} \in \mathbb{R}^{H \times L}$ 学习日周期、峰谷震荡在未来时域的周期性延拓。

**第四步：重构合成输出**  
$$\hat{\mathbf{Y}}_{\text{trans}} = \hat{\mathbf{Y}}_{\text{trend}} + \hat{\mathbf{Y}}_{\text{seasonal}} \in \mathbb{R}^{B \times C \times H}$$
最后转置为标准多通道预测输出：
$$\hat{\mathbf{Y}} = \hat{\mathbf{Y}}_{\text{trans}}^\top \in \mathbb{R}^{B \times H \times C}$$

#### 4.6.3 / 6.3.3 通道共享机制与极致参数效率分析
在 DLinear 原始论文中，模型提供了通道独立（`individual=True`，即每个通道拥有一组独立的映射参数）与通道共享（`individual=False`）两种配置。在本项目生产实现 `my_report/src/models/modern_models.py` 中，我们采用标准**通道共享权重（Shared Channel Weights）**模式：
- 线性季节层参数量：$L \times H + H$
- 线性趋势层参数量：$L \times H + H$
- **模型总参数量**：$2 \times (L \times H + H)$

在实际实验中，参数量呈现出令人惊叹的极简特性：
- **Electricity 数据集**（$L=48, H=24$）：
  $$\text{Params} = 2 \times (48 \times 24 + 24) = \mathbf{2,352}$$
- **Solar 数据集**（$L=288, H=144$）：
  $$\text{Params} = 2 \times (288 \times 144 + 144) = \mathbf{83,232}$$

**实证价值与帕累托前沿分析**：
在实测对比中，DLinear 凭借仅 2,352 个参数，在电力负荷预测上取得了 2.7894 MAE 和 0.4473 MASE，极度逼近参数量是其 32 倍的 PatchTST（0.4159）和 170 倍的 1D-CNN（0.4148）；在训练吞吐与延迟方面，单批次推理耗时仅 **0.97 ms**，全周期训练仅耗时 **1.34 秒**！
而在光伏活跃出力时段（Active Slice），DLinear 达到 12.2175 MAE，甚至超越了 Seasonal Naive 基线（12.6383）与 PatchTST（13.0294）。这证明：**对于具有明确物理周期分解特征的时序任务，极简线性分解不仅能杜绝过拟合，且构成了无可争议的工业边缘算力帕累托最优解**。

---

### 4.7 / 6.4 通道独立时间块 Transformer 模型 PatchTST (ICLR 2023)

#### 4.7.1 / 6.4.1 理论动机：突破单点注意力局限的现代化范式
PatchTST（Nie et al., ICLR 2023）是当前长时序预测领域的代表性前沿模型。它针对传统 Transformer 存在的信息局部性丢失与计算效率低下问题，提出了两大核心颠覆性创新：
1. **时间切块（Patching）**：借鉴计算机视觉 Vision Transformer (ViT) 的思想，将一段连续的局部时间子序列聚合成一个 Patch，作为自注意力模型处理的基本 Token。
   - **语义完整性**：单个时间步标量缺乏上下文，而一个包含 16 个采样点的 Patch 自然包含了局部的上升沿、下降沿、波峰波谷及局部波动率等宏观语义；
   - **注意力计算开销降维**：切块后序列长度由 $L$ 锐减至 $N \approx L/S$（步长为 $S$），注意力矩阵大小缩小为原来的 $(S/L)^2$，计算与显存开销从 $\mathcal{O}(L^2)$ 骤降至 $\mathcal{O}((L/S)^2)$。
2. **通道独立性（Channel Independence, CI）**：完全打破多元时序模型中“强行跨通道计算注意力”的传统思路，将每个变量通道视为独立的单变量时序，在 Batch 维度展开，共享同一个 Transformer 编码器。

```
PatchTST 架构数据流示意图：
输入张量 X: [B, L, C] 
     │
     ▼ Transpose & Reshape (通道独立 Channel Independence)
[B * C, L, 1] ──► 有效训练样本容量扩增至 B * C 倍
     │
     ▼ Replication Padding (历史末尾时间步复制补齐)
[B * C, L + pad, 1]
     │
     ▼ Unfold 切片操作 (patch_len=16, stride=8)
[B * C, num_patches, patch_len]
     │
     ▼ Linear Patch Embedding (16 -> d_model=64) + 可学习位置编码 (Learnable Pos Embedding)
[B * C, num_patches, 64]
     │
     ▼ Transformer Encoder (2层, 4头自注意力, d_ff=128, Dropout=0.1)
[B * C, num_patches, 64]
     │
     ▼ Flatten Head (展平所有 patch token: num_patches * 64)
[B * C, num_patches * 64]
     │
     ▼ Linear Projection Head (num_patches * 64 -> pred_len H)
[B * C, H]
     │
     ▼ Reshape & Transpose (通道维度恢复)
输出预测 Y_hat: [B, H, C]
```

#### 4.7.2 / 6.4.2 通道独立性 (Channel Independence) 数学表述与三大理论红利
设输入张量为 $\mathbf{X} \in \mathbb{R}^{B \times L \times C}$。

**通道独立张量变换**：
将输入张量转置并重塑，将物理通道维度 $C$ 完全并入批处理 Batch 维度：
$$\mathbf{X}_{\text{ci}} = \text{Reshape}\left(\mathbf{X}^\top, [B \cdot C, L, 1]\right) \in \mathbb{R}^{(B \cdot C) \times L \times 1}$$
此时，网络接收到的是 $B \cdot C$ 条独立的单通道时序片段。所有通道在整个前向与反向传播过程中**完全共享同一套 Transformer 权重参数**。

**三大理论红利深度推导**：
1. **数据容量等效倍增（Data Token Multiplier）**：  
   在深度学习优化中，样本稀缺往往导致高容量模型发生灾难性过拟合。在本项目中，$C=16$ 条通道全部共享网络，使得网络在每个 Epoch 中实际接收到的时序更新样本量扩大了 **16 倍**！  
   在后续敏感性实验 B 中，这一机制带来了惊人的样本效率：在仅使用 25% 训练数据时，PatchTST 的 MASE 仅从 0.4159 微降至 0.4295（降幅仅 3.3%），而单体通道混叠的 LSTM 则从 0.4209 剧烈退化至 0.9206（性能崩溃 118%）！这证明 CI 机制在数据稀疏场景下具备决定性的抗过拟合优势。
2. **阻断跨通道虚假相关与相移噪声（Spurious Correlation Decoupling）**：  
   在真实微电网或分布式光伏群中，不同用户用电具有不同的相位差，不同电站所处微气候也存在对流时滞。若采用多通道混合注意力，注意力机制极易拟合这些随机相移，产生“虚假因果关联”。CI 机制迫使模型专注学习时序信号内部的时间动力学因果，提高了跨场景泛化能力。
3. **架构可扩展性与即插即用（Arbitrary Channel Scalability）**：  
   传统全通道模型的输入输出层与通道数 $C$ 强绑定；而 PatchTST 的核心编码器参数与 $C$ 完全解耦。当电网新增量测用户或新增光伏电站时，无需重新设计或重训编码器，即可直接进行跨序列推理。

#### 4.7.3 / 6.4.3 时间块切分 (Patching) 与末端自适应补齐机制推导
**切块参数设定**：设定切块长度 $P = 16$，滑动步长 $S = 8$。

**覆盖性与末端复制补齐算法**：
为保证历史序列被无缝覆盖，并确保**最贴近未来预测起点的末端观测数据得到充分重叠编码**，我们推导了自适应补齐公式：
若序列未能被整数步长划分，即 $(L - P) \pmod S \neq 0$：
$$N_{\text{patch}} = \left\lfloor \frac{L - P}{S} \right\rfloor + 2$$
$$L_{\text{pad}} = (N_{\text{patch}} - 1) \cdot S + P - L$$
否则：
$$N_{\text{patch}} = \frac{L - P}{S} + 1, \quad L_{\text{pad}} = 0$$

当需要补齐时，对序列末尾执行时间步复制填充（Replication Padding）：
$$\mathbf{X}_{\text{pad}} = \left[ \mathbf{X}_{\text{ci}} \, ; \, \mathbf{X}_{\text{ci}}[:, L-1:, :].\text{repeat}(1, L_{\text{pad}}, 1) \right] \in \mathbb{R}^{(B \cdot C) \times (L + L_{\text{pad}}) \times 1}$$

**Unfold 切片生成**：
利用滑动窗口将补齐后的序列展开为二维 Patch 集合：
$$\mathbf{P} = \text{Unfold}\left(\mathbf{X}_{\text{pad}}, \text{size}=P, \text{step}=S\right) \in \mathbb{R}^{(B \cdot C) \times N_{\text{patch}} \times P}$$
每个 Patch $\mathbf{p}_i \in \mathbb{R}^P$ 封装了长度为 16 个时间步的微观波形。

**在本项目两大数据集上的规格核算**：
- **Electricity**（$L=48, P=16, S=8$）：
  $(48 - 16) / 8 + 1 = 5$，整除无填充，**$N_{\text{patch}} = 5$ 个 Tokens**。
  相比原序列 48 个点，自注意力矩阵维度由 $48 \times 48 = 2,304$ 骤降至 $5 \times 5 = \mathbf{25}$，**计算量压缩达 98.9%**！
- **Solar**（$L=288, P=16, S=8$）：
  $(288 - 16) / 8 + 1 = 35$，整除无填充，**$N_{\text{patch}} = 35$ 个 Tokens**。
  自注意力矩阵维度由 $288 \times 288 = 82,944$ 骤降至 $35 \times 35 = \mathbf{1,225}$，**计算量压缩达 98.5%**！

#### 4.7.4 / 6.4.4 可学习位置编码、Transformer 编码器与展平输出头
**第一步：Patch 嵌入与位置编码**  
通过一维线性投影层将每个 Patch 映射至模型隐层维度 $d_{\text{model}} = 64$：
$$\mathbf{E}_{\text{patch}} = \mathbf{P} \mathbf{W}_{\text{emb}} + \mathbf{b}_{\text{emb}} \in \mathbb{R}^{(B \cdot C) \times N_{\text{patch}} \times d_{\text{model}}}$$
叠加可学习的位置编码参数 $\mathbf{W}_{\text{pos}} \in \mathbb{R}^{1 \times N_{\text{patch}} \times d_{\text{model}}}$，以注入绝对时间顺序信息：
$$\mathbf{E}^{(0)} = \mathbf{E}_{\text{patch}} + \mathbf{W}_{\text{pos}} \in \mathbb{R}^{(B \cdot C) \times N_{\text{patch}} \times d_{\text{model}}}$$

**第二步：堆叠 Transformer 编码器**  
采用 $e_{\text{layers}} = 2$ 层的标准 Transformer 编码器。每层包含多头自注意力（$n_{\text{heads}} = 4$）与前馈神经网络（$d_{\text{ff}} = 128$），配合残差连接与 LayerNorm：
$$\begin{aligned}
\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i &= \mathbf{E}^{(l-1)} \mathbf{W}_i^Q, \, \mathbf{E}^{(l-1)} \mathbf{W}_i^K, \, \mathbf{E}^{(l-1)} \mathbf{W}_i^V \\
\text{Head}_i &= \text{Softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_k}}\right) \mathbf{V}_i \\
\mathbf{E}_{\text{attn}}^{(l)} &= \text{LayerNorm}\left(\mathbf{E}^{(l-1)} + \text{Concat}(\text{Head}_1, \dots, \text{Head}_4) \mathbf{W}^O\right) \\
\mathbf{E}^{(l)} &= \text{LayerNorm}\left(\mathbf{E}_{\text{attn}}^{(l)} + \text{FFN}\left(\mathbf{E}_{\text{attn}}^{(l)}\right)\right)
\end{aligned}$$
自注意力机制在 $N_{\text{patch}}$ 个 Token 之间计算相关性权重，实现跨时间块宏观模式依赖建模。

**第三步：展平预测投影头 (Flatten Prediction Head)**  
Transformer 最终输出为 $\mathbf{E}^{(2)} \in \mathbb{R}^{(B \cdot C) \times N_{\text{patch}} \times d_{\text{model}}}$。
将其在时间块与特征维度完全展平：
$$\mathbf{Z}_{\text{flat}} = \text{Reshape}\left(\mathbf{E}^{(2)}, [B \cdot C, N_{\text{patch}} \cdot d_{\text{model}}]\right)$$
通过单个线性层直接投影至未来预测时域步长 $H$：
$$\hat{\mathbf{Y}}_{\text{ci}} = \mathbf{Z}_{\text{flat}} \mathbf{W}_{\text{head}} + \mathbf{b}_{\text{head}} \in \mathbb{R}^{(B \cdot C) \times H}$$
最后，恢复原始批次与通道组织形式：
$$\hat{\mathbf{Y}} = \text{Transpose}\left(\text{Reshape}\left(\hat{\mathbf{Y}}_{\text{ci}}, [B, C, H]\right), 1, 2\right) \in \mathbb{R}^{B \times H \times C}$$

#### 4.7.5 / 6.4.5 参数量严密数学解析推导与长序列扩展性实证
在 `my_report/src/models/modern_models.py` 中，`PatchTSTModel` 的可训练参数量包含四个严谨解耦的模块：
1. **Patch 线性映射层 (`patch_embedding`)**：  
   将单通道 Patch 长度 $P=16$ 投影至隐层维度 $d_{\text{model}}=64$：
   $$P_{\text{patch\_emb}} = P \times d_{\text{model}} + d_{\text{model}} = 16 \times 64 + 64 = 1,088$$
2. **可学习绝对位置编码 (`pos_embedding`)**：  
   为每个时间块分配 $d_{\text{model}}$ 维可学习偏置，参数量为 $N_{\text{patch}} \times d_{\text{model}}$：
   - Electricity ($N_{\text{patch}}=5$)：$5 \times 64 = 320$
   - Solar ($N_{\text{patch}}=35$)：$35 \times 64 = 2,240$
   - 敏感性 168h 窗口 ($N_{\text{patch}}=20$)：$20 \times 64 = 1,280$
3. **两层 Transformer 编码器 (`transformer`)**：  
   每层编码器包含 4 头多头自注意力（$d_{\text{model}}=64$）、前馈网络（$d_{\text{ff}}=128$）与两组 LayerNorm：
   - 联合输入投影矩阵 `in_proj_weight` ($192 \times 64$) 与偏置 ($192$)：$12,288 + 192 = 12,480$
   - 自注意力输出投影矩阵 `out_proj.weight` ($64 \times 64$) 与偏置 ($64$)：$4,096 + 64 = 4,160$
   - 前馈层 1 `linear1.weight` ($128 \times 64$) 与偏置 ($128$)：$8,192 + 128 = 8,320$
   - 前馈层 2 `linear2.weight` ($64 \times 128$) 与偏置 ($64$)：$8,192 + 64 = 8,256$
   - 两个 LayerNorm 层缩放与平移参数：$2 \times (64 + 64) = 256$
   - 单层参数量精确小计：$12,480 + 4,160 + 8,320 + 8,256 + 256 = \mathbf{33,472}$
   - 两层 Transformer 总参数量：$2 \times 33,472 = \mathbf{66,944}$
4. **展平直接多步投影头 (`head`)**：  
   将展开的所有 Token 特征映射至未来预测步长 $H$：
   $$P_{\text{head}} = (N_{\text{patch}} \times d_{\text{model}}) \times H + H = (N_{\text{patch}} \times 64 + 1) \times H$$
   - Electricity ($N_{\text{patch}}=5, H=24$)：$(5 \times 64) \times 24 + 24 = 320 \times 24 + 24 = 7,680 + 24 = 7,704$
   - Solar ($N_{\text{patch}}=35, H=144$)：$(35 \times 64) \times 144 + 144 = 2,240 \times 144 + 144 = 322,560 + 144 = 322,704$
   - 敏感性 168h 窗口 ($N_{\text{patch}}=20, H=24$)：$(20 \times 64) \times 24 + 24 = 1,280 \times 24 + 24 = 30,720 + 24 = 30,744$

**全模型理论参数量与实测数据库完全严密闭合**：
- **Electricity 任务**：
  $$P_{\text{Total}}^{\text{Elec}} = 1,088 + 320 + 66,944 + 7,704 = \mathbf{76,056}$$
  理论推导值与 `my_report/experiments.db` 实测记录 **76,056** 100% 精确一致！
- **Solar 任务**：
  $$P_{\text{Total}}^{\text{Solar}} = 1,088 + 2,240 + 66,944 + 322,704 = \mathbf{392,976}$$
  理论推导值与 `my_report/experiments.db` 实测记录 **392,976** 100% 精确一致！
- **168h 历史窗口敏感性配置**：
  $$P_{\text{Total}}^{168\text{h}} = 1,088 + 1,280 + 66,944 + 30,744 = \mathbf{100,056}$$
  理论推导值与 `my_report/experiments.db` 实测记录 **100,056** 100% 精确一致！

**长序列扩展优势与自注意力动态**：
在敏感性实验 A 中，当将历史窗口由 24 小时（$L=24$）拓展至 168 小时（$L=168$，覆盖一整周）时，PatchTST 展现出极为显著的扩展潜力：**MASE 由 0.5053 大幅优化至 0.3969（误差下降 21.5%），夺得全场所有实验配置的绝对最优精度**！  
**数学机理解析**：
- 在极短历史窗口 $L=24$ 时，根据切分公式 $(24 - 16)/8 + 1 = 2$，全序列仅能生成 **2 个 Patch Tokens**。在仅有 2 个 Token 的极小序列上，4 头自注意力的相关性矩阵退化为简单的双元素互相关，多头表达能力严重受限；
- 而当历史拓展至 168 小时（$L=168$）时，生成 Token 数量增至 $(168 - 16)/8 + 1 = \mathbf{20}$ 个 Tokens。20 个时间块自注意力矩阵不仅将运算量相比点级序列压缩了 $98.58\%$（$20^2 = 400$ vs $168^2 = 28,224$），而且能够清晰建模前序工作日与周末之间的多尺度相位依赖，完全释放了现代 Transformer 的长程全局关联建模威力。

---

### 4.8 / 6.5 三大进阶时序网络综合横向对比与物理适配分析 (Comparative Synthesis)

为了清晰呈现各模型的理论机理与适用边界，下表系统性总结了 CNN–LSTM、DLinear 与 PatchTST 在数学原理、通道策略、参数规模、计算复杂度及物理适应性上的全景对比：

| 评估维度 | 级联混合网络 (CNN–LSTM) | 经典分解线性模型 (DLinear) | 通道独立 Patch Transformer (PatchTST) |
| :--- | :--- | :--- | :--- |
| **提出年份与会议** | 经典混合架构 | AAAI 2023 | ICLR 2023 |
| **核心归纳偏置** | 局部特征抽取 + 宏观因果记忆 | 趋势-季节项时序正交解耦 | 局部波形语义切块 + 跨块长程全局自注意力 |
| **时序局部建模手段** | 1D-CNN (卷积核 $k=3$) 局部滤波 | 移动平均池化 (核大小 25) 提取趋势项 | 时间切块 (Patching, 长度 16, 步长 8) |
| **长程依赖建模手段** | 单层 LSTM 循环门控更新 | 单层线性映射矩阵直接投影 | 2 层多头自注意力编码器 ($n=4$) |
| **通道处理范式** | 全通道混合卷积与循环映射 | 通道共享参数（按单通道独立映射） | **通道独立性 (Channel Independence, CI)** |
| **多通道样本倍增效应** | 无（常规 Batch 样本） | 无（常规 Batch 样本） | **有（等效样本量扩大 16 倍）** |
| **时间计算复杂度** | $\mathcal{O}(L \cdot d_{\text{conv}} + L \cdot d_{\text{lstm}}^2)$ | $\mathcal{O}(L \cdot H)$ 极度精简 | $\mathcal{O}((L/S)^2 \cdot d_{\text{model}})$ 规避平方爆炸 |
| **Electricity 参数量** | 61,504 | **2,352 (极简帕累托最优)** | 76,056 |
| **Solar 参数量** | 186,304 | **83,232 (极简帕累托最优)** | 392,976 |
| **单批次推理耗时 (ms)** | 1.81 ms | **0.97 ms (最快)** | 3.95 ms (Electricity) / 38.50 ms (Solar) |
| **少样本学习稳定性** | 较弱（25% 数据下有一定退化） | 较强（线性权重不易过拟合） | **极强（25% 数据下 MASE 仅降 3.3%）** |
| **超长历史拓展性** | 一般（受限于 LSTM 梯度衰减） | 较好（参数随 $L$ 线性增长） | **卓越（168h 窗口达全场最优 0.3969 MASE）** |
| **Electricity 物理适配度** | 优（尖峰切片 MAE 2.7072 表现出色） | 良好（MASE 0.4473，逼近深度模型） | **卓越（MASE 0.4159，长周期时序表现顶尖）** |
| **Solar 物理适配度** | 良好（过滤云层高频扰动） | **卓越（日间活跃切片 MAE 12.2175 最优）** | 良好（日间活跃切片 MAE 13.0294） |
| **工业工程落地推荐场景** | 侧重捕获负荷瞬态突变冲击的高负荷区 | **嵌入式电表、低成本微网控制器、边端部署** | **集中式云端调度中心、具有超长历史周数据的系统** |

通过上述深入剖析可见：
1. **DLinear** 证明了复杂神经网络在传统强周期物理时序中绝非不可替代，其以微秒级推理与极少参数确立了工程落地不可逾越的效率标杆；
2. **PatchTST** 展现了现代 Transformer 在解决语义切块与通道独立性之后，在超长历史挖掘与少样本泛化上的强大理论优势；
3. **CNN–LSTM** 则证明了经典卷积与循环门控在频域与时域特征分工上的有效性，在抵御局部突变冲击上依然保持着强劲的生命力。
