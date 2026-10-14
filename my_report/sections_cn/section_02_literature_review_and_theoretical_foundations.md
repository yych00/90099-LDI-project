# 第2章 全面文献综述与理论基础 (Literature Review & Theoretical Foundations)

> **编写责任人**：学生 1 & 学生 2（理论基础、时序模型分类学与领域物理特性协同撰写）  
> **核心研究范围**：时序多步预测范式形式化推导、经典统计学与启发式时序方法、循环神经网络与门控演进（LSTM/GRU）、时间卷积（1D-CNN）与混合级联架构（CNN--LSTM）、现代 Transformer 机制反思与 PatchTST 分块通道独立性、现代极简线性分解模型（DLinear）、能源系统特定物理动力学机理（电力负荷双峰与光伏昼夜硬约束）、评价指标理论对比与全景综合对比矩阵。

---

## 2.1 时序多步预测问题数学形式化与范式辨析

### 2.1.1 多步预测任务的数学形式化定义

在现代电力系统调度运行与可再生能源消纳场景中，时间序列预测本质上是在给定历史多通道连续观测数据的条件下，对未来离散时域内的系统状态轨迹进行联合概率分布估计或点预测（Godahewa et al., 2021; Box et al., 2015）。

设系统由 $C$ 个同步采集的物理量测通道（如 $C$ 个变电站用户节点，或 $C$ 座集中式光伏电站）组成，离散时间步索引记为 $t \in \{1, 2, \dots, T\}$。在第 $t$ 个采样时刻，系统的多通道瞬时观测向量记为：
$$\mathbf{x}_t = [x_{t, 1}, x_{t, 2}, \dots, x_{t, C}]^\top \in \mathbb{R}^C$$

连续 $t$ 个时间步构成的历史全集矩阵记为 $\mathbf{X}_{1:t} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_t]^\top \in \mathbb{R}^{t \times C}$。

在工业预测与实际计算约束下，模型通常仅观测预测决策起点（Forecast Origin）$t$ 之前长度为 $L$ 的滑动历史回看窗口（Lookback Context Window）。该输入张量严格定义为半开半闭区间：
$$\mathbf{X}_{t-L:t} = [\mathbf{x}_{t-L}, \mathbf{x}_{t-L+1}, \dots, \mathbf{x}_{t-1}]^\top \in \mathbb{R}^{L \times C}$$
采用区间 $[t-L, t)$ 严密确保输入序列仅包含起点 $t$ 之前的已知历史，彻底杜绝当前决策时刻 $t$ 或未来的任何信息渗透。

预测任务的目标是推断系统在未来离散前瞻时域跨度 $H$（Prediction Horizon）内的完整状态演化轨迹：
$$\mathbf{Y}_{t:t+H} = [\mathbf{x}_{t}, \mathbf{x}_{t+1}, \dots, \mathbf{x}_{t+H-1}]^\top \in \mathbb{R}^{H \times C}$$
多步预测模型可形式化为一个由参数集 $\boldsymbol{\theta}$ 参数化的非线性映射函数 $f_{\boldsymbol{\theta}}: \mathbb{R}^{L \times C} \to \mathbb{R}^{H \times C}$，其点预测输出矩阵记为：
$$\hat{\mathbf{Y}}_{t:t+H} = f_{\boldsymbol{\theta}}(\mathbf{X}_{t-L:t})$$

在本项目中，针对电力小时负荷数据集（Electricity Hourly），输入窗口为过去 48 小时（$L=48$ 步），预测目标为未来 24 小时（$H=24$ 步）；针对光伏 10 分钟出力数据集（Solar 10-Minute），输入窗口为过去 48 小时（$L=288$ 步），预测目标为未来 24 小时（$H=144$ 步）。两类任务均涵盖 $C=16$ 个固定通道。

---

### 2.1.2 预测范式对比：递归预测、独立直接预测与联合向量直接映射

在多步时序预测领域，实现从历史回看窗口到未来多步轨迹的技术路径主要分为三种范式：

```
范式 1: 递归自回归多步预测 (Iterated / Recursive Forecasting)
[x_{t-L:t}] --> [Model f_θ] --> \hat{x}_t
[x_{t-L+1:t}, \hat{x}_t] --> [Model f_θ] --> \hat{x}_{t+1}  (将预测值作为伪观测循环反馈)

范式 2: 独立直接多步预测 (Independent Direct Forecasting)
[x_{t-L:t}] --> [Model f_{θ, 1}] --> \hat{x}_t
[x_{t-L:t}] --> [Model f_{θ, 2}] --> \hat{x}_{t+1}
...
[x_{t-L:t}] --> [Model f_{θ, H}] --> \hat{x}_{t+H-1}  (针对每个超前步分别独立训练单一模型)

范式 3: 联合向量直接多步映射 (Joint Multi-Step Vector Forecasting) [本项目采用]
[x_{t-L:t}] --> [Model f_θ] --> \hat{Y}_{t:t+H} \in R^{H \times C}  (单次前向传播直接输出完整未来矩阵)
```

#### 1. 递归自回归多步预测 (Recursive / Iterated Forecasting)
该范式仅训练一个单步预测器 $\hat{\mathbf{x}}_{t} = f_{\boldsymbol{\theta}}(\mathbf{X}_{t-L:t})$。在获取第一步预测后，将其作为“伪观测值”追加到历史窗口末尾，移除最旧的一个历史观测点，滚动向前推进预测第 $t+1$ 步，以此类推直至达到时域 $H$（如经典自回归概率预测模型 DeepAR，Salinas et al., 2020）。
- **优点**：模型紧凑，参数量与预测时域 $H$ 解耦，理论上支持任意长度的外推。
- **致命缺陷（复合误差级联传递，Compounding Error Propagation）**：
  设在第 $h$ 步的单步预测误差为 $\boldsymbol{\epsilon}_h = \mathbf{x}_{t+h-1} - \hat{\mathbf{x}}_{t+h-1}$。在无偏线性一阶自回归过程 $x_t = \phi x_{t-1} + \epsilon_t$（其中 $|\phi| < 1, \epsilon_t \sim \mathcal{N}(0, \sigma^2)$）下，$h$ 步前向预测误差的方差为：
  $$\operatorname{Var}(\hat{x}_{t+h|t} - x_{t+h}) = \sigma^2 \sum_{j=0}^{h-1} \phi^{2j} = \sigma^2 \frac{1 - \phi^{2h}}{1 - \phi^2}$$
  而在复杂的非线性深度神经网络中，前序时步生成的微小伪观测扰动会使后续输入的特征分布脱离真实数据流形。这种**输入分布漂移（Exposure Bias / Covariate Shift）**使得单步小误差在多步递归中被逐级非线性放大，导致远期预测轨迹极易发生相位严重偏移、波形平滑塌缩或发散震荡。同时，推演 $H=144$ 步需要串行执行 144 次模型前向计算，推理延迟显著增加。

#### 2. 独立直接多步预测 (Independent Direct Forecasting)
针对未来的每一个超前步长 $h \in \{1, 2, \dots, H\}$，分别训练一个独立的专用模型 $f_{\boldsymbol{\theta}_h}: \mathbb{R}^{L \times C} \to \mathbb{R}^{C}$，即 $\hat{\mathbf{x}}_{t+h-1} = f_{\boldsymbol{\theta}_h}(\mathbf{X}_{t-L:t})$。
- **优点**：各步预测独立建模，完全规避了自回归误差的递归级联放大；针对特定步长直接最小化经验风险。
- **缺陷**：必须维护 $H$ 套完全独立的模型参数，当光伏预测步长达到 $H=144$ 时，模型参数量和训练耗时直接膨胀 144 倍，工程存储与维护成本极高；更严重的是，该范式人为割裂了未来时间步之间的联合协方差结构 $\operatorname{Cov}(\hat{\mathbf{x}}_{t+h_1}, \hat{\mathbf{x}}_{t+h_2})$，容易产生缺乏时间平滑性的锯齿状离散预测轨迹。

#### 3. 联合向量直接多步映射 (Joint Multi-Step Vector Forecasting)
使用统一的网络拓扑，在单次前向传播中直接输出未来 $H$ 步的联合预测矩阵 $\hat{\mathbf{Y}} \in \mathbb{R}^{H \times C}$。网络利用输出投影头或时空自注意力/卷积层隐式学习跨时间步的时序协方差。
- **优点**：
  1. **零自回归累积误差**：输入完全基于真实历史量测，杜绝了自回归递归的误差发散与分布漂移风险；
  2. **端到端并行优化与低延迟**：单次 GPU/CPU 前向推理耗时仅数毫秒（$\mathcal{O}(1)$ 计算图），且联合均方误差损失函数驱动网络在未来 24 小时的整体时序包络上全局协同优化，保证预测轨迹具备自然的物理连续性。
- **本项目决策**：全面确立**联合向量直接多步映射**范式作为所有深度学习与现代基准模型的统一架构。

---

### 2.1.3 严格时序防信息泄漏准则

时间维度的单向因果性决定了任何未来信息的反向渗透都将导致严重的前瞻偏差（Look-Ahead Bias / Temporal Data Leakage）。本项目确立了四项防泄漏准则：
1. **时序单向定序切分**：按时间轴次序将总序列划分为训练区间 $[0, b_1)$、验证区间 $[b_1, b_2)$、测试区间 $[b_2, T)$，其中第一切分点 $b_1 = \lfloor 0.70 T \rfloor$、第二切分点 $b_2 = \lfloor 0.80 T \rfloor$。绝对禁止对原始时间点或滑动窗口执行跨区间随机混洗（Random Shuffling）；
2. **预测目标窗口严格区间隔离**：训练、验证与测试的预测目标窗口 $\mathbf{Y}_{t:t+H}$ 必须完整留在各自所属的区间内，约束条件为：训练期 $t+H \le b_1$、验证期 $t \ge b_1$ 且 $t+H \le b_2$、测试期 $t \ge b_2$ 且 $t+H \le T$；
3. **特征缩放统计量严格隔离**：各通道的 $z$-score 标准化均值 $\mu_c$ 与标准差 $\sigma_c$ 仅且只能在训练目标区间 $[0, b_1)$ 上计算拟合：
   $$\mu_c = \frac{1}{b_1} \sum_{\tau=0}^{b_1-1} X_{\tau, c}, \quad \sigma_c = \sqrt{\frac{1}{b_1} \sum_{\tau=0}^{b_1-1} (X_{\tau, c} - \mu_c)^2 + \epsilon}$$
   其中 $\epsilon = 10^{-8}$ 提供零方差数值保护。严禁在切分前对全数据集执行标准化。所有模型预测值必须经过逆标准化还原至原始物理尺度后方可计算最终指标；
4. **无外生变量单向信息约束**：仅使用历史内生负荷/出力观测，不提供未来数值天气预报（NWP）或跨预测起点的后验插值特征，确保基准对比的信息公平性。

---

## 2.2 经典统计学与启发式时序方法

### 2.2.1 基础惯性基准：持续性模型 (Persistence Model / Random Walk)

持续性模型（随机游走基准）直接复制预测决策起点前最后一个已知观测值：
$$\hat{\mathbf{y}}_{t+h} = \mathbf{x}_{t-1}, \quad \forall h \in [0, H-1]$$
该模型假设目标系统服从无漂移项的一阶鞅过程（Martingale Process）：$\mathbb{E}[\mathbf{x}_{t+h} \mid \mathcal{F}_t] = \mathbf{x}_{t-1}$。在极短时域（如 10 分钟至 1 小时内），由于受电网热惯性与晴空太阳辐射连续性支配，持续性模型表现优良；但随着前瞻步长拓展，由于预测轨迹为恒定水平直线，完全丧失对日内周期波动的表征能力，误差随 $h$ 呈发散增长。

---

### 2.2.2 周期性朴素基准：日季节性朴素模型 (Daily Seasonal Naive)

日季节性朴素模型基于系统强日周期自相似性假设，直接取前一天同一时刻的历史观测：
$$\hat{\mathbf{y}}_{t+h} = \mathbf{x}_{t - m + (h \bmod m)}, \quad \forall h \in [0, H-1]$$
其中 $m$ 代表日周期采样步长（电力小时数据 $m=24$，光伏 10 分钟数据 $m=144$）。
- **光伏出力场景的极强基准**：受天体几何学决定，日出、日落与夜间零出力高度稳定。夜间连续数十个采样点出力严格为零，日季节性模型能以近乎 $100\%$ 的准确度预测夜间零值，在全天综合指标中构建起极高的基准门槛；
- **电力负荷场景的周内偏差**：在工作日与周末交替时（周五至周六、周日至周一），负荷水平存在台阶式跳变，日季节性模型会产生系统性阶跃偏差。

---

### 2.2.3 经典统计自回归模型：ARIMA / SARIMA / ARMAX 理论体系

经典时间序列分析建立在 Wold 线性分解定理与差分平稳化理论之上（Box et al., 2015）：
1. **ARIMA 模型**：$\text{ARIMA}(p, d, q)$ 利用滞后算子 $B$（$B^k y_t = y_{t-k}$）表述为：
   $$\Phi_p(B) (1 - B)^d y_t = \Theta_q(B) \epsilon_t$$
   其中 $\Phi_p(B) = 1 - \sum_{i=1}^p \phi_i B^i$ 为自回归多项式，$(1 - B)^d$ 为差分算子，$\Theta_q(B) = 1 + \sum_{j=1}^q \theta_j B^j$ 为滑动平均多项式，$\epsilon_t \sim \mathcal{N}(0, \sigma^2)$。
2. **SARIMA 模型**：引入季节差分 $(1 - B^m)^D$ 与季节滞后多项式 $\boldsymbol{\Phi}_P(B^m)$、$\boldsymbol{\Theta}_Q(B^m)$ 处理周期相关性：
   $$\Phi_p(B) \boldsymbol{\Phi}_P(B^m) (1 - B)^d (1 - B^m)^D y_t = \Theta_q(B) \boldsymbol{\Theta}_Q(B^m) \epsilon_t$$
3. **ARMAX 与项目无外生变量设定边界**：引入外生变量 $\mathbf{u}_t$ 扩展为 $\text{ARMAX}(p, q, b)$：
   $$\Phi(B) y_t = \Theta(B) \epsilon_t + \sum_{k=1}^K \Gamma_k(B) u_{t, k}$$
   但在因果预测中，利用 ARMAX 预测未来 $H$ 步负荷的前提是必须事先获得未来高精度外生变量预报（如未来 24 小时的数值天气预报 NWP）。本项目严格受限于**纯历史内生观测条件**，无未来外生输入，此时 ARMAX 退化为标准 ARIMA。此外，在 $C=16$ 维高维通道下，向量自回归（VARIMA）参数量按 $\mathcal{O}(p C^2)$ 急剧膨胀，且对于光伏 $m=144$ 的超长滞后，协方差矩阵求逆存在严重的数值病态性与算力瓶颈。

---

### 2.2.4 经典统计模型的理论局限性与国际基准实证启示

Makridakis 等人在 M4 国际时序竞赛总结中指出（Makridakis et al., 2018），尽管纯统计与简单基准在强季节性时序上极具竞争力，但经典统计方法受限于三点内在瓶颈：线性高斯生成假设难以表征云团遮蔽等非线性突变；多变量多通道协同建模面临维数灾难；线性自回归记忆随滞后步数呈指数快速衰减，难以跨越上百步维持长程稳定表征。

---

## 2.3 经典深度时序表征：循环神经网络与门控演进

### 2.3.1 经典 RNN 与随时间反向传播算法 (BPTT)

循环神经网络（RNN）通过隐状态递归建模序列时间依赖：
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
其中 $\mathbf{h}_t \in \mathbb{R}^{d_h}$ 为隐藏记忆向量。利用随时间反向传播（BPTT）优化参数时，总损失 $\mathcal{L} = \sum_{t=1}^L \mathcal{L}_t$ 对自转移权重矩阵 $\mathbf{W}_{hh}$ 的偏导数为：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^L \sum_{k=1}^t \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

---

### 2.3.2 梯度消失与梯度爆炸的严格数学推导与谱半径分析

长程依赖学习的核心在于雅可比矩阵连乘项 $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}$。由链式法则展开：
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \prod_{j=k+1}^t \operatorname{diag}\left(1 - \tanh^2(\mathbf{a}_j)\right) \mathbf{W}_{hh}^\top$$
其中 $\mathbf{a}_j = \mathbf{W}_{hh} \mathbf{h}_{j-1} + \mathbf{W}_{xh} \mathbf{x}_j + \mathbf{b}_h$。根据矩阵范数相乘性质：
$$\left\| \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \right\| \le \prod_{j=k+1}^t \left\| \operatorname{diag}\left(1 - \tanh^2(\mathbf{a}_j)\right) \right\| \cdot \|\mathbf{W}_{hh}^\top\|_2 \le (\gamma \lambda_{\max})^{t - k}$$
其中 $\gamma \le 1$ 为激活函数导数上界，$\lambda_{\max} = \|\mathbf{W}_{hh}\|_2$ 为权重矩阵的最大奇异值（谱半径）。Pascanu 等人（2013）证明了：
- **梯度消失 (Gradient Vanishing)**：若 $\lambda_{\max} < 1$ 或网络处于饱和区导致 $\gamma \lambda_{\max} < 1$，当回看跨度 $t - k \gg 1$ 时，连乘项呈几何指数衰减：$\lim_{t-k \to \infty} (\gamma \lambda_{\max})^{t-k} = 0$，使得网络完全无法捕获远期历史依赖；
- **梯度爆炸 (Gradient Exploding)**：若 $\lambda_{\max} > 1$ 且网络处于非饱和区，连乘项呈指数发散，导致数值溢出与训练崩溃。

这一理论缺陷直接解释了本项目为何坚决剔除原生 Vanilla RNN，全面采用门控结构的深层原因。

---

### 2.3.3 长短期记忆网络 (LSTM, Hochreiter & Schmidhuber 1997)

Hochreiter 与 Schmidhuber（1997）提出了 LSTM，核心创新在于引入独立细胞状态 $\mathbf{c}_t$ 与三组乘性门控，构建了线性的恒等误差传送带（Constant Error Carousel, CEC）：
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \quad \text{(遗忘门)}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \quad \text{(输入门)}$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c) \quad \text{(候选细胞状态)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(细胞状态加法更新)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \quad \text{(输出门)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(隐藏状态输出)}$$
其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 为 Sigmoid 函数，$\odot$ 为阿达马（Hadamard）逐项乘积。在细胞状态更新中，直接线性导数路径为 $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \operatorname{diag}(\mathbf{f}_t)$。当网络学习到 $\mathbf{f}_t \approx \mathbf{1}$ 时，误差沿着加法传送带无衰减长程传递，彻底从机制上解决了 BPTT 的梯度消失问题。

---

### 2.3.4 门控循环单元 (GRU, Cho et al. 2014)

Cho 等人（2014）提出了精简架构 GRU，移除了独立细胞状态，将内部状态整合为单一隐状态 $\mathbf{h}_t$，并通过更新门与重置门进行平衡：
$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) \quad \text{(重置门)}$$
$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) \quad \text{(更新门)}$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \quad \text{(候选隐藏状态)}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(凸组合插值更新)}$$
在相同隐层维度 $d_h$ 下，GRU 仅需 3 组门控权重矩阵，参数量相比 LSTM 精简了整整 $25\%$，在轻量级电力时序任务中收敛更快、抗轻度过拟合能力更强。

---

## 2.4 局部时序特征抽取：一维时间卷积与混合级联架构

### 2.4.1 一维时间卷积网络 (1D-CNN) 原理与归纳偏置

一维时间卷积网络通过滑动卷积核对时间序列局部子片段进行特征抽取（Bai et al., 2018）。对于输入张量 $\mathbf{X} \in \mathbb{R}^{B \times C_{\text{in}} \times L}$，卷积核尺寸为 $K$，输出特征通道为 $C_{\text{out}}$，卷积计算为：
$$Z_{j, t} = \operatorname{ReLU}\left( \operatorname{BatchNorm}\left( \sum_{c=1}^{C_{\text{in}}} \sum_{k=0}^{K-1} W_{j, c, k} X_{c, t + k - \lfloor K/2 \rfloor} + b_j \right) \right)$$
堆叠 $M$ 层核尺寸为 $K$ 的卷积层后，网络在时间轴上的有效感受野大小满足：
$$RF = 1 + M \cdot (K - 1)$$
在本项目配置下（$M=2, K=3$），感受野为 $RF = 5$ 步。1D-CNN 的核心归纳偏置包括：
1. **局部感知野（Local Receptive Field）**：聚焦相邻时步差分形态，能极高信噪比地捕获设备启停尖峰或光伏瞬态跌落；
2. **时间平移等变性与权值共享**：时间轴复用相同卷积核，有效抑制自由参数量；
3. **高度并行计算**：全时步前向特征在 GPU 上一次性并行展开，吞吐量远超串行推进的 RNN；
4. **自适应全局池化适配**：末端引入 $\operatorname{AdaptiveAvgPool1d}(16)$，将任意输入长度 $L$（如 24h、48h、168h）统一映射为固定维特征，赋予模型在历史窗口敏感性实验中的网络结构复用性。

---

### 2.4.2 因果卷积 vs. 历史窗口内非因果卷积边界辨析

在流式自回归生成中必须使用因果卷积（严格限制时刻 $t$ 的卷积核只能访问 $\le t$ 的历史）。但在本项目统一评测设定下，输入张量是明确已经完全发生的历史窗口 $\mathbf{X}_{t-L:t} = [\mathbf{x}_{t-L}, \dots, \mathbf{x}_{t-1}]^\top$。在该历史窗口内部，所有 $L$ 个采样点在物理时序上均属于“既成事实”（最大时步 $t-1 < t$）。因此，在历史窗口内部采用标准对称卷积提取全局波形特征完全合法，绝不存在跨越预测起点 $t$ 的未来信息泄漏，必须厘清该理论认知。

---

### 2.4.3 混合级联网络 (CNN--LSTM) 的理论归纳偏置假设

CNN--LSTM 级联网络采用串行信息流管道：
$$\mathbf{X}_{t-L:t} \xrightarrow{\text{1D-Conv}} \mathbf{Z}_{\text{conv}} \xrightarrow{\text{BatchNorm + ReLU}} \mathbf{Z} \xrightarrow{\text{LSTM}} \mathbf{h}_{L-1} \xrightarrow{\text{Linear Head}} \hat{\mathbf{Y}} \in \mathbb{R}^{H \times C}$$
- **正向协同假设**：卷积前端充当低通滤波与局部特征提取器，滤除高频噪声并捕获瞬态波形，大幅减轻后置 LSTM 记忆单元的认知负担，使 LSTM 专注于建模宏观周期相位与日内演化；
- **潜在代价与过平滑风险**：级联结构增加了反向传播计算图深度，在训练样本受限时存在过拟合与平滑掉尖峰极值的风险，需通过消融实验严格检验。

---

## 2.5 现代时序架构：从 Transformer 机制演进到线性解耦反思

### 2.5.1 标准 Transformer 在时序预测中的瓶颈与反思

标准点积自注意力机制为 $\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$（Vaswani et al., 2017）。然而在长程时序预测中暴露出三大致命瓶颈：
1. **时间点级 Tokenization 缺陷**：单个离散数值不具备独立语义，点积注意力极易拟合高频噪声导致虚假相关；后续 Informer（Zhou et al., 2021）尝试采用 ProbSparse 稀疏自注意力，Autoformer（Wu et al., 2021）引入自相关机制，但仍难以彻底摆脱点级建模脆弱性；
2. **$\mathcal{O}(L^2)$ 计算与显存开销**：限制了对光伏等高频长序列历史的摄入能力；
3. **置换不变性破坏时序顺序**：对时间序列内在的严格先后顺序缺乏保真度。虽然 N-BEATS（Oreshkin et al., 2020）探索了纯前馈基函数扩展，但时序领域依然亟需兼顾全局依赖与局部形态的架构突破。

---

### 2.5.2 PatchTST (ICLR 2023)：时序切片与通道独立性的范式革命

Nie 等人（2023）提出了 PatchTST，确立了两项重大机制革新：
1. **时序分块切片机制 (Temporal Patching)**：
   将长度为 $L$ 的序列切分为长度 $P$、步长 $S$ 的重叠 Patches。有效 Token 数量为：
   $$N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1$$
   当 $(L - P) \bmod S \ne 0$ 时，末尾追加长度为 $(N-1)S + P - L$ 的 Padding，确保最新历史观测被完整覆盖。单个 Token 汇聚连续局部形态，注意力复杂度由 $\mathcal{O}(L^2)$ 骤降至 $\mathcal{O}(N^2)$。
2. **通道独立性 (Channel Independence, CI) 范式**：
   传统模型采用通道混合（Channel Mixing），试图同时拟合时间与通道协方差，但在有限样本下极易产生高维过拟合。PatchTST 将 $C$ 个通道展平至 Batch 维度：
   $$\mathbf{X} \in \mathbb{R}^{B \times L \times C} \longrightarrow \mathbf{X}_{\text{CI}} \in \mathbb{R}^{(B \cdot C) \times L \times 1}$$
   所有通道共享同一个 Transformer 编码器。CI 范式不仅消除了通道间的虚假相关与维度灾难，更为主干网络提供了整整 $C$ 倍（本项目中为 $16\times$）的等效训练样本扩增，结合可逆实例归一化（RevIN）思想（Kim et al., 2022），极大提升了少样本环境下的训练稳定性与泛化性能。

---

### 2.5.3 现代线性解耦反思：DLinear (AAAI 2023)

针对复杂注意力机制的必要性，Zeng 等人（2023）提出了极简加法分解模型 DLinear：
1. **移动平均趋势分解**：采用核宽 $K_{\text{ma}}=25$ 的对称平滑核提取趋势项 $\mathbf{X}_{\text{trend}} = \operatorname{AvgPool}(\mathbf{X})$，季节周期残差项为 $\mathbf{X}_{\text{seasonal}} = \mathbf{X} - \mathbf{X}_{\text{trend}}$；
2. **双分支独立线性层映射**：
   $$\hat{\mathbf{Y}} = \mathbf{X}_{\text{trend}} \mathbf{W}_{\text{trend}}^\top + \mathbf{b}_{\text{trend}} + \mathbf{X}_{\text{seasonal}} \mathbf{W}_{\text{seasonal}}^\top + \mathbf{b}_{\text{seasonal}}$$
   在通道共享设置下（`individual=False`），两组权重共享于全部 $C=16$ 通道，总参数量仅为 $2 \times L \times H + 2 \times H = 2,352$。

DLinear 具备固有的时序滞后顺序保真度，近乎无过拟合倾向，且前向推理达到亚毫秒级，是工业嵌入式边缘电表的帕累托最优方案。

---

## 2.6 领域特定能源物理机理与时序动力学特性

### 2.6.1 电力系统用户负荷动力学

电力负荷由人类社会活动直接支配，具有双重强周期结构（Hong & Fan, 2016）：
$$y_t = T_t + S_t^{(24)} + S_t^{(168)} + I_t$$
- **日内双峰特征 ($S_t^{(24)}$)**：晨间 8:00--11:00 早高峰与傍晚 18:00--21:00 晚高峰，凌晨 2:00--5:00 达到夜间基底负荷谷底；
- **工作日/周末阶跃 ($S_t^{(168)}$)**：周六周日工商业负荷系统性下降 $15\% \sim 30\%$，构成拓展至 168 小时周周期学习的核心物理依据；
- **分布式用户的高方差特性**：16 个离散用户节点缺乏大电网的中心极限定理平滑效应，呈现强随机脉冲与突变台阶。

---

### 2.6.2 光伏发电出力物理学原理与绝对硬约束

光伏电池出力直接受太阳辐射与大气物理控制（Sobri et al., 2018; Inman et al., 2013; Zhang et al., 2014）：
$$P_{\text{clear}}(t) = \eta A I_{\text{sc}} E_0 \max(0, \cos \theta_z(t))$$
其中天顶角满足 $\cos \theta_z(t) = \sin \phi \sin \delta + \cos \phi \cos \delta \cos \omega(t)$。
- **夜间绝对物理零出力**：夜间 $\theta_z(t) > 90^\circ \implies P_{\text{pv}}(t) \equiv 0$。夜间零值占据近 $50\%$ 采样点，日季节性模型凭借完美预测夜间零值在此区间获得极强视在优势，要求必须补充日间活跃出力切片评估；
- **气象随机爬坡事件 (Ramp Events)**：碎积云移动造成功率爬坡率 $\text{RR}_{\Delta t} = \frac{|P_t - P_{t-\Delta t}|}{\Delta t}$ 在数分钟内高达 $80\%$，属于极度非线性随机突变；
- **物理非负性约束**：真实功率严格满足 $P_{\text{pv}}(t) \ge 0$。

---

## 2.7 时间序列评估指标体系的理论剖析

### 2.7.1 尺度依赖误差指标：MAE 与 RMSE

$$\text{MAE} = \frac{1}{N H C} \sum_{i=1}^N \sum_{h=1}^H \sum_{c=1}^C \left| Y_{i, h, c} - \hat{Y}_{i, h, c} \right|$$
$$\text{RMSE} = \sqrt{\frac{1}{N H C} \sum_{i=1}^N \sum_{h=1}^H \sum_{c=1}^C \left( Y_{i, h, c} - \hat{Y}_{i, h, c} \right)^2}$$

MAE 对应拉普拉斯误差分布假设（中位数最优），对离群值具有鲁棒性；RMSE 对应高斯分布假设（条件均值最优），对大误差施加二次方重罚，在防范电网跳闸过载等极端风险中具有最高安全权重。

---

### 2.7.2 相对百分比指标的病态性分析：严禁将 MAPE 作为光伏主评指标

$$\text{MAPE} = \frac{1}{N H C} \sum_{i=1}^N \sum_{h=1}^H \sum_{c=1}^C \left| \frac{Y_{i, h, c} - \hat{Y}_{i, h, c}}{Y_{i, h, c}} \right| \times 100\%$$

当实际物理观测值 $Y \to 0$ 时，分母趋零引发数值奇异发散：$\lim_{Y \to 0^+} |\frac{Y - \hat{Y}}{Y}| = +\infty$。光伏夜间存在近半数的严格零值，且晨昏数值微小（$Y \approx 10^{-4}$），导致 MAPE 在数学上不可定义且数值极度病态，人工平滑 $\epsilon$ 会严重扭曲评价客观性，因此严禁作为主评指标。

---

### 2.7.3 季节性平均绝对缩放误差 (Seasonal MASE) 的理论优越性

Hyndman 与 Koehler（2006）提出的 Seasonal MASE 定义为：
$$\text{MASE}_c = \frac{\text{MAE}_c}{\max\left(\frac{1}{b_1 - m} \sum_{\tau=m}^{b_1-1} \left| X_{\tau, c} - X_{\tau-m, c} \right|, \; \epsilon\right)}$$
其中分母严格且仅在训练区间 $[0, b_1)$ 上计算（日周期步长 $m=24$ 对应电力，$m=144$ 对应光伏），$\epsilon=10^{-8}$ 提供零方差数值保护。Seasonal MASE 实现了不同功率量级通道间的无偏归一化；$\text{MASE} < 1.0$ 具有清晰的物理意义，标志着模型有效超越了简单前一日机械复制基准。

---

## 2.8 理论全景综合对比矩阵与指导意义

下表全景汇总了 8 类模型的归纳偏置、感受野、通道交互与理论特征：

| 架构类别 | 代表模型 | 核心归纳偏置 | 时序感受野机制 | 通道处理范式 | 主干与投影头复杂度 | 核心理论优势与局限 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **启发式基准** | Persistence | 鞅过程, $\mathbb{E}[x_{t+h}]=x_{t-1}$ | 仅末端 1 步 | 通道独立 | $0$ | 零计算开销; 随预测时域发散, 无周期响应 |
| **启发式基准** | Seasonal Naive | 日周期严格自相似 | 固定滞后 $m$ 步 | 通道独立 | $0$ | 光伏夜间零值极强; 无法响应天气突变与周转变 |
| **经典循环** | LSTM | 门控线性误差传递 (CEC) | 递归状态隐式累积 | 通道混合 | $\mathcal{O}(4(Cd_h + d_h^2)) + \mathcal{O}(d_h H C)$ | 抑制梯度消失; 串行推进计算慢, 复杂多步易漂移 |
| **经典循环** | GRU | 重置与更新双门控 | 递归状态隐式累积 | 通道混合 | $\mathcal{O}(3(Cd_h + d_h^2)) + \mathcal{O}(d_h H C)$ | 参数比 LSTM 少 25%; 仍受限于串行计算 |
| **时间卷积** | 1D-CNN | 局部感知野, 平移等变性 | 卷积核局部感知 + 池化 | 通道混合 | $\mathcal{O}(\sum K C_{\text{in}} C_{\text{out}}) + \mathcal{O}(P_{\text{pool}} C_o H C)$ | 高度并行加速, 捕获骤变; 输出头参数占比较大 |
| **混合级联** | CNN--LSTM | 局部滤波与长程趋势解耦 | 局部卷积窗 + 全局递归 | 卷积先混, 再入循环 | 介于 CNN 与 LSTM 之间 | 兼顾局部微结构与宏观趋势; 存在过平滑风险 |
| **现代极简线性** | DLinear | 趋势-季节移动平均解耦 | 全局历史直接线性全连接 | 通道共享 (CI) | $\mathcal{O}(2 \cdot L \cdot H)$ | 时序顺序保真, 极抗过拟合, 亚毫秒推理; 非线性能力弱 |
| **现代时序大模型** | PatchTST | 时序切片, 通道独立自注意力 | 跨 Patch 宏观自注意力 | 通道独立 (CI) | $\mathcal{O}(N^2 d_m + d_m d_{\text{ff}}) + \mathcal{O}(N d_m H)$ | 计算骤降, 泛化强, $C\times$ 数据扩增; 超短窗口受限 |

### 对后续实验设计的理论指导与推论
1. **关于多步输出机制**：理论推导证明自回归递归必然带来误差级联，后续所有深度模型统一采用直接多步全连接向量输出；
2. **关于指标选择**：光伏夜间零值必然造成 MAPE 数值发散，确立原尺度 MAE、RMSE 及无量纲 Seasonal MASE 为核心三元指标，并增设白天活跃出力切片评测；
3. **关于消融与敏感性预期**：PatchTST 依托通道独立性理论具备 $16\times$ 的等效数据增强效应，预期在训练数据量敏感性实验（25% 极小样本）中展现出远超传统 LSTM 的样本高效性；而历史窗口敏感性实验中，拓展至 168 小时完整周周期将对捕捉电力周末效应产生决定性正向增益。

---

## 参考文献 (References)

1. **Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.** (2015). *Time series analysis: forecasting and control*. John Wiley & Sons.
2. **Hyndman, R. J., & Koehler, A. B.** (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
3. **Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P.** (2021). Monash time series forecasting repository. *Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.
4. **Hochreiter, S., & Schmidhuber, J.** (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
5. **Cho, K., van Merriënboer, B., Gülçehre, Ç., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y.** (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Empirical Methods in Natural Language Processing (EMNLP)*, 1724-1734.
6. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998-6008.
7. **Zeng, A., Chen, M., Zhang, L., & Xu, Q.** (2023). Are transformers effective for time series? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128.
8. **Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J.** (2023). A time series is worth 64 words: Long-term forecasting with transformers. *International Conference on Learning Representations (ICLR)*.
9. **Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W.** (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.
10. **Wu, H., Xu, J., Wang, J., & Long, M.** (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 22419-22430.
11. **Zhang, Y., Wang, J., & Wang, X.** (2014). Review on probabilistic forecasting of wind power generation. *Renewable and Sustainable Energy Reviews*, 32, 255-270.
12. **Sobri, S., Koohi-Kamali, S., & Rahim, N. A.** (2018). Solar photovoltaic generation forecasting methods: A review. *Energy Conversion and Management*, 156, 459-497.
13. **Hong, T., & Fan, S.** (2016). Probabilistic electric load forecasting: A tutorial review. *International Journal of Forecasting*, 32(3), 914-938.
14. **Inman, R. H., Pedro, H. T., & Coimbra, C. F.** (2013). Solar forecasting methods for renewable energy integration. *Progress in Energy and Combustion Science*, 39(6), 535-576.
15. **Bai, S., Kolter, J. Z., & Koltun, V.** (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.
16. **Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J.** (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *International Conference on Learning Representations (ICLR)*.
17. **Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y.** (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *International Conference on Learning Representations (ICLR)*.
18. **Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T.** (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191.
19. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V.** (2018). The M4 Competition: Results, findings, problems and ways forward. *International Journal of Forecasting*, 34(4), 802-808.
20. **Pascanu, R., Mikolov, T., & Bengio, Y.** (2013). On the difficulty of training recurrent neural networks. *International Conference on Machine Learning (ICML)*, 1310-1318.
