# 基于多源特征工程与稳健融合的加密货币市场预测模型研究

## 摘要

加密货币市场短期收益预测具有信号弱、噪声高、特征维度高和分布不稳定等特点。本文以 Kaggle DRW Crypto Market Prediction 任务为研究对象，基于匿名市场特征预测未来短期收益相关目标，官方评价指标为预测值与真实标签之间的 Pearson correlation。原始训练集包含 525886 行样本、785 个数值特征和目标列 `label`，测试集包含 538150 行样本。由于特征高度匿名且存在明显冗余，本文构建了一套从线性基线、特征筛选、稳健验证、结构化特征工程到低权重非线性信号融合的完整建模流程。

实验首先验证了 Ridge 等线性模型在高维金融特征上的强基线作用，并通过 Pearson top-k 和 Spearman top-k 筛选获得稳定提升。随后，本文引入 correlation medoid、SHAP-stable XGBoost、symbolic interaction features、supervised AutoEncoder 和 supervised MLP 等多类互补信号，并通过小权重融合缓解单一模型在 public/private 分布差异下的过拟合风险。最终模型采用多阶段 ensemble 结构，在 Kaggle private leaderboard 上取得 0.10550 的 Pearson score；按实验记录中的榜单截图，该结果 private 排名约为第 14 名。实验表明，该任务的关键不在于盲目增加模型复杂度，而在于对高维匿名特征进行稳定筛选、构造可转移的新信号，并在 public/private 偏差下采取保守融合策略。

**关键词**：加密货币预测；Pearson correlation；Ridge 回归；特征工程；模型融合；稳健验证

---

## 1. 问题背景与任务定义

### 1.1 金融短期预测的建模困难

金融市场短期收益预测通常属于弱信号预测问题。与图像、文本等任务相比，市场数据中的目标信号往往被大量噪声覆盖，且特征与标签之间的关系可能随时间变化而漂移。加密货币市场具有交易活跃、波动剧烈、结构快速变化等特点，因此模型不仅需要在训练样本上获得较高相关性，还必须具备对隐藏测试分布的泛化能力。

本任务中的大部分特征为匿名变量，除 `bid_qty`、`ask_qty`、`buy_qty`、`sell_qty`、`volume` 等少数可命名市场特征外，其余特征以 `X1-X780` 的形式给出。这意味着建模过程中无法依赖明确的业务语义解释每个特征，而必须更多依靠统计相关性、特征冗余结构、时间稳定性和模型输出行为进行判断。

此外，Kaggle 比赛通常包含 public leaderboard 和 private leaderboard。public 分数只基于测试集的一部分，private 分数基于隐藏测试集的另一部分。若模型过度追逐 public 分数，可能出现 public 提升而 private 下降的情况。因此，本文特别关注模型在不同验证口径下的稳定性，并将 public/private divergence 作为模型分析的重要部分。

### 1.2 任务目标与评价指标

本任务的目标是对测试集中每个样本生成一个连续预测值 `prediction`，使其与隐藏真实标签 `label` 的 Pearson correlation 尽可能高。

Pearson correlation 的定义为：

$$
\rho(y, \hat{y}) =
\frac{
\sum_{i=1}^{n}(y_i-\bar{y})(\hat{y}_i-\bar{\hat{y}})
}{
\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\sqrt{\sum_{i=1}^{n}(\hat{y}_i-\bar{\hat{y}})^2}
}
$$

其中，$y_i$ 表示真实标签，$\hat{y}_i$ 表示模型预测值，$\bar{y}$ 和 $\bar{\hat{y}}$ 分别表示真实值和预测值均值。

与 RMSE 不同，Pearson correlation 更关注预测值与真实值之间的线性相关方向，而不直接惩罚预测值的绝对尺度偏差。这一特点对模型选择产生了重要影响：部分模型在 RMSE 或本地 holdout 上表现较好，但并不一定能在 leaderboard 上获得更高 Pearson 分数。因此，本文在实验中同时记录 holdout Pearson、RMSE、public score 和 private score，并重点分析它们之间的不一致。

---

## 2. 数据与预处理

### 2.1 数据结构

本项目使用 Kaggle DRW Crypto Market Prediction 官方数据集。数据文件包括训练集、测试集和提交样例文件。

| 数据文件                    |     行数 |  列数 | 说明                        |
| ----------------------- | -----: | --: | ------------------------- |
| `train.parquet`         | 525886 | 786 | 785 个特征 + `label`         |
| `test.parquet`          | 538150 | 786 | 测试特征，提交时按样例文件对齐           |
| `sample_submission.csv` | 538150 |   2 | 包含 `ID` 和 `prediction` 两列 |

训练集中，`label` 为目标列，其余数值列作为候选特征。测试集中的 `label` 为占位或不可用信息，不参与建模。最终提交文件需保持与 `sample_submission.csv` 中的 `ID` 顺序一致。

### 2.2 预处理策略

本文采用统一的表格数据预处理流程：

1. 仅使用数值特征；
2. 排除目标列 `label` 和提交标识列 `ID`；
3. 缺失值使用训练 split 的中位数填充；
4. 数值特征使用训练 split 的均值和标准差进行标准化；
5. 所有特征筛选、top-k 排名、interaction scoring、AE/MLP 训练均只在训练部分拟合，避免验证集信息泄露。

对于 Ridge、XGBoost、MLP 等不同分支，预处理器均在对应训练数据上单独拟合，然后应用于验证集或测试集。这样可以保证实验比较的可复现性和防泄露性。

### 2.3 验证切分与泄露控制

主验证口径采用 row-order chronological split，即按原始行顺序将前 80% 样本作为训练集，后 20% 样本作为 holdout 验证集：

$$
D_{\text{train}} = D_{1:\lfloor 0.8n \rfloor}, \quad
D_{\text{valid}} = D_{\lfloor 0.8n \rfloor+1:n}
$$

这一切分方式保留了样本的时间顺序假设，避免随机切分带来的潜在时间泄露。对于部分需要更强稳定性分析的分支，本文进一步使用 rolling validation 和 purged group time split。例如，在 SHAP-stable XGBoost 和 Sprint-A interaction stability 中，训练集被划分为多个连续 group，并在验证某一 group 时剔除相邻 group，以降低邻近时间样本带来的泄露风险。

---

## 3. Baseline 建模

### 3.1 Full-feature baseline

首先，本文在全部原始数值特征上训练 Ridge、LightGBM 及二者的加权融合模型，作为未经特征筛选的基础对照。

Ridge 回归的目标函数为：

$$
\min_{\mathbf{w}}
\left\{
\sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top \mathbf{w})^2
+
\alpha \lVert \mathbf{w} \rVert_2^2
\right\}
$$

其中，$\alpha$ 为正则化强度。较大的 $\alpha$ 会压缩模型参数，降低过拟合风险。

Full-feature baseline 的 holdout 结果如下：

| 模型                    | Holdout Pearson |
| --------------------- | --------------: |
| Full Ridge            |        0.097166 |
| Full LightGBM         |        0.063721 |
| Full Ridge + LightGBM |        0.100263 |

结果显示，Ridge 明显强于单独 LightGBM，说明该任务中存在较强的线性相关结构。直接使用树模型并未取得优势，可能是由于原始特征维度高、冗余强且噪声较多。因此，后续实验不再简单追求复杂模型，而是首先围绕特征筛选与信号构造展开。

### 3.2 Pearson top-k Ridge

为了降低高维匿名特征中的冗余和噪声，本文首先使用 train-only Pearson correlation 对特征排序。对每个特征 $x_j$，计算其与标签 $y$ 的绝对 Pearson 相关系数：

$$
s_j = |\rho(x_j, y)|
$$

然后选择排名前 $k$ 的特征训练 Ridge 模型。实验比较了 top50、top100、top200、top300、top500 以及 full feature 等不同宽度。

该阶段的关键发现为：

| 模型分支                          |  Public | Private | 说明                 |
| ----------------------------- | ------: | ------: | ------------------ |
| top100 Ridge                  | 0.03716 | 0.08901 | 单一 top100 Ridge 信号 |
| top50/top100 Ridge blend      | 0.04652 | 0.09280 | top50 与 top100 互补  |
| high-alpha top50/top100 blend | 0.05160 | 0.09638 | Beta2 最优线性基线       |

Pearson top-k Ridge 的提升说明，原始 785 个特征中存在大量冗余和噪声。通过简单的相关性筛选，模型不仅更稳定，而且在 private leaderboard 上显著优于未经筛选的模型。

---

## 4. 特征筛选与稳健性探索

### 4.1 Spearman top-k signal

Pearson correlation 衡量线性关系，而 Spearman correlation 衡量排序关系。对于金融数据，特征与收益之间的关系可能并不严格线性，而表现为更弱的单调或排序信号。因此，本文进一步使用 Spearman top-k 特征筛选构造新的 Ridge 信号。

Spearman top50 Ridge 以小权重加入 Pearson Ridge stack 后，private score 提升到 0.10003。后续权重微调显示，Spearman signal 在 0.225 到 0.235 附近形成 private plateau，继续增加权重会导致 private 回落。

这一结果说明，Spearman top50 与 Pearson top50/top100 不是完全同质的信号，它补充了排序维度的信息。但该分支很快进入局部平台，继续做同轴权重微调收益有限。

### 4.2 Residual selection 与 rolling stability

为了进一步寻找互补信号，本文尝试了 residual feature selection 和 rolling-stability-aware feature selection。

Residual selection 的思想是先用当前 best 模型得到训练集预测 $\hat{y}$，再计算残差：

$$
r_i = y_i - \hat{y}_i
$$

然后寻找与残差 $r$ 高相关的特征，希望捕捉当前模型尚未解释的信息。rolling-stability-aware selection 则将训练集按时间顺序分段，寻找在多个时间窗口中相关性较稳定的特征。

实验结果显示，这两类方法在本地 holdout 或 public leaderboard 上有时能取得改善，但 private 转移不稳定。特别是 residual-heavy blend 往往带来 public 提升但 private 下降。这说明，在该任务后期，简单追求本地残差或局部稳定性并不能保证隐藏测试集泛化。

### 4.3 Correlation medoid 与 SHAP-stable XGBoost

受高分方案启发，本文进一步尝试从特征冗余结构出发构造新的信号。首先计算 feature-feature correlation，并以 $1 - |\rho(x_i, x_j)|$ 作为距离，对高度相关特征进行聚类。每个 cluster 中选取 medoid feature，即与同 cluster 其他成员绝对相关性总和最高的特征，作为该特征簇的代表。

单独使用 medoid features 训练 Ridge 并未取得理想结果：

| 分支                    |  Public | Private | 结论        |
| --------------------- | ------: | ------: | --------- |
| Medoid Ridge blend    | 0.05437 | 0.09565 | 直接作为信号不转移 |
| SHAP-stable XGB blend | 0.05634 | 0.10043 | 小幅真实提升    |

虽然 medoid Ridge 作为直接信号失败，但 medoid features 为后续 XGBoost + SHAP 稳定筛选提供了更干净的输入空间。本文在 purged group folds 中训练 XGBoost，并利用 TreeSHAP 提取各 fold 的重要特征，再选择在多个 fold 中稳定出现的特征。基于这些特征训练的 SHAP-stable XGBoost 小权重加入后，将 private score 从 0.10003 提升到 0.10043。这是本文第一个有效的非线性/树模型补充信号。

---

## 5. Symbolic interaction 特征工程

### 5.1 Interaction 生成与筛选

在特征筛选的基础上，本文进一步构造 symbolic interaction features。基础特征池由以下特征组成：

1. SHAP-stable features；
2. Spearman top50 中排名靠前的特征；
3. Pearson top50/top100 中排名靠前的特征。

去重后，核心特征池控制在 40 个特征左右。对于任意两个核心特征 $x$ 和 $y$，生成如下二阶 interaction：

$$
x+y,\quad x-y,\quad x\cdot y,\quad
\frac{x}{|y|+\epsilon},\quad
\frac{y}{|x|+\epsilon},\quad
\max(x,y),\quad
\min(x,y)
$$

其中，$\epsilon=10^{-6}$ 用于避免除零问题。

![图 1：特征工程漏斗](figures/01_main/feature_funnel.png)

图 1 展示了主任务中的特征空间变化：原始 785 个可用特征先被压缩为 40 个核心特征，再生成 5460 个二阶 interaction 候选，最终筛选出 120 个 interaction features，并与 40 个 core features 共同构成 160 维结构化输入。该过程体现了本文的核心建模思路：不是直接扩大模型输入，而是在受控候选空间中构造可解释、可筛选的新特征。

生成的候选 interaction 通过以下 train-only 分数筛选：

1. 与标签的绝对 Pearson correlation；
2. 与标签的抽样 Spearman correlation；
3. 与当前 best 残差的绝对 Pearson correlation；
4. 候选特征之间的高相关去重。

最终保留 120 个 interaction features，分别训练 Ridge 和 XGBoost，并与当前 best 进行小权重融合。

![图 2：入选 interaction 操作类型分布](figures/01_main/interaction_operator_distribution.png)

图 2 展示了最终 120 个 selected interaction features 的操作类型分布。不同操作类型共同入选，说明有效 interaction 并不只来自单一乘法或比值形式，而是来自多种二阶组合对原始匿名特征的补充表达。

### 5.2 Interaction Ridge 结果

Interaction 分支带来了本文最大的一次结构性提升。

| 分支                           |  Public | Private | 结论      |
| ---------------------------- | ------: | ------: | ------- |
| XGB interaction blend        | 0.05657 | 0.10044 | 基本持平    |
| Ridge interaction-only blend | 0.06362 | 0.10302 | 最大结构性提升 |

结果表明，interaction features 确实补充了原始 top-k 信号。值得注意的是，XGBoost 在 interaction 特征上并不强，真正有效的是 Ridge interaction-only signal。这说明，在经过手工构造的 interaction 空间中，线性模型能够稳定捕捉一部分新的可转移结构，而更复杂的树模型反而容易引入噪声。

### 5.3 Interaction refinement 与边界

为验证 interaction 分支是否仍有空间，本文进一步进行了 Beta5-B 和 Sprint-A。Beta5-B 将 interaction 数量扩展到 240，并提高 interaction signal 权重；Sprint-A 则用 purged fold-stable scoring 重新筛选 interaction features。

代表性结果如下：

| Candidate                                 |  Public | Private |
| ----------------------------------------- | ------: | ------: |
| 240 interactions, weight 0.20             | 0.06797 | 0.10256 |
| Sprint-A top250_min3_fill120, weight 0.05 | 0.06637 | 0.10514 |
| Sprint-A top250_min4_fill120, weight 0.15 | 0.06766 | 0.10437 |

这些结果说明，interaction 方向确实有效，但简单扩大 interaction 数量或提高权重会更多追逐 public/holdout，而不一定改善 private。最终本文保留 Beta5-A 的 120 interaction Ridge blend 作为主线组件。

---

## 6. Representation 与 neural signals

### 6.1 AutoEncoder features

为了测试低维表征学习是否能进一步提取隐藏结构，本文在 40 个 core features 和 120 个 selected interactions 的 160 维结构化输入上训练 AutoEncoder。模型将输入压缩到 8、16 或 32 维 bottleneck，并尝试了 denoising 和 supervised auxiliary head。

无监督 AE 的结果不理想。虽然部分 AE signal 能提升 holdout/public，但 private 明显下降。

| 分支                           |  Public | Private | 结论         |
| ---------------------------- | ------: | ------: | ---------- |
| AE ridge, weight 0.15        | 0.06488 | 0.10169 | private 下降 |
| AE ridge, weight 0.10        | 0.06454 | 0.10225 | 仍低于当前 best |
| supervised AE8, weight 0.025 | 0.06454 | 0.10303 | 极小提升       |

后续实验显示，扩大 bottleneck、加入轻度 denoising 或更换 seed 都没有改善 private。将 AE 改为 supervised AE，即在重构输入的同时用 latent representation 辅助预测标准化目标，能产生极弱但可转移的信号。最终 supervised AE8 只以 0.025 的极小权重加入模型。

这说明，AE representation 在当前实现下不能作为主力信号，只能作为非常保守的辅助组件。

### 6.2 Supervised MLP signal

在 AE 之后，本文进一步训练 supervised MLP。MLP 输入仍为 160 维结构化特征，网络结构包括 `[256,128,64]` 和 `[128,64,32]` 两类隐藏层设置，优化器比较了 SGD 和 AdamW。损失函数为 MSE 与 Pearson loss 的加权组合：

$$
\mathcal{L} =
0.6 \cdot \text{MSE}(y,\hat{y})
+
0.4 \cdot (1-\rho(y,\hat{y}))
$$

其中，

$$
\text{MSE}(y,\hat{y})=
\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

MLP 的定位不是替代当前 best，而是生成低权重 nonlinear complementary signal。

关键结果如下：

| Candidate                     |  Public | Private | 结论                  |
| ----------------------------- | ------: | ------: | ------------------- |
| AdamW seed mean, weight 0.025 | 0.06588 | 0.10411 | 保守提升                |
| AdamW seed2026, weight 0.075  | 0.06553 | 0.10550 | 最终 best             |
| AdamW seed3026, weight 0.10   | 0.06930 | 0.10304 | public 高但 private 低 |

后续 Sprint-B 对 AdamW MLP 进行 seed stability 检查，结果显示新 seed 和 seed mean 均未复现 seed2026 的 private 增益。因此，MLP 分支虽然有效，但具有 seed sensitivity。最终模型保留 `wide_adamw_lr001_seed2026` 的 0.075 小权重，而不是使用多 seed 平均。

---

## 7. 最终模型与融合策略

### 7.1 最终模型结构

最终模型不是单一模型，而是由多阶段互补信号逐步叠加得到。其结构可以写成如下递推形式：

$$
\text{beta3} =
0.3825 \cdot \text{top50\_ridge}_{\alpha=200000}
+
0.3825 \cdot \text{top100\_ridge}_{\alpha=50}
+
0.2350 \cdot \text{spearman\_top50\_ridge}
$$

$$
\text{beta4} =
0.75 \cdot \text{beta3}
+
0.25 \cdot \text{shap\_stable\_xgboost}
$$

$$
\text{beta5} =
0.85 \cdot \text{beta4}
+
0.15 \cdot \text{ridge\_interactions\_only}
$$

$$
\text{beta6\_2} =
0.975 \cdot \text{beta5}
+
0.025 \cdot \text{supervised\_ae8\_ridge}
$$

$$
\text{final} =
0.925 \cdot \text{beta6\_2}
+
0.075 \cdot \text{wide\_adamw\_mlp\_seed2026}
$$

![图 3：最终多信号融合结构](figures/01_main/final_pipeline.png)

图 3 总结了最终模型结构。可以看到，最终预测并不是由单一复杂模型给出，而是由 Ridge top-k 主干、SHAP-stable XGBoost、interaction Ridge、supervised AE 和 supervised MLP 按阶段叠加得到。

最终提交在 Kaggle leaderboard 上得到：

| 指标              |      分数 |
| --------------- | ------: |
| Public Pearson  | 0.06553 |
| Private Pearson | 0.10550 |

根据最终榜单截图，该结果在 private leaderboard 上约为第 14 名；public 排名明显低于 private 排名。public 与 private 排名的差异说明，最终模型不是 public-oriented 的刷榜结果，而是在 hidden private 分布上具有更好的泛化表现。

### 7.2 小权重融合的必要性

本文后期大量实验显示，复杂信号往往存在如下模式：

1. 较大权重能提升 holdout 或 public；
2. private 不一定提升，甚至下降；
3. 小权重互补信号更容易转移到 private。

例如，larger interaction set、AE heavier blend、MLP 0.10 weight、Sprint-A/Sprint-B 候选都出现了 public 或 holdout 较好但 private 不如最终 best 的现象。因此，最终融合策略不是选择本地最优模型，而是选择已验证的低权重互补信号。

从建模角度看，这一策略相当于将最终预测分解为强主干信号与多个弱补充信号：

$$
\hat{y}_{\text{final}} =
\sum_{m=1}^{M} w_m \hat{y}^{(m)}, \quad
\sum_{m=1}^{M} w_m = 1
$$

其中，$\hat{y}^{(m)}$ 表示不同信号族的预测，$w_m$ 为融合权重。有效的补充信号不需要单独很强，但需要与主模型存在一定差异，并且在 private 上能提供稳定增益。

---

## 8. 实验结果与对比分析

### 8.1 阶段性结果

主任务最终提升过程如下：

| 阶段                      |  Public | Private | 说明                           |
| ----------------------- | ------: | ------: | ---------------------------- |
| Beta2 Ridge             | 0.05160 | 0.09638 | strong linear top-k baseline |
| Beta3 Spearman          | 0.05456 | 0.10003 | rank-based feature selection |
| Beta4 SHAP-stable XGB   | 0.05634 | 0.10043 | tree-based stable signal     |
| Beta5 Interaction Ridge | 0.06362 | 0.10302 | largest structural gain      |
| Beta6.2 Supervised AE   | 0.06454 | 0.10303 | tiny representation gain     |
| Beta7 MLP               | 0.06553 | 0.10550 | final nonlinear correction   |

![图 4：主任务 private 分数阶梯](figures/01_main/score_ladder.png)

图 4 展示了 private score 从 Beta2 到 Beta7 的逐步提升。该图说明最终结果来自多类信号的持续累积，而不是某一次模型替换带来的偶然提升。

### 8.2 Public/private divergence

![图 5：Public 与 private 分数关系](figures/01_main/public_private_scatter.png)

从 public/private scatter 可以看出，public 分数较高的候选不一定 private 更高。例如，部分 heavy MLP、wide interaction 和 AE 候选在 public 上表现更强，但 private 明显低于最终 best。这说明 public leaderboard 对模型选择存在误导风险，尤其是在后期模型差异较小、候选高度依赖细微信号时。

因此，本文最终不采用 public 最高的候选，而是选择在 private 已验证、且融合权重较保守的模型结构。

### 8.3 Component correlation 与互补性

![图 6：验证集组件预测相关性](figures/01_main/component_correlation_heatmap.png)

component correlation heatmap 用于观察不同信号族预测之间的相关性。若所有组件高度同质，则 ensemble 很难带来额外收益；若组件差异过大且单体不稳定，也可能引入噪声。本文最终有效组件大多与主模型存在较高但非完全一致的相关性，说明它们并不是完全独立模型，而是主干信号上的结构性修正。

---

## 9. 模型分析与失败实验

### 9.1 成功信号分析

本文最终保留的主要信号包括：

1. Pearson top-k Ridge：提供强线性基线；
2. Spearman top50 Ridge：补充排序关系；
3. SHAP-stable XGBoost：提供树模型稳定特征信号；
4. symbolic interaction Ridge：提供最大结构性增益；
5. supervised AE8：提供极小表征修正；
6. supervised MLP：提供低权重非线性修正。

其中，interaction Ridge 是后期最重要的结构性提升。它说明，在匿名特征缺乏业务语义的情况下，通过统计筛选得到的二阶组合可以构造出新的有效特征空间。

### 9.2 失败方向分析

本文没有回避失败实验。主要失败方向包括：

1. Full LightGBM 不如 Ridge；
2. medoid Ridge 不转移；
3. direct residual blend 不稳定；
4. SHAP-stable 规则微调未超过原始 Beta4-B；
5. 扩大到 240 interaction features 后，public 提升但 private 下降；
6. 无监督 AE 和更宽 AE 表征不转移；
7. MLP seed mean 不如单个 seed2026；
8. 后期 Sprint-A/Sprint-B 均未超过最终 best。

![图 7：本地 holdout 提升与 private 提升对比](figures/01_main/holdout_private_delta.png)

这些失败实验说明，该任务的模型选择不能只依赖本地 holdout 或 public 分数。许多更复杂、更重的信号会提高局部指标，却降低 hidden private 泛化。

### 9.3 与第一名方案的差距

本文参考了第一名方案中的若干方法级思想，包括 correlation clustering、SHAP-stable feature selection、symbolic interactions、AE features 和 MLP signals。本文已经复现了这些方法方向，但没有完全复现第一名的最终效果。

差距主要体现在以下方面：

1. 第一名的 symbolic feature engineering 可能更深入，包括更复杂的二阶、三阶组合和 feature recycling；
2. 第一名的 MLP 是主模型，而本文 MLP 只能作为低权重补充信号；
3. 本文 AE features 没有形成强 standalone representation；
4. 后期本地验证和 public leaderboard 难以可靠排序 private 候选，说明验证体系仍有局限。

因此，本文的主要贡献不是复现第一名，而是在有限时间内构建了一条可复现、可解释、可分析的建模路线，并系统验证了哪些信号能够转移到 private，哪些方向只是 public/holdout trap。

---

## 10. 总结

本文围绕 Kaggle DRW Crypto Market Prediction 主任务，构建了一个从强线性基线到多信号融合的完整预测系统。实验表明，在高维匿名金融特征场景下，Ridge 等简单线性模型在经过严格特征筛选后非常强；复杂模型并不能直接替代线性模型，而应作为低权重互补信号加入。

本文的主要发现包括：

1. Pearson/Spearman top-k Ridge 是稳定强基线；
2. SHAP-stable XGBoost 可提供小幅树模型补充；
3. symbolic interaction features 是最大的结构性增益来源；
4. AutoEncoder 表征在当前实现下效果有限，仅 supervised AE8 有极小收益；
5. supervised MLP 可提供非线性补充，但存在 seed sensitivity；
6. public/private divergence 明显，模型选择不能只追 public 或 holdout；
7. 最终有效策略是多类互补信号的小权重保守融合。

![图 8：不同信号族的转移效果总结](figures/01_main/signal_family_summary.png)

最终模型在 private leaderboard 上达到 0.10550，private 排名约第 14 名。该结果说明，本文方法虽然未完全复现第一名方案，但已经在隐藏测试集上表现出较强泛化能力。后续若继续扩展，可重点研究更系统的 feature recycling、更复杂但可控的 symbolic regression，以及更稳定的 supervised representation learning。

---

## 图表清单

| 图号 | 文件 | 用途 |
| --- | --- | --- |
| 图 1 | `figures/01_main/feature_funnel.png` | 展示特征工程漏斗 |
| 图 2 | `figures/01_main/interaction_operator_distribution.png` | 展示 interaction 操作类型分布 |
| 图 3 | `figures/01_main/final_pipeline.png` | 展示最终模型融合结构 |
| 图 4 | `figures/01_main/score_ladder.png` | 展示 private 分数阶段性提升 |
| 图 5 | `figures/01_main/public_private_scatter.png` | 展示 public/private divergence |
| 图 6 | `figures/01_main/component_correlation_heatmap.png` | 展示组件预测相关性 |
| 图 7 | `figures/01_main/holdout_private_delta.png` | 展示本地提升与 private 提升不一致 |
| 图 8 | `figures/01_main/signal_family_summary.png` | 总结各信号族成败 |
