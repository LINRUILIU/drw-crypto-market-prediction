teacher's question:
大作业内容（从下面三个中任选其一）：
1. 作业内容见链接，有task1、task2、task1&2。内容链接如下： https://erel-mir.github.io/challenge/mmctr-track2/，数据链接以及测试链接：https://www.codabench.org/competitions/5372/。学生通过my submissions 上传结果进行测试。
2. https://www.kaggle.com/competitions/drw-crypto-market-prediction/overview 官网题目（如需要提交测试在报告中呈现成绩截图）加上学生自行命题其他1-2个任务
3. https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview 官网题目（如需要提交测试在报告中呈现成绩截图）加上学生自行命题其他1-2个任务


该大作业是小组作业，学生自行组队完成，每组成员不超过3人。

作业上传形式包括代码，报告（报告中附上结果截图），PPT等。每个小组需提交8分钟左右的PPT报告视频，录制好上传。

书面报告需上传PDF格式，其内容格式为论文格式，需包含摘要、背景、项目简介、模型方法、实验结果（对比实验、模型实验）、模型分析、总结等部分。

评分标准分为演讲得分，论文报告得分，评估得分，创新得分。

对于参加比赛获奖情况较好的同学，可以在原有的参赛项目上进行扩展，作为大作业提交。注意：除了比赛题目，你需要其他定义新的数学建模问题来解决，上传内容和报告内容需严格按照作业格式来完成。（论文报告内容可以包含之前的比赛内容，但需另外提供一个pdf文件说明比赛题目，比赛获奖情况，以及扩展问题）

my choice: 2.DRW - Crypto Market Prediction，单人成组。按照主题目+扩展子题展开。

plan from chatGPT:
可以。下面我直接给你一个**定版方案**。这个版本的核心原则是：**主任务保证评估得分，三个扩展问题服务论文报告和创新得分，不让工作量失控。**

---

# 一、最终题目定版

## 中文题目

**高维匿名特征下的加密货币短期收益预测与信号稳定性分析**

## 英文题目

**Short-Term Cryptocurrency Return Prediction under High-Dimensional Anonymous Features: Feature Selection, Temporal Stability and Signal Interpretation**

这个题目够稳。它同时覆盖：

* 官方任务：加密货币短期收益预测；
* 数学建模问题：高维特征、降维、泛化、时间稳定性；
* 金融解释问题：预测值是否能形成方向性信号。

DRW 官方题目目标是基于生产特征数据预测加密货币短期未来价格/收益变动，评价指标与 Pearson correlation 相关；公开题面和相关 notebook 都提到数据包含分钟级市场数据、匿名专有特征和少量公开市场特征。([Kaggle][1])

---

# 二、总研究问题定版

## 总问题

在高维匿名金融特征、强噪声和非平稳市场环境下，如何构建一个稳健的加密货币短期收益预测模型，并检验其预测信号是否具有稳定性和方向性解释价值？

这个总问题可以拆成四个问题。

---

## RQ1：官方预测问题

**如何利用 DRW 提供的匿名特征与公开市场特征，预测加密货币未来短期收益或价格变动目标，使预测结果与真实目标的 Pearson correlation 尽可能高？**

对应报告部分：

* 主任务；
* 模型方法；
* 评估得分；
* Kaggle / late submission 结果截图。

---

## RQ2：高维特征筛选与降维问题

**在大量匿名、高相关、可能冗余的特征存在时，怎样筛选或压缩特征，才能在减少维度和保持预测能力之间取得平衡？**

比较对象：

* 原始全部特征；
* Pearson 相关性筛选；
* PCA；
* 相关性聚类后取代表特征；
* 相关性聚类后取簇均值；
* 可选：LightGBM importance / permutation importance / SHAP 筛选。

Elastic Net 本身适合处理高维和相关变量，原论文强调其结合 L1 与 L2 正则，并具有处理相关变量组的能力；PCA 则是经典降维方法，适合在高维连续特征中压缩主要方差信息。([OUP Academic][2])

---

## RQ3：时间稳定性与市场状态问题

**模型的预测能力是否在不同时间段保持稳定？如果不同阶段 Pearson correlation 明显变化，是否说明存在市场状态切换或分布漂移？**

比较对象：

* 前 80% 训练、后 20% 验证；
* rolling validation；
* 不同时间段 Pearson；
* 不同时间段特征重要性；
* 不同时间段 target 分布与预测分布。

金融时间序列中普通随机 K 折容易产生时间泄露或过度乐观估计，金融机器学习里常讨论 purging / embargo 这类方法来降低 look-ahead bias。你不一定完整实现复杂版，但报告里要明确采用时间顺序切分，而不是随机切分。([维基百科][3])

---

## RQ4：预测信号的方向性解释问题

**模型预测值是否具有排序意义？即预测值越高的样本，其真实未来收益是否整体更高？**

比较对象：

* top 10% 预测样本；
* middle 80% 预测样本；
* bottom 10% 预测样本；
* 各组真实 target 均值；
* top-bottom 差值；
* 方向命中率；
* 无交易成本的理论 long-short 信号检验。

这部分不要写成“真实交易策略”。应该写成：

> 本文只进行预测信号有效性检验，不考虑手续费、滑点、延迟、仓位约束和风险控制，因此不构成真实交易回测。

这个边界能防止老师质疑你“怎么证明能赚钱”。

---

# 三、数据来源定版

## 数据来源

**Kaggle：DRW - Crypto Market Prediction**

数据来源于 DRW 在 Kaggle 发布的加密货币市场预测竞赛。竞赛目标是利用历史分钟级加密货币市场数据和匿名生产特征，预测未来短期价格或收益变动。([Kaggle][1])

## 数据内容

报告中建议这样写：

| 数据部分     | 说明                                                              |
| ------------ | ----------------------------------------------------------------- |
| 训练集       | 包含历史时间顺序样本、公开市场特征、匿名专有特征、目标变量        |
| 测试集       | 包含同结构特征，不含目标变量，用于生成提交文件                    |
| 公开市场特征 | 如 bid_qty、ask_qty、buy_qty、sell_qty、volume 等市场微观结构变量 |
| 匿名特征     | DRW 提供的高维匿名特征，通常命名为 X 类特征                       |
| 目标变量     | 未来短期加密货币价格或收益变动的匿名化目标                        |
| 提交文件     | 对测试集每条样本生成预测值                                        |

公开 notebook 中提到该任务使用约 780 个匿名专有特征和 5 个公开市场特征；也有社区复现项目记录了 525K+ 训练样本和大量匿名特征。但不同社区资料对具体特征数量写法略有差异，最终报告里要以你实际下载的数据列数为准。([Kaggle][4])

---

# 四、评价指标定版

## 主评价指标：Pearson correlation

官方主指标使用预测值与真实目标之间的 Pearson correlation。你的报告中应写公式：

[
\rho(\hat{y}, y)=
\frac{\sum_{i=1}^{n}(\hat{y_i}-\bar{\hat{y}})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(\hat{y_i}-\bar{\hat{y}})^2}
\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
]

其中：

* (y_i)：真实目标；
* (\hat{y_i})：模型预测值；
* (\bar{y})：真实目标均值；
* (\bar{\hat{y}})：预测值均值。

## 辅助评价指标

| 指标                                  | 用途                         |
| ------------------------------------- | ---------------------------- |
| Validation Pearson                    | 主要离线验证指标             |
| Kaggle Public / Late Submission Score | 官方平台评估截图             |
| MSE / RMSE                            | 作为误差辅助指标，不作为核心 |
| Top-bottom target spread              | 检验预测信号排序意义         |
| Directional accuracy                  | 检验方向命中率               |
| Rolling Pearson std                   | 衡量时间稳定性               |

注意：**Pearson 是核心，不要让 MSE 抢戏。**
金融预测里预测值绝对大小未必最重要，排序和相关性更重要。

---

# 五、实验内容定版

你的实验矩阵不要太复杂。建议这样设计。

## 实验 0：数据探索与预处理

目标：理解数据结构，为后续建模提供依据。

内容：

1. 样本数量、特征数量；
2. 缺失值比例；
3. target 分布；
4. 公开市场特征分布；
5. 匿名特征均值、方差、相关性；
6. 特征与 target 的 Pearson 相关性；
7. 时间维度上的 target 均值、方差变化。

输出结果：

* 数据基本信息表；
* target 分布图；
* 特征相关性热力图；
* top-k 特征相关性柱状图。

---

## 实验 1：主任务预测模型对比

目标：完成官方预测任务，获得可提交结果。

模型：

| 模型                      | 作用                              |
| ------------------------- | --------------------------------- |
| Ridge                     | 强基线，稳定，适合高维连续特征    |
| ElasticNet                | 正则化 + 变量选择，对相关特征友好 |
| LightGBM                  | 主力非线性模型                    |
| XGBoost                   | 树模型对照                        |
| CatBoost                  | 可选，不一定必须做                |
| Ridge + LightGBM 加权融合 | 最终提交模型                      |

LightGBM 和 XGBoost 都是成熟的梯度提升树方法，XGBoost 论文强调其可扩展树提升系统，LightGBM 论文强调其在高维和大规模数据上的效率；CatBoost 主要优势是 ordered boosting 和对 categorical features 的处理，不过本题大多是连续匿名特征，所以它可以作为对照，不是必须主力。([KDD][5])

输出结果：

| 模型              | 验证 Pearson |  MSE | 训练时间 | 特征数量 | 备注       |
| ----------------- | -----------: | ---: | -------: | -------: | ---------- |
| Ridge             |              |      |          |          | baseline   |
| ElasticNet        |              |      |          |          | 正则化筛选 |
| LightGBM          |              |      |          |          | 主力模型   |
| XGBoost           |              |      |          |          | 对照模型   |
| Weighted Ensemble |              |      |          |          | 最终提交   |

---

## 实验 2：高维特征筛选与降维实验

目标：回答 RQ2。

实验组：

| 实验组 | 方法                      | 目的                           |
| ------ | ------------------------- | ------------------------------ |
| A      | 全部特征                  | 原始 baseline                  |
| B      | Pearson top-k             | 保留与 target 相关性最高的特征 |
| C      | PCA                       | 线性降维，压缩方差信息         |
| D      | 相关性聚类 + 代表特征     | 去冗余                         |
| E      | 相关性聚类 + 簇均值       | 降低噪声                       |
| F      | LightGBM importance top-k | 模型驱动筛选                   |
| G      | SHAP top-k                | 可解释性筛选，可选             |

SHAP 的原论文提出用 Shapley values 统一解释模型预测，适合用来做树模型的特征重要性解释；但它计算较重，所以建议只抽样做，不要全量做。([arXiv][6])

建议的 k 值：

* top 50；
* top 100；
* top 200；
* top 300。

输出结果：

| 特征方案       | 特征数 | Ridge Pearson | LightGBM Pearson | 训练时间 | 结论 |
| -------------- | -----: | ------------: | ---------------: | -------: | ---- |
| 全部特征       |        |               |                  |          |      |
| Pearson top100 |    100 |               |                  |          |      |
| PCA 100维      |    100 |               |                  |          |      |
| 聚类代表特征   |        |               |                  |          |      |
| 聚类簇均值     |        |               |                  |          |      |

核心分析点：

> 降维不一定提高最高分，但可以降低训练成本、减少冗余、提高稳定性。
> 如果 Ridge 在筛选后提升明显，说明线性模型受噪声和冗余影响较大。
> 如果 LightGBM 对全部特征更强，说明树模型能自动处理一部分非线性和冗余结构。

---

## 实验 3：时间稳定性实验

目标：回答 RQ3。

实验设计：

1. 按时间顺序划分：

   * 前 80% 训练；
   * 后 20% 验证。

2. rolling validation：

   * fold 1：前 50% 训练，后 10% 验证；
   * fold 2：前 60% 训练，后 10% 验证；
   * fold 3：前 70% 训练，后 10% 验证；
   * fold 4：前 80% 训练，后 10% 验证。

3. 可选 embargo：

   * 训练集和验证集之间空出若干分钟或若干样本，降低边界泄露风险。

输出结果：

| 时间段 | Ridge Pearson | LightGBM Pearson | Ensemble Pearson | target std | 结论 |
| ------ | ------------: | ---------------: | ---------------: | ---------: | ---- |
| Fold 1 |               |                  |                  |            |      |
| Fold 2 |               |                  |                  |            |      |
| Fold 3 |               |                  |                  |            |      |
| Fold 4 |               |                  |                  |            |      |

图：

* rolling Pearson 折线图；
* 不同时间段 target 分布；
* 不同时间段模型预测分布；
* 特征重要性随时间变化图。

核心分析点：

> 如果不同时间段 Pearson 波动较大，说明加密货币短期收益预测存在明显非平稳性。模型在某一阶段有效，不代表在全部市场状态下稳定有效。

---

## 实验 4：预测信号解释实验

目标：回答 RQ4。

实验设计：

把验证集样本按预测值排序：

| 分组       | 样本             |
| ---------- | ---------------- |
| Bottom 10% | 预测值最低的 10% |
| Middle 80% | 中间样本         |
| Top 10%    | 预测值最高的 10% |

计算：

[
\text{Spread} = mean(y_{top10}) - mean(y_{bottom10})
]

方向命中率：

[
Acc_{direction} = \frac{1}{n}\sum_{i=1}^{n} I(sign(\hat{y_i})=sign(y_i))
]

输出结果：

| 分组       | 样本占比 | 平均预测值 | 平均真实 target | 方向命中率 |
| ---------- | -------: | ---------: | --------------: | ---------: |
| Bottom 10% |      10% |            |                 |            |
| Middle 80% |      80% |            |                 |            |
| Top 10%    |      10% |            |                 |            |

核心分析点：

> 如果 top 10% 的真实 target 均值明显高于 bottom 10%，说明模型预测值不仅有统计相关性，也具有一定的排序和方向解释意义。

---

# 六、结果要求定版

最后你至少要拿到这些结果。

## 必须结果

1. **一次 Kaggle / late submission 截图**
   用于报告和 PPT。

2. **模型对比表**
   Ridge、ElasticNet、LightGBM、XGBoost、Ensemble。

3. **特征筛选对比表**
   全特征、Pearson top-k、PCA、聚类代表特征、聚类簇均值。

4. **rolling validation 表和图**
   证明你考虑了时间稳定性。

5. **预测信号分组图**
   top / middle / bottom 真实 target 均值。

6. **最终模型流程图**
   数据 → 预处理 → 特征筛选 → 模型训练 → 融合 → 提交 → 分析。

## 理想结果

如果你能做到这些，报告会明显更强：

* LightGBM 或 Ridge 至少有一个稳定正 Pearson；
* 融合模型略优于单模型；
* 特征筛选后训练时间下降；
* top 10% 的真实 target 均值高于 bottom 10%；
* rolling validation 显示不同时间段表现存在波动。

---

# 七、技术路线定版

## 主技术路线

```text
数据读取
  ↓
数据探索 EDA
  ↓
缺失值处理 / 数据类型压缩
  ↓
时间顺序切分
  ↓
基础模型训练：Ridge / ElasticNet
  ↓
树模型训练：LightGBM / XGBoost
  ↓
特征筛选与降维实验
  ↓
rolling validation 时间稳定性分析
  ↓
加权融合
  ↓
Kaggle 提交
  ↓
预测信号分组解释
  ↓
论文报告 + PPT 视频
```

## 最低可行版本

你先完成这个：

```text
读取 train/test
→ 时间 80/20 切分
→ Ridge baseline
→ LightGBM baseline
→ Pearson 评价
→ 生成 submission
→ Kaggle 提交截图
```

这个完成后，作业就不会崩。

## 标准完成版本

然后做：

```text
Pearson top100 / top200 特征筛选
→ Ridge 和 LightGBM 重跑
→ rolling validation
→ top-bottom 信号分组
```

## 加分版本

最后再做：

```text
相关性聚类降维
→ PCA 对比
→ LightGBM importance
→ SHAP 抽样解释
→ 简单加权融合
```

---

# 八、重要论文与资料清单

你不用读很多。下面这些够用了。

## 1. 比赛与数据来源

| 用途                 | 资料                                          |
| -------------------- | --------------------------------------------- |
| 官方题目、数据、提交 | Kaggle DRW - Crypto Market Prediction         |
| baseline 参考        | Kaggle notebooks / GitHub community solutions |

DRW 竞赛题面说明其目标是预测加密货币短期未来价格变动；Kaggle 数据页说明数据是分钟级历史加密货币市场数据。([Kaggle][1])

---

## 2. 机器学习模型

| 方法        | 论文 / 资料                | 用途           |
| ----------- | -------------------------- | -------------- |
| XGBoost     | Chen & Guestrin, 2016      | 树模型对照     |
| LightGBM    | Ke et al., 2017            | 主力树模型     |
| CatBoost    | Prokhorenkova et al., 2018 | 可选对照       |
| Elastic Net | Zou & Hastie, 2005         | 高维正则化基线 |
| SHAP        | Lundberg & Lee, 2017       | 模型解释       |

XGBoost 是可扩展树提升系统，LightGBM 强调高效 GBDT，CatBoost 强调 ordered boosting 和 categorical feature 支持，SHAP 用 Shapley values 解释模型输出。([KDD][5])

---

## 3. 降维与高维建模

| 方法        | 资料                                   | 用途             |
| ----------- | -------------------------------------- | ---------------- |
| PCA         | Jolliffe, Principal Component Analysis | 降维理论         |
| Elastic Net | Zou & Hastie                           | 高维相关特征筛选 |
| 相关性聚类  | 层次聚类 / correlation clustering 思路 | 去冗余           |

PCA 是经典主成分降维方法；Elastic Net 对高维和相关变量组有优势，适合解释为什么你不仅用 Ridge，还考虑 ElasticNet。([cda.psych.uiuc.edu][7])

---

## 4. 金融时间序列验证

| 方法                    | 资料                       | 用途 |
| ----------------------- | -------------------------- | ---- |
| walk-forward validation | 金融时间序列常规验证       |      |
| purged CV / embargo     | 降低金融标签重叠导致的泄露 |      |
| rolling validation      | 检验时间稳定性             |      |

Purged cross-validation 被用于金融时间序列和结构化数据，核心目的是避免由于时间重叠或标签依赖未来窗口带来的 look-ahead bias。([维基百科][3])

---

## 5. 加密货币预测背景

| 资料                                        | 用途                                   |
| ------------------------------------------- | -------------------------------------- |
| Cryptocurrency price prediction survey      | 背景综述                               |
| Cryptocurrency return prediction ML paper   | 支撑树模型、机器学习用于 crypto return |
| XGBoost / LightGBM crypto prediction papers | 支撑模型选择                           |

近年的综述文献显示，加密货币预测研究大量使用机器学习和深度学习方法；也有研究比较 ensemble learning、deep learning、LightGBM 等方法在加密货币预测中的表现。([MDPI][8])

---

# 九、已有 baseline 方向

你可以参考但不要直接照抄。

## baseline 1：Ridge + 特征工程

社区复现项目中有方案使用 Ridge、相关性筛选和简单特征工程，并报告了相对不错的验证相关性；这说明本题不一定需要复杂深度学习，线性模型在高质量匿名特征上可能很强。([GitHub][9])

适合你采用：

* Ridge；
* top-k correlation features；
* 市场微观结构特征；
* 简单加权融合。

---

## baseline 2：LightGBM / XGBoost

Kaggle notebooks 和社区方案普遍会尝试 LightGBM、XGBoost、ensemble。LightGBM 适合表格型高维数据，XGBoost 适合作为稳健对照。([Kaggle][10])

适合你采用：

* LightGBMRegressor；
* XGBRegressor；
* early stopping；
* 时间切分验证；
* 特征重要性分析。

---

## baseline 3：加权融合

简单融合公式：

[
\hat{y}*{ens} = w_1\hat{y}*{ridge} + w_2\hat{y}*{lgbm} + w_3\hat{y}*{xgb}
]

建议先用：

[
\hat{y}*{ens} = 0.5\hat{y}*{ridge} + 0.5\hat{y}_{lgbm}
]

然后再网格搜索：

[
w \in {0,0.1,0.2,\ldots,1.0}
]

[
\hat{y}*{ens}=w\hat{y}*{ridge}+(1-w)\hat{y}_{lgbm}
]

不要一开始做复杂 stacking。对课程作业来说，**简单融合更可解释，也更容易写进报告。**

---

# 十、并行优化方向

这里分成“可以并行做”和“后期叠加做”。

## A. 可以并行做的优化方向

这些互不冲突，可以同时推进。

### 方向 1：特征筛选

* Pearson top-k；
* LightGBM importance top-k；
* 去掉低方差特征；
* 去掉高度相关冗余特征。

### 方向 2：模型对比

* Ridge；
* ElasticNet；
* LightGBM；
* XGBoost。

### 方向 3：时间验证

* 80/20 split；
* rolling validation；
* 不同时间段 score 对比。

### 方向 4：信号解释

* top-bottom 分组；
* 方向命中率；
* 分位数 target 均值图。

这些方向可以分别产出报告图表。

---

## B. 后期叠加优化方向

这些要等 baseline 跑通后再做。

### 优化 1：特征工程

公开市场特征可以构造：

[
order_imbalance = \frac{bid_qty - ask_qty}{bid_qty + ask_qty + \epsilon}
]

[
trade_imbalance = \frac{buy_qty - sell_qty}{buy_qty + sell_qty + \epsilon}
]

[
liquidity = bid_qty + ask_qty
]

[
volume_ratio = \frac{buy_qty + sell_qty}{volume + \epsilon}
]

社区方案也常基于 bid/ask、buy/sell、volume 构造订单流、流动性和成交比例类特征。([GitHub][11])

### 优化 2：特征聚类

流程：

```text
计算特征相关系数矩阵
→ 距离 = 1 - |corr|
→ 层次聚类
→ 每簇选择与 target 相关性最高的特征
→ 或者每簇取均值
→ 重训 Ridge / LightGBM
```

这个是你最有“数学建模感”的创新点。

### 优化 3：模型融合

先做简单加权，不做复杂 stacking。

可比较：

* Ridge；
* LightGBM；
* XGBoost；
* Ridge + LightGBM；
* Ridge + LightGBM + XGBoost。

### 优化 4：近期样本加权

如果发现后期数据和测试集更接近，可以做时间衰减权重：

[
w_t = \exp(\lambda \cdot \frac{t}{T})
]

越靠后的样本权重越高。

但这属于后期优化。报告里可以作为“非平稳市场下的时间加权建模尝试”。

### 优化 5：预测值后处理

可以尝试：

* 标准化预测；
* clipping 极端预测值；
* rank transform；
* ensemble 前统一尺度。

注意不要过度调 leaderboard。报告中只写“为降低极端值影响，对预测值进行标准化或截尾处理”。

---

# 十一、最终实验优先级

你按这个顺序做，不要乱。

## 第一优先级：保底完成

1. 下载数据；
2. 数据读取；
3. 时间顺序 80/20；
4. Pearson metric；
5. Ridge baseline；
6. LightGBM baseline；
7. 生成 submission；
8. Kaggle 提交截图。

## 第二优先级：报告变强

1. Pearson top-k 特征筛选；
2. PCA 降维；
3. 相关性聚类代表特征；
4. rolling validation；
5. top-bottom 信号解释。

## 第三优先级：创新加分

1. 聚类簇均值；
2. SHAP 抽样解释；
3. 时间衰减权重；
4. 简单融合权重搜索；
5. 预测信号分位数组合图。

---

# 十二、最终定版摘要逻辑

你的报告可以围绕这一句话展开：

> 本文以 DRW Crypto Market Prediction 为基础，研究高维匿名特征条件下的加密货币短期收益预测问题。首先构建 Ridge、ElasticNet、LightGBM、XGBoost 等基线模型完成官方预测任务；其次比较 Pearson 筛选、PCA、相关性聚类等特征筛选与降维方法；然后通过时间顺序切分和滚动验证分析模型在不同市场阶段的稳定性；最后通过预测值分组检验模型信号的方向性解释价值。实验结果从预测精度、特征压缩、时间稳定性和信号解释四个角度评估模型表现。

这个框架可以直接定死。下一步不是继续扩题，而是开始写代码跑：

**Ridge baseline + LightGBM baseline + Pearson top-k 特征筛选。**
这三件事跑通后，你的大作业就已经进入可控状态。

[1]: https://www.kaggle.com/competitions/drw-crypto-market-prediction?utm_source=chatgpt.com "DRW - Crypto Market Prediction"
[2]: https://academic.oup.com/jrsssb/article/67/2/301/7109482?utm_source=chatgpt.com "Regularization and Variable Selection Via the Elastic Net"
[3]: https://en.wikipedia.org/wiki/Purged_cross-validation?utm_source=chatgpt.com "Purged cross-validation"
[4]: https://www.kaggle.com/code/adilshamim8/drw-crypto-market-prediction?utm_source=chatgpt.com "DRW - Crypto Market Prediction"
[5]: https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf?utm_source=chatgpt.com "XGBoost: A Scalable Tree Boosting System"
[6]: https://arxiv.org/abs/1705.07874?utm_source=chatgpt.com "A Unified Approach to Interpreting Model Predictions"
[7]: https://cda.psych.uiuc.edu/statistical_learning_course/Jolliffe%20I.%20Principal%20Component%20Analysis%20%282ed.%2C%20Springer%2C%202002%29%28518s%29_MVsa_.pdf?utm_source=chatgpt.com "Principal Component Analysis, Second Edition"
[8]: https://www.mdpi.com/2571-9394/6/3/34?utm_source=chatgpt.com "Cryptocurrency Price Prediction Algorithms: A Survey and ..."
[9]: https://github.com/coderback/drw-crypto-market-prediction "GitHub - coderback/DRW-Crypto-Market-Prediction: DRW Crypto Market Prediction - Kaggle Competition Solution achieving 89.7% of winner performance with comprehensive EDA analysis and feature engineering pipeline · GitHub"
[10]: https://www.kaggle.com/code/ravaghi/drw-crypto-market-prediction-ensemble?utm_source=chatgpt.com "DRW Crypto Market Prediction | Ensemble"
[11]: https://github.com/Kalyan1210/DRW-Crypto-Prediction-Kaggle-Competition "GitHub - Kalyan1210/DRW-Crypto-Prediction-Kaggle-Competition · GitHub"
