# 主任务报告大纲：加密货币短期收益预测

## 写作定位

主任务已经冻结，最终主任务模型为：

```text
0.925 * beta6_2_current_best + 0.075 * wide_adamw_lr001_seed2026
```

Kaggle 分数：

| Public | Private | 状态 |
| ---: | ---: | --- |
| 0.06553 | 0.10550 | 主任务最终选择 |

其中 private 分数达到公开榜单的第 14 名。

报告不应写成“不断刷榜”的流水账，而应写成一个清晰的建模过程：

```text
高维匿名特征
-> 泄露控制与时间顺序验证
-> 线性 top-k 强基线
-> 多种特征筛选准则
-> 结构化 interaction 特征
-> 低权重非线性/表征信号
-> 保守融合与 public/private 偏差分析
```

核心论点：

1. 该任务的主要难点不是模型形式本身，而是高维匿名特征中的冗余、噪声与验证不稳定。
2. Ridge 等简单线性模型在严格特征筛选后非常强，复杂模型必须作为互补信号而不是直接替代。
3. 有效提升来自“不同信号族的小权重叠加”，而不是单个大模型。
4. public/private 与本地验证存在明显不一致，因此最终策略要强调稳健性、低权重融合和失败实验的解释。

## 摘要

建议写 250-400 字，包含：

- 问题：基于订单簿和匿名市场特征预测未来短期收益，评价指标为 Pearson correlation。
- 数据规模：训练集 `525886` 行、测试集 `538150` 行；特征为 `bid_qty`、`ask_qty`、`buy_qty`、`sell_qty`、`volume` 与 `X1-X780`，目标列为 `label`。
- 方法：Ridge/LightGBM 基线，Pearson/Spearman top-k，SHAP-stable XGB，symbolic interactions，supervised AE，supervised MLP，小权重融合。
- 结果：最终 private `0.10550`；最佳模型是多信号 ensemble。
- 结论：高维匿名金融预测更依赖稳定特征工程和验证设计，而不是盲目增加模型复杂度。

## 1. 问题背景与任务定义

### 1.1 金融短期收益预测的特点

要说明：

- 短期收益信号弱、噪声高。
- 特征匿名，不能依赖业务语义解释每个 `X` 特征。
- public/private 分布可能不同，导致本地或 public 提升不一定转化到 private。
- 因此需要可复现、可对比、可解释的实验设计。

### 1.2 预测目标与评价指标

内容：

- 目标变量：`label`，表示未来短期收益相关目标。
- 预测输出：测试集每行一个 `prediction`。
- 官方评价：Pearson correlation。
- 写出 Pearson 公式：

```text
corr(y, p) =
sum((y_i - mean(y)) * (p_i - mean(p)))
/
sqrt(sum((y_i - mean(y))^2) * sum((p_i - mean(p))^2))
```

说明：

- Pearson 只关心线性相关和方向，不直接等价于 RMSE。
- 这解释了为什么部分 RMSE/holdout 改善没有转化为 leaderboard 改善。

## 2. 数据与预处理

### 2.1 数据结构

表格：

| 数据文件 | 行数 | 列数 | 说明 |
| --- | ---: | ---: | --- |
| `train.parquet` | 525886 | 786 | 785 个特征 + `label` |
| `test.parquet` | 538150 | 786 | 测试特征，提交按 `sample_submission.csv` 对齐 |
| `sample_submission.csv` | 538150 | 2 | `ID,prediction` |

特征组成：

- 原始可命名市场特征：`bid_qty`、`ask_qty`、`buy_qty`、`sell_qty`、`volume`。
- 匿名特征：`X1-X780`。
- 目标：`label`。

### 2.2 预处理策略

写清楚：

- 只使用数值特征。
- `label` 和提交 `ID` 不进入 feature columns。
- 缺失值使用训练集 median 填充。
- 标准化使用训练集均值和标准差。
- 每个 top-k / interaction / MLP 分支单独拟合预处理器，避免验证集信息泄露。

### 2.3 验证切分与泄露控制

主验证口径：

```text
row-order chronological split: 前 80% 训练，后 20% 验证
```

扩展验证：

- rolling validation 用于时间稳定性分析。
- purged group split 用于 SHAP-stable XGB 和 Sprint-A interaction stability。

需要强调：

- 特征筛选只在 train split 上完成。
- top-k 排名、interaction scoring、AE/MLP 训练均不使用 validation label。
- Kaggle submission 只使用 test features 和 sample submission 对齐。

## 3. Baseline 建模

### 3.1 Full-feature Baseline

内容：

- 使用全部原始特征训练 Ridge、LightGBM，以及二者融合。
- 作为“未经特征筛选”的对照。

结果表：

| 模型 | Holdout Pearson |
| --- | ---: |
| Full Ridge | 0.097166 |
| Full LightGBM | 0.063721 |
| Full Ridge + LightGBM | 0.100263 |

解释：

- Ridge 明显强于单独 LightGBM，说明线性相关结构很重要。
- 直接树模型没有立刻发挥优势，后续需要更干净的特征空间。

### 3.2 Pearson Top-k Ridge

内容：

- 使用 train-only `abs(corr(feature, label))` 排名。
- 比较 top50/top100/top200/top300/top500。
- Ridge alpha 搜索。

关键结果：

- top100 Ridge 单独 private 达到 `0.08901`。
- top50/top100 blend private 达到 `0.09280`。
- 高 alpha 精调后达到 private `0.09638`。

解释：

- 简单 Pearson top-k 已经形成强基线。
- 说明高维特征中存在大量冗余和噪声，全量输入不是最优。

## 4. 特征筛选与稳健性探索

### 4.1 Spearman Top-k

动机：

- Pearson 看线性关系，Spearman 看排序关系。
- 金融信号可能更多表现为 rank/order 而非严格线性。

结果：

- Spearman top50 小权重加入后 private 到 `0.10003`。
- `0.225-0.235` 附近形成 private plateau。

解释：

- Spearman 是与 Pearson Ridge 不完全同质的补充信号。
- 继续扫权重收益有限，后续需要新信号来源。

### 4.2 Residual Selection 和 Rolling Stability

内容：

- residual Pearson：用当前 best 的残差重新找特征。
- rolling-stability Pearson：按时间段比较特征稳定性。

结论：

- 本地或 public 有时提升，但 private 转移不稳定。
- 单纯追求“验证更稳”不等于 leaderboard 更好。

### 4.3 Correlation Medoid 与 SHAP-stable XGB

动机来自高分方案：

- 先对 feature-feature correlation 聚类，降低冗余。
- 再用 XGB + SHAP 在 purged folds 中找稳定重要特征。

结果：

| 分支 | Public | Private | 结论 |
| --- | ---: | ---: | --- |
| Medoid Ridge blend | 0.05437 | 0.09565 | 直接 medoid Ridge 不转移 |
| SHAP-stable XGB blend | 0.05634 | 0.10043 | 小幅真实提升 |

解释：

- medoid 作为直接信号不强，但结构化筛选思路有价值。
- SHAP-stable XGB 是第一个有效的非线性/树模型信号。

## 5. Symbolic Interaction 特征工程

### 5.1 Interaction 生成

基础特征池：

- SHAP-stable features。
- Spearman top50 前若干。
- Pearson top50/top100 前若干。
- 去重后控制在 40 个核心特征。

二阶 interaction 操作：

```text
x + y
x - y
x * y
x / (abs(y) + 1e-6)
y / (abs(x) + 1e-6)
max(x, y)
min(x, y)
```

筛选策略：

- train-only Pearson(label)。
- train-only Pearson(current-best residual)。
- 抽样 Spearman(label)。
- 高相关去重，最终保留 120 个 interaction features。

### 5.2 Interaction Ridge 结果

关键结果：

| 分支 | Public | Private | 结论 |
| --- | ---: | ---: | --- |
| XGB interaction blend | 0.05657 | 0.10044 | 近似持平 |
| Ridge interaction-only blend | 0.06362 | 0.10302 | 最大结构性提升 |

解释：

- interaction 特征确实补充了原始 top-k 信号。
- Ridge 比 XGB 更适合当前 interaction 信号。
- 这说明构造后的线性组合比直接堆树更稳。

### 5.3 Interaction Refinement 与 Sprint-A

内容：

- 扩到 240 interactions、加大权重，public/holdout 好看但 private 下降。
- Sprint-A 改用 fold-stable interaction selection。

Sprint-A 结果：

| Candidate | Public | Private |
| --- | ---: | ---: |
| `top250_min3_fill120`, weight `0.05` | 0.06637 | 0.10514 |
| `top250_min4_fill120`, weight `0.15` | 0.06766 | 0.10437 |

结论：

- interaction 方向有效，但已到局部边界。
- 简单扩大和稳定筛选都未超过最终 best。

## 6. Representation / Neural Signals

### 6.1 AutoEncoder Features

内容：

- 输入：40 core + 120 interactions。
- AE bottleneck：8/16/32。
- 尝试 denoising 与 supervised auxiliary head。

结果：

| 分支 | Public | Private | 结论 |
| --- | ---: | ---: | --- |
| AE ridge `w0.15` | 0.06488 | 0.10169 | private 下降 |
| AE ridge `w0.10` | 0.06454 | 0.10225 | 仍低于当前 |
| supervised AE8 `w0.025` | 0.06454 | 0.10303 | 极小提升 |

解释：

- 无监督 AE 没有学到足够可转移的压缩表示。
- supervised AE 有弱信号，但只能极小权重使用。

### 6.2 Supervised MLP Signal

设置：

- 输入仍为 160 维结构化特征。
- MLP hidden `[256, 128, 64]` 或 `[128, 64, 32]`。
- loss：`0.6 * MSE + 0.4 * (1 - Pearson)`。
- 优化器比较：SGD / AdamW。
- 只作为 low-weight complementary signal。

关键结果：

| Candidate | Public | Private | 结论 |
| --- | ---: | ---: | --- |
| AdamW seed mean `w0.025` | 0.06588 | 0.10411 | 保守提升 |
| AdamW seed2026 `w0.075` | 0.06553 | 0.10550 | 最终 best |
| AdamW seed3026 `w0.10` | 0.06930 | 0.10304 | public 高但 private 低 |

Sprint-B seed stability：

| Candidate | Public | Private |
| --- | ---: | ---: |
| new-seed mean `w0.025` | 0.06573 | 0.10376 |
| seed4026 `w0.075` | 0.06944 | 0.10321 |
| seed2526 `w0.0875` | 0.06397 | 0.10205 |

结论：

- MLP 可以提供非线性补充信号，但 seed 稳定性差。
- 最终使用 seed2026 的 `0.075` 小权重，而不是平均多个 seed。

## 7. 最终模型与融合策略

### 7.1 最终模型结构

建议写成递推公式：

```text
beta3 = 0.3825 * top50_ridge_alpha200000
      + 0.3825 * top100_ridge_alpha50
      + 0.2350 * spearman_top50_ridge

beta4 = 0.75 * beta3
      + 0.25 * shap_stable_xgboost

beta5 = 0.85 * beta4
      + 0.15 * ridge_interactions_only

beta6_2 = 0.975 * beta5
        + 0.025 * supervised_ae8_ridge

final = 0.925 * beta6_2
      + 0.075 * wide_adamw_lr001_seed2026
```

### 7.2 为什么采用小权重融合

论点：

- 单个复杂信号容易提高 public/holdout，但 private 不稳。
- 有效信号通常表现为“低权重可转移”。
- 融合权重不是越大越好，尤其是 AE/MLP/interaction widening。

### 7.3 最终结果

表格：

| 阶段 | Public | Private |
| --- | ---: | ---: |
| Pearson/Spearman Ridge | 0.05456 | 0.10003 |
| + SHAP-stable XGB | 0.05634 | 0.10043 |
| + Interaction Ridge | 0.06362 | 0.10302 |
| + Supervised AE tiny | 0.06454 | 0.10303 |
| + Supervised MLP | 0.06553 | 0.10550 |

## 8. 失败实验与建模反思

### 8.1 失败方向

需要写入而不是回避：

- LightGBM full feature 不如 Ridge。
- medoid Ridge 不转移。
- residual direct blend 不稳定。
- SHAP-stable 规则微调无提升。
- int240 / 更大 interaction 权重 public 高但 private 低。
- AE 大权重 public/holdout 高但 private 差。
- MLP seed mean 不如单个 seed2026。

### 8.2 public/private divergence

重点：

- 多个实验出现 public 提升但 private 下降。
- 这说明验证体系在后期已经难以可靠排序。
- 最终模型选择更强调 private 已验证的低权重组件。

### 8.3 与第一名方案的差距

客观写：

- 已复现方法级路线：medoid、SHAP-stable、interactions、AE、MLP。
- 但未复现第一名的完整 feature recycling、复杂 symbolic regression 和高质量 representation pipeline。
- 本项目的 MLP 是补充信号，第一名 MLP 是主模型，差距主要来自特征工程深度和训练体系。

## 9. 结论

建议结论结构：

1. Ridge top-k 是高维匿名金融特征下的强基线。
2. Spearman、SHAP-stable XGB、interaction Ridge、supervised AE、MLP 分别提供了不同层面的互补信号。
3. interaction features 是最大结构性提升。
4. 表征模型并非无效，但必须低权重使用。
5. 后期 public/private 分歧明显，稳健模型选择比追逐本地最优更重要。

## 可视化清单

下一步主任务可视化建议按优先级生成。

### Figure 1: Score Ladder

目的：

- 展示 private 从 Ridge baseline 到 final 的逐步提升。

数据来源：

- `docs/main_baseline_results.md`
- `docs/main_task_retrospective.md`

形式：

- 横轴为阶段，纵轴为 private score。
- 标注关键节点：Spearman、SHAP-stable、Interaction、AE、MLP。

### Figure 2: Public vs Private Divergence

目的：

- 解释为什么不能只追 public 或 holdout。

数据来源：

- `docs/main_baseline_results.md`
- Kaggle submission table。

形式：

- scatter plot：x=public score，y=private score。
- 用颜色区分信号族：Ridge/Spearman、SHAP/XGB、Interaction、AE、MLP、Sprint。

### Figure 3: Final Ensemble Pipeline

目的：

- 用流程图解释最终模型不是单模型，而是多层信号融合。

形式：

```text
Pearson/Spearman Ridge
        |
        + SHAP-stable XGB
        |
        + Interaction Ridge
        |
        + Supervised AE8
        |
        + Supervised MLP
        v
Final prediction
```

### Figure 4: Feature Engineering Funnel

目的：

- 展示从 785 原始特征到 40 core features、120 interactions、160 structured inputs 的压缩与扩展路径。

数据来源：

- `runs/01_main/beta5_interactions/base_feature_pool.txt`
- `runs/01_main/beta5_interactions/selected_interaction_definitions.csv`

形式：

- funnel / sankey-like bar。

### Figure 5: Interaction Operator Distribution

目的：

- 展示最终 120 个 interaction 中各操作类型的数量。

数据来源：

- `runs/01_main/beta5_interactions/selected_interaction_definitions.csv`

形式：

- bar chart：`add/sub/mul/div/max/min`。

### Figure 6: Component Correlation Matrix

目的：

- 证明最终模型由多个不完全同质的信号构成。

数据来源：

- validation predictions:
  - Pearson/Spearman Ridge components。
  - SHAP-stable XGB。
  - interaction Ridge。
  - supervised AE。
  - MLP seed2026。

形式：

- heatmap：component prediction correlation。

### Figure 7: Holdout Delta vs Private Delta

目的：

- 展示本地提升和 private 提升不总一致。

数据来源：

- 各阶段 `blend_candidate_metrics.csv`。
- Kaggle public/private result 手工表。

形式：

- scatter plot 或 paired bar。

### Figure 8: Successful vs Failed Signal Families

目的：

- 把实验结论浓缩成一张方法对比图。

形式：

- grouped bar 或 table-like plot。
- 分类：selected / useful but not selected / failed / public trap。

## 图表产物建议路径

建议后续统一输出到：

```text
reports/figures/01_main/
```

建议文件名：

```text
score_ladder.png
public_private_scatter.png
final_pipeline.png
feature_funnel.png
interaction_operator_distribution.png
component_correlation_heatmap.png
holdout_private_delta.png
signal_family_summary.png
```

## 报告写作顺序

建议顺序：

1. 先写第 1-3 节，定问题、数据、baseline。
2. 画 Figure 1、Figure 2，确认整体叙事。
3. 写第 4-6 节，解释特征筛选、interaction、AE/MLP。
4. 画 Figure 3-6。
5. 写第 7-9 节，定最终模型、失败实验和结论。
6. 最后补摘要和图表说明。

## 需要补充的可视化脚本

下一步建议新增：

```text
scripts/make_main_report_figures.py
```

职责：

- 读取 docs 中的阶段结果和 runs 中的候选指标。
- 生成 `reports/figures/01_main/` 下的所有主任务图。
- 输出一个 `figure_manifest.csv`，记录每张图的数据来源和用途。

当前状态：脚本已实现，已生成 8 张主任务图和 `figure_manifest.csv`。
