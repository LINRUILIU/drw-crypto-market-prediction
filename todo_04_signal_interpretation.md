# TODO 04: Extension Task 3 - Predictive Signal Interpretation

## Goal

Translate the final prediction signal into intuitive financial and presentation-friendly evidence. This extension should show whether higher predicted values correspond to higher realized targets on average, without claiming to build a real tradable strategy.

## Scope

This is intentionally lightweight. It should support the report and PPT, not reopen the leaderboard sprint.

## Phase 1: Analysis Table

- [x] Use validation predictions from the frozen selected main model if available.
- [x] If the exact frozen validation prediction must be reconstructed, document the reconstruction formula and component paths.
- [x] Save one table with sample index, prediction, target, model name, and chronological segment.
- [x] Keep validation labels untouched by fitting or tuning in this analysis.

## Phase 2: Ranking Interpretation

- [x] Sort validation samples by predicted value.
- [x] Define bottom 10%, middle 80%, and top 10% groups.
- [x] Compute average prediction and average true target per group.
- [x] Compute top-minus-bottom target spread.
- [x] Split predictions into 10 deciles and compute mean target per decile.
- [x] Compute Spearman rank correlation as a ranking metric.

## Phase 3: Directional Interpretation

- [x] Compute sign agreement between prediction and target.
- [x] Compare all samples versus strong-signal samples.
- [x] Define strong-signal samples by top/bottom prediction quantiles.
- [x] Report directional accuracy carefully as signal interpretation, not as trading performance.

## Phase 4: Simplified Long-Short Explanation

- [x] Define a theoretical long group as the top prediction quantile.
- [x] Define a theoretical short group as the bottom prediction quantile.
- [x] Compute no-cost top-bottom target spread.
- [x] Clearly state that this is not a real backtest.
- [x] Do not add transaction cost, slippage, latency, turnover, or risk control unless explicitly modeled.

## Required Outputs

- [x] `runs/04_signal/signal_interpretation_table.csv`
- [x] `runs/04_signal/metrics_signal_interpretation.csv`
- [x] `runs/04_signal/top_middle_bottom_summary.csv`
- [x] `runs/04_signal/decile_target_summary.csv`
- [x] `runs/04_signal/directional_accuracy_summary.csv`
- [x] `runs/04_signal/theoretical_top_bottom_spread.csv`
- [x] `runs/04_signal/reconstruction_check.csv`
- [x] `reports/figures/04_signal/top_middle_bottom_target_mean.png`
- [x] `reports/figures/04_signal/decile_target_mean.png`
- [x] `reports/figures/04_signal/directional_accuracy_summary.png`
- [x] `reports/figures/04_signal/signal_interpretation_summary.png`

## Acceptance Criteria

- [x] The analysis shows whether higher predictions correspond to higher realized target on average.
- [x] Strong-signal and weak-signal groups are compared.
- [x] The report avoids overclaiming profitability.
- [x] The final PPT has at least one chart that non-finance readers can understand quickly.
- [x] The section connects Pearson correlation to ranking and directional meaning without changing the official metric.
