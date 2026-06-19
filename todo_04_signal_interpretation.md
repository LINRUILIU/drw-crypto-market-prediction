# TODO 04: Extension Task 3 - Predictive Signal Interpretation

## Goal

Translate the final prediction signal into intuitive financial and presentation-friendly evidence. This extension should show whether higher predicted values correspond to higher realized targets on average, without claiming to build a real tradable strategy.

## Scope

This is intentionally lightweight. It should support the report and PPT, not reopen the leaderboard sprint.

## Phase 1: Analysis Table

- [ ] Use validation predictions from the frozen selected main model if available.
- [ ] If the exact frozen validation prediction must be reconstructed, document the reconstruction formula and component paths.
- [ ] Save one table with sample index, prediction, target, model name, and chronological segment.
- [ ] Keep validation labels untouched by fitting or tuning in this analysis.

## Phase 2: Ranking Interpretation

- [ ] Sort validation samples by predicted value.
- [ ] Define bottom 10%, middle 80%, and top 10% groups.
- [ ] Compute average prediction and average true target per group.
- [ ] Compute top-minus-bottom target spread.
- [ ] Split predictions into 10 deciles and compute mean target per decile.
- [ ] Compute Spearman rank correlation as a ranking metric.

## Phase 3: Directional Interpretation

- [ ] Compute sign agreement between prediction and target.
- [ ] Compare all samples versus strong-signal samples.
- [ ] Define strong-signal samples by top/bottom prediction quantiles.
- [ ] Report directional accuracy carefully as signal interpretation, not as trading performance.

## Phase 4: Simplified Long-Short Explanation

- [ ] Define a theoretical long group as the top prediction quantile.
- [ ] Define a theoretical short group as the bottom prediction quantile.
- [ ] Compute no-cost top-bottom target spread.
- [ ] Clearly state that this is not a real backtest.
- [ ] Do not add transaction cost, slippage, latency, turnover, or risk control unless explicitly modeled.

## Required Outputs

- [ ] `runs/04_signal/signal_interpretation_table.csv`
- [ ] `runs/04_signal/metrics_signal_interpretation.csv`
- [ ] Top-middle-bottom target mean plot.
- [ ] Decile target mean plot.
- [ ] Directional accuracy table.
- [ ] Theoretical top-bottom spread table.
- [ ] PPT-friendly signal interpretation figure.

## Acceptance Criteria

- [ ] The analysis shows whether higher predictions correspond to higher realized target on average.
- [ ] Strong-signal and weak-signal groups are compared.
- [ ] The report avoids overclaiming profitability.
- [ ] The final PPT has at least one chart that non-finance readers can understand quickly.
- [ ] The section connects Pearson correlation to ranking and directional meaning without changing the official metric.
