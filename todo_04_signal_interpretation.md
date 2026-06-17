# TODO 04: Extension Task 3 - Predictive Signal Interpretation

## Goal

Evaluate whether model predictions have directional and ranking meaning, without claiming to build a real trading system.

## Checkpoints

- [ ] Prepare validation predictions
  - [ ] Use predictions from the selected main model and ensemble.
  - [ ] Keep validation labels untouched by model fitting.
  - [ ] Save prediction, target, time index, and model name in one analysis table.

- [ ] Top-middle-bottom grouping
  - [ ] Sort validation samples by predicted value.
  - [ ] Define bottom 10%, middle 80%, and top 10%.
  - [ ] Compute average prediction and average true target per group.
  - [ ] Compute top-bottom target spread.
  - [ ] Plot group-level true target mean.

- [ ] Decile ranking analysis
  - [ ] Split predictions into 10 deciles.
  - [ ] Compute mean true target for each decile.
  - [ ] Compute monotonicity or rank trend.
  - [ ] Plot decile target curve.

- [ ] Directional accuracy
  - [ ] Compute sign agreement between prediction and target.
  - [ ] Compute directional accuracy for all samples.
  - [ ] Compute directional accuracy for strong-signal samples only.
  - [ ] Compare strong-signal and weak-signal groups.

- [ ] Rank correlation and signal strength
  - [ ] Compute Spearman rank correlation if useful.
  - [ ] Compute Rank IC by time segment if time grouping is available.
  - [ ] Compare Pearson and rank-based metrics.

- [ ] Simplified theoretical long-short signal
  - [ ] Define long group as top prediction quantile.
  - [ ] Define short group as bottom prediction quantile.
  - [ ] Compute no-cost theoretical spread.
  - [ ] Report clearly that this is not a real trading backtest.
  - [ ] Do not include transaction cost, slippage, latency, or risk control unless explicitly modeled.

## Required Outputs

- [ ] `metrics_signal_interpretation.csv`
- [ ] Top-middle-bottom target mean plot
- [ ] Decile target mean plot
- [ ] Directional accuracy table
- [ ] Theoretical top-bottom spread table
- [ ] PPT-friendly signal interpretation figure

## Acceptance Criteria

- [ ] The analysis shows whether higher predictions correspond to higher realized target on average.
- [ ] The report avoids overclaiming profitability.
- [ ] Strong-signal and weak-signal groups are compared.
- [ ] The final PPT has at least one intuitive chart that non-finance readers can understand.

