# Main Task Retrospective and Sprint Decision

## Current Selected Submission

Selected main-task submission after Beta7.1:

```text
0.925 * beta6_2_current_best + 0.075 * wide_adamw_lr001_seed2026
```

Leaderboard result:

| Public | Private | Status |
| ---: | ---: | --- |
| 0.06547 | 0.11043 | Best observed post-competition score for the same frozen Beta7 final file |

Post-competition audit note, 2026-06-21: the selected file is `submissions/01_main/beta7_mlp_signal/submission_blend_current_w0p925_wide_adamw_lr001_seed2026_w0p075.csv`. This keeps the same Beta7 formula and does not define a new modeling branch. Because the score was obtained after the competition ended, it should be described as the best observed score, not as an official leaderboard rank. The earlier `0.06553 / 0.10550` result remains the development-time logged score for the same final line.

Beta7.1 found a narrow private plateau around MLP signal weight `0.075-0.0875`. The `0.0875` candidate tied private `0.10550` but had lower public, so the `0.075` Beta7 submission remains selected.

## Score Ladder

| Stage | Selected signal | Public | Private | Interpretation |
| --- | --- | ---: | ---: | --- |
| Beta2 | top50/top100 Ridge alpha blend | - | 0.09638 | Strong linear top-k baseline |
| Beta3 | Spearman top50 low-weight blend | 0.05456 | 0.10003 | Rank-based feature selection added transferable signal |
| Beta4 | SHAP-stable XGB blend | 0.05634 | 0.10043 | First non-linear/tree-based gain |
| Beta5-A | 120 symbolic interactions + Ridge | 0.06362 | 0.10302 | Largest structural jump |
| Beta6.2 | Supervised AE8 tiny blend | 0.06454 | 0.10303 | Marginal representation gain only at `0.025` |
| Beta7 | Supervised MLP low-weight blend | 0.06553 | 0.10550 | Useful non-linear correction, not a standalone replacement |

The main result did not come from one model swap. It came from a layered signal stack:

```text
Ridge top-k
+ Spearman top-k
+ SHAP-stable XGB
+ symbolic interaction Ridge
+ tiny supervised AE
+ low-weight supervised MLP
```

## Signal Ledger

| Signal family | Transfer result | Keep? | Notes |
| --- | --- | --- | --- |
| Pearson top-k Ridge | Strong | Yes | Foundation of the current pipeline. Alpha and width tuning are saturated. |
| Spearman top50 Ridge | Strong at low weight | Yes | Improved private to the `0.100` range, but direct weight tuning plateaued. |
| Residual top-k Ridge | Weak / unstable | No active sprint | Had local signal but did not transfer when directly blended. |
| Medoid Ridge | Failed as direct signal | Keep as feature-selection context only | Correlation clustering was useful conceptually, but medoid Ridge itself did not transfer. |
| SHAP-stable XGB | Small but real | Yes | Beta4 improved private; Beta4.1 refinement did not. |
| Interaction Ridge | Strong | Yes | Biggest post-Beta4 gain. This is the best place to search for one more structured feature improvement. |
| Interaction XGB | Weak | No active sprint | Nearly tied prior best, but did not beat interaction Ridge. |
| Wider interaction Ridge | Public/holdout trap | No | `int240` improved public but reduced private. |
| Unsupervised AE | Failed | No | AE8/16/32 and denoising did not transfer. |
| Supervised AE | Tiny positive | Frozen | Only worth keeping at `0.025`; not worth further AE-only tuning now. |
| Supervised MLP | Positive at low weight | Yes, limited | AdamW seed2026 at weight `0.075` helped; heavier or seed-mean variants were less reliable. |

## Closed Directions

These axes should stay closed unless a new hypothesis changes the setup:

- Ridge feature width, alpha, and weight-only sweeps.
- Spearman-only weight micro-tuning.
- Direct residual Ridge blending.
- SHAP-stable feature-rule and XGB early-stopping sweeps.
- Larger interaction counts and heavier interaction weights.
- Unsupervised or wider AE representations.
- MLP weight-only tuning around the existing seed2026 signal.

The repeated pattern is that holdout and public often over-reward larger or heavier signals. Private improvements came from small, complementary additions that were structurally different from the current best.

## First-Place Gap

The project has reproduced the method-level outline from the first-place writeup:

- correlation-aware feature reduction;
- XGB/SHAP stable feature selection;
- symbolic interaction features;
- AE-style representation features;
- supervised MLP signals.

The gap is not that these ideas were ignored. The gap is implementation depth:

- First-place feature engineering appears to be more selective and more iterative, especially around symbolic combinations and feature recycling.
- Their MLP likely benefited from a cleaner final feature representation; in this project, MLP is useful only as a low-weight correction.
- AE features did not become a strong standalone representation here, which means the representation stage is not matching the high-ranking pipeline.
- Local validation cannot reliably rank late-stage candidates. Several public/holdout improvements reduced private score.

This makes a full first-place reproduction risky as a short sprint. It is a larger rebuild of the feature-engineering pipeline, not a small model change.

## Sprint Decision

Do not immediately restart the whole project to reproduce first place. Run at most two bounded sprints. If neither beats private `0.10550`, freeze the main task and move to report consolidation.

### Sprint-A: Fold-Stable Interaction Selection

Priority: highest.

Rationale: Beta5-A interaction Ridge was the largest structural gain. Beta5-B showed that simply adding more interactions is harmful, so the next test should improve interaction selection quality rather than interaction quantity.

Proposed design:

- Reuse the existing 40-feature core pool and pairwise symbolic operators.
- Score interactions inside purged contiguous folds instead of only one row-order train split.
- Track both label correlation and current-best residual correlation per fold.
- Select features by fold appearance and average rank, with pure-stable and stable-plus-fill variants recorded separately.
- Keep the final selected set near `120` features.
- Train Ridge only first; do not add XGB unless Ridge shows a private-relevant signal.
- Blend with the current best at conservative weights such as `0.05`, `0.10`, and `0.15`.
- Submit at most two candidates.

Success criterion:

```text
Beat private 0.10550 without requiring a larger interaction weight than Beta5-A.
```

Stop criterion:

```text
If local/public improves but private drops, close interaction selection and do not widen the feature set again.
```

### Sprint-B: AdamW MLP Seed Stability

Priority: second.

Rationale: Beta7 found a useful supervised MLP signal, but the winning candidate was a single AdamW seed. This may be either a real complementary signal or seed luck. The sprint should test stability, not model size.

Proposed design:

- Fix the successful Beta7 architecture and optimizer: `wide_adamw_lr001`.
- Train a small seed set around the successful recipe, for example `1026`, `2026`, `2526`, `3026`, `4026`.
- Do not change input features, hidden sizes, loss, or blend weights at the same time.
- Evaluate single-seed, filtered seed-mean, and low-correlation seed ensemble signals.
- Blend only at `0.025`, `0.05`, `0.075`, and possibly `0.0875`.
- Submit at most two candidates.

Success criterion:

```text
Find a seed or filtered ensemble that beats private 0.10550 while staying in the same low-weight region.
```

Stop criterion:

```text
If only one seed works and seed means regress, keep Beta7 selected and close MLP tuning.
```

## Sprint Results

Sprint-A and Sprint-B were both executed after the retrospective.

| Sprint | Candidate | Public | Private | Result |
| --- | --- | ---: | ---: | --- |
| Sprint-A | `top250_min3_fill120` interaction Ridge, signal weight `0.05` | 0.06637 | 0.10514 | Below current best |
| Sprint-A | `top250_min4_fill120` interaction Ridge, signal weight `0.15` | 0.06766 | 0.10437 | Below current best |
| Sprint-B | AdamW new-seed mean, signal weight `0.025` | 0.06573 | 0.10376 | Below current best |
| Sprint-B | AdamW seed4026, signal weight `0.075` | 0.06944 | 0.10321 | Public improved, private dropped |
| Sprint-B | AdamW seed2526, signal weight `0.0875` | 0.06397 | 0.10205 | Below current best |

Final sprint interpretation:

- Sprint-A did not improve the selected interaction branch. Fold-stable interaction selection produced sane low-correlation probes, but private score stayed below `0.10550`.
- Sprint-B confirmed that the MLP branch is seed-unstable. The previous seed2026 signal remains unusually good; new seeds and seed means did not reproduce its private gain.
- The selected main-task submission remains `0.925 * beta6_2_current_best + 0.075 * wide_adamw_lr001_seed2026`.
- The main task should now be frozen for report consolidation unless the project explicitly shifts to a larger first-place-reproduction effort.

## Report Positioning

The final report should not present this as a simple model leaderboard chase. The defensible modeling story is:

- high-dimensional anonymous features require feature selection and redundancy control;
- linear models are unexpectedly strong when feature selection is careful;
- non-linear models help only after structured feature engineering;
- public/private divergence makes conservative low-weight ensembling more reliable than local-best selection;
- the strongest final model is an ensemble of several complementary signal families, not a single complex model.

This is a coherent main-task narrative even if no further sprint improves the score.
