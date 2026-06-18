from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class InteractionDefinition:
    name: str
    left: str
    right: str
    operation: str
    left_index: int
    right_index: int
    clip_low: float | None = None
    clip_high: float | None = None


def build_ordered_feature_pool(feature_sources: Iterable[Iterable[str]], max_features: int) -> list[str]:
    """Merge feature lists in priority order while preserving first occurrence."""
    selected: list[str] = []
    seen: set[str] = set()
    for source in feature_sources:
        for feature in source:
            if feature in seen:
                continue
            selected.append(str(feature))
            seen.add(str(feature))
            if len(selected) >= int(max_features):
                return selected
    return selected


def generate_pairwise_interactions(
    base_features: list[str],
    operations: list[str],
) -> list[InteractionDefinition]:
    """Create all pairwise interaction definitions for the configured operations."""
    definitions: list[InteractionDefinition] = []
    for left_idx, left in enumerate(base_features):
        for right_idx in range(left_idx + 1, len(base_features)):
            right = base_features[right_idx]
            for operation in operations:
                name = f"int__{operation}__{left}__{right}"
                definitions.append(
                    InteractionDefinition(
                        name=name,
                        left=left,
                        right=right,
                        operation=operation,
                        left_index=left_idx,
                        right_index=right_idx,
                    )
                )
    return definitions


def _compute_operation(left: np.ndarray, right: np.ndarray, operation: str, eps: float) -> np.ndarray:
    if operation == "add":
        values = left + right
    elif operation == "sub":
        values = left - right
    elif operation == "mul":
        values = left * right
    elif operation == "div_l_abs_r":
        values = left / (np.abs(right) + float(eps))
    elif operation == "div_r_abs_l":
        values = right / (np.abs(left) + float(eps))
    elif operation == "max":
        values = np.maximum(left, right)
    elif operation == "min":
        values = np.minimum(left, right)
    else:
        raise ValueError(f"Unsupported interaction operation: {operation}")
    return values.astype(np.float32, copy=False)


def transform_interaction_block(
    base_values: np.ndarray,
    definitions: list[InteractionDefinition],
    eps: float,
    fit_clips: bool = False,
    clip_quantiles: tuple[float, float] = (0.001, 0.999),
) -> tuple[np.ndarray, list[InteractionDefinition]]:
    """Transform a block of interaction features.

    When ``fit_clips`` is true, clipping bounds are fitted from ``base_values`` and
    returned as updated definitions. Otherwise existing bounds are applied.
    """
    if not definitions:
        return np.empty((len(base_values), 0), dtype=np.float32), []

    output = np.empty((len(base_values), len(definitions)), dtype=np.float32)
    fitted: list[InteractionDefinition] = []
    low_q, high_q = clip_quantiles
    for col_idx, definition in enumerate(definitions):
        values = _compute_operation(
            base_values[:, definition.left_index],
            base_values[:, definition.right_index],
            definition.operation,
            eps,
        )
        if fit_clips:
            low = float(np.nanquantile(values, low_q))
            high = float(np.nanquantile(values, high_q))
        else:
            low = definition.clip_low
            high = definition.clip_high
        if low is not None and high is not None:
            values = np.clip(values, low, high, out=values)
        output[:, col_idx] = values
        fitted.append(
            InteractionDefinition(
                name=definition.name,
                left=definition.left,
                right=definition.right,
                operation=definition.operation,
                left_index=definition.left_index,
                right_index=definition.right_index,
                clip_low=low,
                clip_high=high,
            )
        )
    return output, fitted


def definitions_to_frame(definitions: list[InteractionDefinition]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": definition.name,
                "left": definition.left,
                "right": definition.right,
                "operation": definition.operation,
                "left_index": definition.left_index,
                "right_index": definition.right_index,
                "clip_low": definition.clip_low,
                "clip_high": definition.clip_high,
            }
            for definition in definitions
        ]
    )


def definitions_from_frame(frame: pd.DataFrame) -> list[InteractionDefinition]:
    return [
        InteractionDefinition(
            name=str(row["feature"]),
            left=str(row["left"]),
            right=str(row["right"]),
            operation=str(row["operation"]),
            left_index=int(row["left_index"]),
            right_index=int(row["right_index"]),
            clip_low=float(row["clip_low"]) if pd.notna(row["clip_low"]) else None,
            clip_high=float(row["clip_high"]) if pd.notna(row["clip_high"]) else None,
        )
        for _, row in frame.iterrows()
    ]


def _pearson_against_centered(values: np.ndarray, centered_target: np.ndarray, target_std: float) -> np.ndarray:
    mean = values.mean(axis=0, dtype=np.float64)
    mean_sq = np.square(values, dtype=np.float64).mean(axis=0)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    cov = (values.astype(np.float64).T @ centered_target) / len(centered_target)
    denom = std * float(target_std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov, dtype=np.float64), where=denom > 0)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def _spearman_scores(values: np.ndarray, ranked_target_centered: np.ndarray, ranked_target_std: float) -> np.ndarray:
    scores = np.zeros(values.shape[1], dtype=np.float64)
    n_rows = values.shape[0]
    for col_idx in range(values.shape[1]):
        ranked = rankdata(values[:, col_idx], method="average").astype(np.float64)
        ranked -= ranked.mean()
        std = ranked.std()
        if std <= 0 or ranked_target_std <= 0:
            scores[col_idx] = 0.0
        else:
            scores[col_idx] = float((ranked @ ranked_target_centered) / (n_rows * std * ranked_target_std))
    scores[~np.isfinite(scores)] = 0.0
    return scores


def score_interactions(
    base_values: np.ndarray,
    definitions: list[InteractionDefinition],
    label: np.ndarray,
    residual: np.ndarray,
    eps: float,
    clip_quantiles: tuple[float, float],
    block_size: int,
    spearman_sample_rows: int,
) -> tuple[pd.DataFrame, list[InteractionDefinition]]:
    """Fit clips and score interactions with train-only label/residual criteria."""
    label = label.astype(np.float64, copy=False)
    residual = residual.astype(np.float64, copy=False)
    label_centered = label - label.mean()
    residual_centered = residual - residual.mean()
    label_std = float(label.std())
    residual_std = float(residual.std())

    sample_size = min(int(spearman_sample_rows), len(label))
    if sample_size < len(label):
        sample_idx = np.linspace(0, len(label) - 1, sample_size, dtype=np.int64)
    else:
        sample_idx = np.arange(len(label), dtype=np.int64)
    sample_label = label[sample_idx]
    ranked_label = rankdata(sample_label, method="average").astype(np.float64)
    ranked_label -= ranked_label.mean()
    ranked_label_std = float(ranked_label.std())

    rows: list[dict[str, float | str | int]] = []
    fitted_definitions: list[InteractionDefinition] = []
    for start in range(0, len(definitions), int(block_size)):
        block_defs = definitions[start : start + int(block_size)]
        values, fitted_block = transform_interaction_block(
            base_values,
            block_defs,
            eps=eps,
            fit_clips=True,
            clip_quantiles=clip_quantiles,
        )
        pearson_label = _pearson_against_centered(values, label_centered, label_std)
        pearson_residual = _pearson_against_centered(values, residual_centered, residual_std)
        sample_values = values[sample_idx]
        spearman_label = _spearman_scores(sample_values, ranked_label, ranked_label_std)

        for idx, definition in enumerate(fitted_block):
            abs_pearson_label = abs(float(pearson_label[idx]))
            abs_pearson_residual = abs(float(pearson_residual[idx]))
            abs_spearman_label = abs(float(spearman_label[idx]))
            metric_values = {
                "abs_pearson_label": abs_pearson_label,
                "abs_pearson_residual": abs_pearson_residual,
                "abs_spearman_label": abs_spearman_label,
            }
            best_metric = max(metric_values, key=metric_values.get)
            rows.append(
                {
                    "feature": definition.name,
                    "left": definition.left,
                    "right": definition.right,
                    "operation": definition.operation,
                    "left_index": definition.left_index,
                    "right_index": definition.right_index,
                    "clip_low": definition.clip_low,
                    "clip_high": definition.clip_high,
                    "pearson_label": float(pearson_label[idx]),
                    "abs_pearson_label": abs_pearson_label,
                    "pearson_residual": float(pearson_residual[idx]),
                    "abs_pearson_residual": abs_pearson_residual,
                    "spearman_label": float(spearman_label[idx]),
                    "abs_spearman_label": abs_spearman_label,
                    "best_metric": best_metric,
                    "max_abs_score": float(metric_values[best_metric]),
                }
            )
        fitted_definitions.extend(fitted_block)
    scores = pd.DataFrame(rows)
    for metric in ["abs_pearson_label", "abs_pearson_residual", "abs_spearman_label"]:
        scores[f"rank_{metric}"] = scores[metric].rank(method="min", ascending=False)
    scores["best_rank"] = scores[
        ["rank_abs_pearson_label", "rank_abs_pearson_residual", "rank_abs_spearman_label"]
    ].min(axis=1)
    return scores.sort_values(["best_rank", "max_abs_score", "feature"], ascending=[True, False, True]), fitted_definitions


def candidate_pool_from_scores(scores: pd.DataFrame, top_k_each_metric: int) -> pd.DataFrame:
    selected_features: set[str] = set()
    for metric in ["abs_pearson_label", "abs_pearson_residual", "abs_spearman_label"]:
        top = scores.sort_values([metric, "feature"], ascending=[False, True]).head(int(top_k_each_metric))
        selected_features.update(top["feature"].astype(str).tolist())
    return (
        scores[scores["feature"].isin(selected_features)]
        .sort_values(["best_rank", "max_abs_score", "feature"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def prune_correlated_interactions(
    base_values: np.ndarray,
    candidate_definitions: list[InteractionDefinition],
    candidate_scores: pd.DataFrame,
    eps: float,
    corr_threshold: float,
    max_features: int,
) -> tuple[list[InteractionDefinition], pd.DataFrame]:
    """Greedily remove highly correlated candidate interactions."""
    selected_defs: list[InteractionDefinition] = []
    selected_norms: list[np.ndarray] = []
    log_rows: list[dict[str, float | str | int | bool]] = []
    score_lookup = candidate_scores.set_index("feature")

    for definition in candidate_definitions:
        values, _ = transform_interaction_block(base_values, [definition], eps=eps, fit_clips=False)
        vector = values[:, 0].astype(np.float64, copy=False)
        vector -= vector.mean()
        std = float(vector.std())
        if std <= 0:
            log_rows.append(
                {
                    "feature": definition.name,
                    "selected": False,
                    "reason": "constant",
                    "max_abs_corr_selected": 0.0,
                    "selected_count_before": len(selected_defs),
                }
            )
            continue
        norm = (vector / std).astype(np.float32)
        max_corr = 0.0
        if selected_norms:
            selected_matrix = np.column_stack(selected_norms)
            corr = np.abs((norm.astype(np.float64) @ selected_matrix.astype(np.float64)) / len(norm))
            max_corr = float(np.max(corr))
        keep = max_corr <= float(corr_threshold)
        reason = "selected" if keep else "high_corr"
        if keep:
            selected_defs.append(definition)
            selected_norms.append(norm)
        score_row = score_lookup.loc[definition.name]
        log_rows.append(
            {
                "feature": definition.name,
                "selected": keep,
                "reason": reason,
                "max_abs_corr_selected": max_corr,
                "selected_count_before": len(selected_defs) - int(keep),
                "best_metric": str(score_row["best_metric"]),
                "max_abs_score": float(score_row["max_abs_score"]),
                "best_rank": float(score_row["best_rank"]),
            }
        )
        if len(selected_defs) >= int(max_features):
            break
    return selected_defs, pd.DataFrame(log_rows)


def transform_interactions(
    base_values: np.ndarray,
    definitions: list[InteractionDefinition],
    eps: float,
    block_size: int = 128,
) -> np.ndarray:
    if not definitions:
        return np.empty((len(base_values), 0), dtype=np.float32)
    output = np.empty((len(base_values), len(definitions)), dtype=np.float32)
    for start in range(0, len(definitions), int(block_size)):
        block_defs = definitions[start : start + int(block_size)]
        values, _ = transform_interaction_block(base_values, block_defs, eps=eps, fit_clips=False)
        output[:, start : start + len(block_defs)] = values
    return output
