from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drw_crypto.feature_selection import pearson_feature_ranking  # noqa: E402
from drw_crypto.io import TRAIN_CANDIDATES, ensure_dir, find_first_existing, read_table, write_json  # noqa: E402
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402
from drw_crypto.models import NumpyRidgeRegressor  # noqa: E402
from drw_crypto.preprocessing import TabularPreprocessor, infer_feature_columns, infer_id_column, infer_target_column  # noqa: E402


ROLLING_DIR = ROOT / "runs" / "03_temporal" / "rolling_validation"
ARTIFACT_DIR = ROOT / "runs" / "03_temporal" / "report_artifacts"
FIGURE_DIR = ROOT / "reports" / "figures" / "03_temporal"
RAW_DIR = ROOT / "data" / "raw"
LEADERBOARD_PUBLIC_PATH = ROOT / "docs" / "leaderboard_public_desc.csv"
LEADERBOARD_PRIVATE_PATH = ROOT / "docs" / "leaderboard_private_desc.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal stability report artifacts.")
    parser.add_argument("--skip-embargo", action="store_true", help="Skip the fixed-alpha Ridge embargo check.")
    parser.add_argument("--ridge-alpha", type=float, default=1000.0, help="Fixed Ridge alpha for embargo check.")
    parser.add_argument("--embargo-fraction", type=float, default=0.01, help="Row fraction removed before validation.")
    return parser.parse_args()


def savefig(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def format_numeric(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(float).round(digits)
    return out


def build_rolling_tables() -> dict[str, Any]:
    stability = pd.read_csv(ROLLING_DIR / "scheme_stability_summary.csv")
    keep_cols = ["scheme", "fold_count", "pearson_mean", "pearson_std", "pearson_min", "pearson_max", "rmse_mean"]
    rolling = stability.loc[:, keep_cols].sort_values(["pearson_mean", "pearson_std"], ascending=[False, True])
    rolling = format_numeric(rolling, keep_cols[2:], digits=6)
    rolling.to_csv(ARTIFACT_DIR / "rolling_stability_report_table.csv", index=False)

    target = pd.read_csv(ROLLING_DIR / "target_distribution_by_fold.csv")
    target_cols = ["fold", "target_mean", "target_std", "target_q05", "target_q50", "target_q95"]
    target_report = target.loc[:, target_cols]
    target_report = format_numeric(target_report, target_cols[1:], digits=6)
    target_report.to_csv(ARTIFACT_DIR / "target_drift_report_table.csv", index=False)

    return {
        "rolling_best_mean_scheme": str(rolling.iloc[0]["scheme"]),
        "rolling_most_stable_scheme": str(stability.sort_values(["pearson_std", "pearson_mean"], ascending=[True, False]).iloc[0]["scheme"]),
        "target_high_mean_fold": str(target.sort_values("target_mean", ascending=False).iloc[0]["fold"]),
        "target_low_mean_fold": str(target.sort_values("target_mean", ascending=True).iloc[0]["fold"]),
    }


def build_feature_stability_tables() -> dict[str, Any]:
    overlap = pd.read_csv(ROLLING_DIR / "feature_overlap.csv")
    overlap_summary = (
        overlap.groupby("scheme", as_index=False)
        .agg(
            pair_count=("jaccard", "size"),
            jaccard_mean=("jaccard", "mean"),
            jaccard_min=("jaccard", "min"),
            overlap_ratio_mean=("overlap_ratio_of_k", "mean"),
            overlap_ratio_min=("overlap_ratio_of_k", "min"),
        )
        .sort_values(["scheme"])
    )
    overlap_summary = format_numeric(
        overlap_summary,
        ["jaccard_mean", "jaccard_min", "overlap_ratio_mean", "overlap_ratio_min"],
        digits=6,
    )
    overlap_summary.to_csv(ARTIFACT_DIR / "feature_overlap_report_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(overlap_summary))
    ax.bar(x - 0.18, overlap_summary["jaccard_mean"], width=0.36, label="Mean Jaccard", color="#2563eb")
    ax.bar(x + 0.18, overlap_summary["jaccard_min"], width=0.36, label="Min Jaccard", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(overlap_summary["scheme"], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Feature-set overlap")
    ax.set_title("Rolling Feature Selection Overlap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig("feature_overlap_summary.png")

    frequency = pd.read_csv(ROLLING_DIR / "feature_selection_frequency.csv")
    top100_stable = frequency[(frequency["scheme"] == "top100") & (frequency["fold_count"] == 4)].copy()
    rank_frames: list[pd.DataFrame] = []
    for path in sorted((ROLLING_DIR / "feature_rankings").glob("pearson_feature_ranking_*.csv")):
        fold = path.stem.replace("pearson_feature_ranking_", "")
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        rank_frames.append(frame.loc[:, ["feature", "rank"]].rename(columns={"rank": f"rank_{fold}"}))

    if rank_frames and not top100_stable.empty:
        ranks = rank_frames[0]
        for frame in rank_frames[1:]:
            ranks = ranks.merge(frame, on="feature", how="outer")
        rank_cols = [col for col in ranks.columns if col.startswith("rank_")]
        ranks["mean_rank"] = ranks[rank_cols].mean(axis=1)
        top100_stable = top100_stable.merge(ranks.loc[:, ["feature", "mean_rank"]], on="feature", how="left")
        top100_stable = top100_stable.sort_values(["mean_rank", "feature"], ascending=[True, True])
    else:
        top100_stable["mean_rank"] = np.nan

    top100_stable.head(30).to_csv(ARTIFACT_DIR / "stable_feature_frequency_table.csv", index=False)
    return {
        "top100_stable_count": int(len(top100_stable)),
        "top100_stable_features_for_drift": top100_stable["feature"].head(10).astype(str).tolist(),
        "min_top100_jaccard": float(overlap_summary.loc[overlap_summary["scheme"] == "top100", "jaccard_min"].iloc[0]),
    }


def find_train_path() -> Path:
    path = find_first_existing(RAW_DIR, None, TRAIN_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Missing train file under data/raw.")
    return path


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.name.lower().endswith(".csv") or path.name.lower().endswith(".csv.zip"):
        return pd.read_csv(path, usecols=columns)
    frame = read_table(path)
    return frame.loc[:, columns]


def slice_by_fraction(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    n_rows = len(df)
    return df.iloc[int(n_rows * start) : int(n_rows * end)]


def build_feature_drift(stable_features: list[str]) -> dict[str, Any]:
    if not stable_features:
        raise ValueError("No stable features available for feature drift analysis.")

    train_path = find_train_path()
    features = read_columns(train_path, stable_features)
    reference = slice_by_fraction(features, 0.0, 0.5).replace([np.inf, -np.inf], np.nan)
    reference_mean = reference.mean(axis=0)
    reference_std = reference.std(axis=0, ddof=0).replace(0.0, np.nan)

    segments = [
        ("fold_50_60", 0.5, 0.6),
        ("fold_60_70", 0.6, 0.7),
        ("fold_70_80", 0.7, 0.8),
        ("fold_80_90", 0.8, 0.9),
    ]
    rows: list[dict[str, Any]] = []
    matrix_rows: list[list[float]] = []
    for fold, start, end in segments:
        segment = slice_by_fraction(features, start, end).replace([np.inf, -np.inf], np.nan)
        segment_mean = segment.mean(axis=0)
        mean_z = ((segment_mean - reference_mean) / reference_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        matrix_rows.append([float(mean_z[feature]) for feature in stable_features])
        for feature in stable_features:
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "reference_mean": float(reference_mean[feature]),
                    "reference_std": float(reference_std[feature]) if np.isfinite(reference_std[feature]) else 0.0,
                    "segment_mean": float(segment_mean[feature]),
                    "mean_z": float(mean_z[feature]),
                }
            )

    drift = pd.DataFrame(rows)
    drift.to_csv(ARTIFACT_DIR / "feature_drift_summary.csv", index=False)

    matrix = np.asarray(matrix_rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vmax = max(0.5, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(stable_features)))
    ax.set_xticklabels(stable_features, rotation=35, ha="right")
    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels([item[0] for item in segments])
    ax.set_title("Stable Feature Mean Drift by Time Segment")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Mean z-shift vs 0-50% reference")
    savefig("stable_feature_drift_heatmap.png")

    return {
        "feature_drift_max_abs_mean_z": float(np.nanmax(np.abs(matrix))),
        "feature_drift_features": stable_features,
    }


def load_train_for_embargo() -> tuple[pd.DataFrame, str, list[str]]:
    path = find_train_path()
    train_df = read_table(path)
    target_col = infer_target_column(train_df, None, "label")
    id_col = infer_id_column(train_df, None, None, target_col, None)
    feature_cols = infer_feature_columns(train_df, target_col, id_col, [], numeric_only=True)
    return train_df, target_col, feature_cols


def run_embargo_check(alpha: float, embargo_fraction: float) -> dict[str, Any]:
    train_df, target_col, feature_cols = load_train_for_embargo()
    n_rows = len(train_df)
    valid_start = int(n_rows * 0.8)
    valid_part = train_df.iloc[valid_start:].copy()
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for gap_fraction in [0.0, float(embargo_fraction)]:
        gap_rows = int(n_rows * gap_fraction)
        train_end = max(1, valid_start - gap_rows)
        train_part = train_df.iloc[:train_end].copy()
        ranking = pearson_feature_ranking(train_part, feature_cols, target_col)
        y_train = train_part[target_col].to_numpy(dtype=np.float64)
        for top_k in [50, 100, 300]:
            selected = ranking["feature"].head(top_k).astype(str).tolist()
            preprocessor = TabularPreprocessor(selected).fit(train_part, missing_fill="median", standardize=True)
            x_train = preprocessor.transform(train_part)
            x_valid = preprocessor.transform(valid_part)
            model = NumpyRidgeRegressor(alpha=float(alpha)).fit(x_train, y_train)
            pred = model.predict(x_valid)
            rows.append(
                {
                    "scheme": f"top{top_k}",
                    "top_k": top_k,
                    "gap_fraction": gap_fraction,
                    "gap_rows": gap_rows,
                    "train_rows": len(train_part),
                    "valid_rows": len(valid_part),
                    "alpha": float(alpha),
                    "pearson": pearson_corr(y_valid, pred),
                    "rmse": rmse(y_valid, pred),
                }
            )

    result = pd.DataFrame(rows)
    baseline = result[result["gap_fraction"] == 0.0].set_index("scheme")
    result["pearson_delta_vs_gap0"] = result.apply(
        lambda row: float(row["pearson"] - baseline.loc[row["scheme"], "pearson"]),
        axis=1,
    )
    result = format_numeric(result, ["gap_fraction", "alpha", "pearson", "rmse", "pearson_delta_vs_gap0"], digits=6)
    result.to_csv(ARTIFACT_DIR / "embargo_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for scheme, group in result.groupby("scheme"):
        group = group.sort_values("gap_fraction")
        ax.plot(group["gap_fraction"], group["pearson"], marker="o", label=scheme)
    ax.set_title("Fixed-Alpha Ridge Embargo Check")
    ax.set_xlabel("Embargo fraction before 80/20 validation")
    ax.set_ylabel("Holdout Pearson")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig("embargo_comparison.png")

    gap_rows = result[result["gap_fraction"] > 0]
    return {
        "embargo_alpha": float(alpha),
        "embargo_fraction": float(embargo_fraction),
        "max_embargo_drop": float(gap_rows["pearson_delta_vs_gap0"].min()) if not gap_rows.empty else 0.0,
        "mean_embargo_delta": float(gap_rows["pearson_delta_vs_gap0"].mean()) if not gap_rows.empty else 0.0,
    }


def submission_history() -> pd.DataFrame:
    rows = [
        ("Pearson top200", "Baseline", 0.02826, 0.06610),
        ("Stability blend", "Baseline", 0.06237, 0.05589),
        ("Submission blend private-safe", "Baseline", 0.03365, 0.06628),
        ("Beta1 top100 Ridge", "Ridge", 0.03716, 0.08901),
        ("Beta2 Ridge", "Ridge", 0.05160, 0.09638),
        ("Beta3 Spearman", "Ridge/Spearman", 0.05456, 0.10003),
        ("Beta4 SHAP XGB", "SHAP/XGB", 0.05634, 0.10043),
        ("Beta4 MLP", "MLP", 0.05959, 0.09729),
        ("Beta5 Interaction", "Interaction", 0.06362, 0.10302),
        ("Beta5B int240", "Interaction", 0.06797, 0.10256),
        ("Beta6 AE w0.15", "AE", 0.06488, 0.10169),
        ("Beta6.2 Sup-AE", "AE", 0.06454, 0.10303),
        ("Beta7 MLP mean", "MLP", 0.06588, 0.10411),
        ("Beta7 MLP logged", "MLP", 0.06553, 0.10550),
        ("Beta7 MLP heavy", "MLP", 0.06930, 0.10304),
        ("Sprint-A w0.05", "Sprint", 0.06637, 0.10514),
        ("Sprint-A w0.15", "Sprint", 0.06766, 0.10437),
        ("Sprint-B mean", "Sprint", 0.06573, 0.10376),
        ("Sprint-B seed4026", "Sprint", 0.06944, 0.10321),
        ("Sprint-B seed2526", "Sprint", 0.06397, 0.10205),
        ("Beta7 final post-freeze", "Post-freeze", 0.06547, 0.11043),
    ]
    frame = pd.DataFrame(rows, columns=["candidate", "family", "public", "private"])
    frame["private_minus_public"] = frame["private"] - frame["public"]
    frame["public_rank"] = frame["public"].rank(method="min", ascending=False).astype(int)
    frame["private_rank"] = frame["private"].rank(method="min", ascending=False).astype(int)
    return frame


def build_public_private_tables() -> dict[str, Any]:
    history = submission_history()
    history.to_csv(ARTIFACT_DIR / "public_private_submission_table.csv", index=False)

    selected = history[history["candidate"] == "Beta7 final post-freeze"].iloc[0]
    logged_beta7 = history[history["candidate"] == "Beta7 MLP logged"].iloc[0]
    public_traps = history[(history["public"] > selected["public"]) & (history["private"] < selected["private"])]
    low_public_high_private = history[(history["public"] < selected["public"]) & (history["private"] >= 0.103)]
    corr = float(history[["public", "private"]].corr().iloc[0, 1])
    summary = pd.DataFrame(
        [
            {
                "candidate_count": len(history),
                "public_private_corr": corr,
                "selected_candidate": selected["candidate"],
                "selected_public": selected["public"],
                "selected_private": selected["private"],
                "public_best_candidate": history.sort_values("public", ascending=False).iloc[0]["candidate"],
                "public_best_private": history.sort_values("public", ascending=False).iloc[0]["private"],
                "private_best_candidate": history.sort_values("private", ascending=False).iloc[0]["candidate"],
                "private_best_public": history.sort_values("private", ascending=False).iloc[0]["public"],
                "logged_beta7_public": logged_beta7["public"],
                "logged_beta7_private": logged_beta7["private"],
                "post_freeze_public": selected["public"],
                "post_freeze_private": selected["private"],
                "public_trap_count_vs_selected": len(public_traps),
                "low_public_high_private_count": len(low_public_high_private),
            }
        ]
    )
    summary.to_csv(ARTIFACT_DIR / "public_private_gap_summary.csv", index=False)

    plot_rows = history.sort_values("private_minus_public", ascending=False).head(10).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563eb" if name == "Beta7 final post-freeze" else "#64748b" for name in plot_rows["candidate"]]
    ax.barh(plot_rows["candidate"], plot_rows["private_minus_public"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Private - public Pearson")
    ax.set_title("Largest Private-over-Public Score Gaps")
    ax.grid(axis="x", alpha=0.25)
    savefig("public_private_gap_bar.png")

    return {
        "public_private_corr": corr,
        "public_trap_count_vs_selected": int(len(public_traps)),
        "public_best_candidate": str(history.sort_values("public", ascending=False).iloc[0]["candidate"]),
        "private_best_candidate": str(history.sort_values("private", ascending=False).iloc[0]["candidate"]),
        "post_freeze_public": float(selected["public"]),
        "post_freeze_private": float(selected["private"]),
    }


def build_competition_leaderboard_tables() -> dict[str, Any]:
    if not LEADERBOARD_PRIVATE_PATH.exists() or not LEADERBOARD_PUBLIC_PATH.exists():
        return {}

    private_board = pd.read_csv(LEADERBOARD_PRIVATE_PATH)
    public_board = pd.read_csv(LEADERBOARD_PUBLIC_PATH)
    for frame in [private_board, public_board]:
        for col in ["publicScore", "publicRank", "privateScore", "privateRank", "rankDiff", "attempts"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    valid_score_mask = (
        private_board[["publicScore", "privateScore"]].notna().all(axis=1)
        & (private_board["publicScore"] > -0.5)
        & (private_board["privateScore"] > -0.5)
    )
    score_board = private_board.loc[valid_score_mask].copy()

    private_board.to_csv(ARTIFACT_DIR / "leaderboard_private_desc_clean.csv", index=False)
    public_board.to_csv(ARTIFACT_DIR / "leaderboard_public_desc_clean.csv", index=False)

    top_public = private_board.nsmallest(20, "publicRank")
    top_private = private_board.nsmallest(20, "privateRank")
    public_winner = private_board.nsmallest(1, "publicRank").iloc[0]
    private_winner = private_board.nsmallest(1, "privateRank").iloc[0]
    not_null_rows = private_board[private_board["teamName"].astype(str).str.lower() == "not_null"]
    not_null = not_null_rows.iloc[0] if not not_null_rows.empty else None

    post_freeze_private = 0.11043
    post_freeze_public = 0.06547
    hypothetical_private_rank = int((private_board["privateScore"] > post_freeze_private).sum() + 1)
    hypothetical_public_rank = int((private_board["publicScore"] > post_freeze_public).sum() + 1)

    summary = pd.DataFrame(
        [
            {
                "team_count": int(len(private_board)),
                "score_valid_team_count": int(len(score_board)),
                "invalid_score_row_count": int((~valid_score_mask).sum()),
                "score_corr": float(score_board[["publicScore", "privateScore"]].corr().iloc[0, 1]),
                "rank_corr": float(private_board[["publicRank", "privateRank"]].corr().iloc[0, 1]),
                "public_winner": public_winner["teamName"],
                "public_winner_public": public_winner["publicScore"],
                "public_winner_private": public_winner["privateScore"],
                "public_winner_private_rank": public_winner["privateRank"],
                "private_winner": private_winner["teamName"],
                "private_winner_private": private_winner["privateScore"],
                "private_winner_public": private_winner["publicScore"],
                "private_winner_public_rank": private_winner["publicRank"],
                "top20_public_mean_private_rank": float(top_public["privateRank"].mean()),
                "top20_private_mean_public_rank": float(top_private["publicRank"].mean()),
                "top20_public_private_rank_over_100_count": int((top_public["privateRank"] > 100).sum()),
                "post_freeze_public": post_freeze_public,
                "post_freeze_private": post_freeze_private,
                "post_freeze_hypothetical_public_rank": hypothetical_public_rank,
                "post_freeze_hypothetical_private_rank": hypothetical_private_rank,
                "not_null_public": float(not_null["publicScore"]) if not_null is not None else np.nan,
                "not_null_private": float(not_null["privateScore"]) if not_null is not None else np.nan,
                "not_null_public_rank": int(not_null["publicRank"]) if not_null is not None else np.nan,
                "not_null_private_rank": int(not_null["privateRank"]) if not_null is not None else np.nan,
                "not_null_attempts": int(not_null["attempts"]) if not_null is not None else np.nan,
                "kaggle_cli_submission_count": 50,
            }
        ]
    )
    summary.to_csv(ARTIFACT_DIR / "leaderboard_public_private_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(score_board["publicScore"], score_board["privateScore"], s=18, alpha=0.35, color="#64748b", label="Final leaderboard teams")
    if not_null is not None:
        ax.scatter([not_null["publicScore"]], [not_null["privateScore"]], s=100, color="#f59e0b", label="Not_Null final board row")
    ax.scatter([post_freeze_public], [post_freeze_private], s=120, facecolors="none", edgecolors="#dc2626", linewidths=2.0, label="Beta7 final post-freeze")
    x_pad = max((score_board["publicScore"].max() - score_board["publicScore"].min()) * 0.06, 0.005)
    y_pad = max((score_board["privateScore"].max() - score_board["privateScore"].min()) * 0.06, 0.005)
    ax.set_xlim(score_board["publicScore"].min() - x_pad, score_board["publicScore"].max() + x_pad)
    ax.set_ylim(score_board["privateScore"].min() - y_pad, score_board["privateScore"].max() + y_pad)
    ax.set_xlabel("Public score")
    ax.set_ylabel("Private score")
    ax.set_title("Competition Public vs Private Scores")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    savefig("leaderboard_score_scatter.png")

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(private_board["publicRank"], private_board["privateRank"], s=18, alpha=0.35, color="#64748b")
    limit = int(max(private_board["publicRank"].max(), private_board["privateRank"].max()))
    ax.plot([1, limit], [1, limit], linestyle="--", linewidth=1.0, color="#94a3b8")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Public rank")
    ax.set_ylabel("Private rank")
    ax.set_title("Public Rank vs Private Rank")
    ax.grid(alpha=0.25)
    savefig("leaderboard_rank_scatter.png")

    return {
        "leaderboard_team_count": int(len(private_board)),
        "leaderboard_score_valid_team_count": int(len(score_board)),
        "leaderboard_invalid_score_row_count": int((~valid_score_mask).sum()),
        "leaderboard_score_corr": float(summary.iloc[0]["score_corr"]),
        "leaderboard_rank_corr": float(summary.iloc[0]["rank_corr"]),
        "top20_public_mean_private_rank": float(summary.iloc[0]["top20_public_mean_private_rank"]),
        "top20_public_private_rank_over_100_count": int(summary.iloc[0]["top20_public_private_rank_over_100_count"]),
        "post_freeze_hypothetical_private_rank": hypothetical_private_rank,
        "post_freeze_hypothetical_public_rank": hypothetical_public_rank,
        "not_null_attempts": int(summary.iloc[0]["not_null_attempts"]) if not pd.isna(summary.iloc[0]["not_null_attempts"]) else None,
        "kaggle_cli_submission_count": 50,
    }


def main() -> int:
    args = parse_args()
    ensure_dir(ARTIFACT_DIR)
    ensure_dir(FIGURE_DIR)

    summary: dict[str, Any] = {}
    summary.update(build_rolling_tables())
    feature_summary = build_feature_stability_tables()
    summary.update(feature_summary)
    summary.update(build_feature_drift(feature_summary["top100_stable_features_for_drift"]))
    if not args.skip_embargo:
        summary.update(run_embargo_check(args.ridge_alpha, args.embargo_fraction))
    summary.update(build_public_private_tables())
    summary.update(build_competition_leaderboard_tables())

    write_json(
        ARTIFACT_DIR / "temporal_report_artifacts_summary.json",
        {
            **summary,
            "artifact_dir": str(ARTIFACT_DIR.relative_to(ROOT)),
            "figure_dir": str(FIGURE_DIR.relative_to(ROOT)),
        },
    )

    print("Temporal report artifacts written to:")
    print(f"  {ARTIFACT_DIR}")
    print(f"  {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
