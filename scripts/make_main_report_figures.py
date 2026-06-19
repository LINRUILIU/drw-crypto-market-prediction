from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures" / "01_main"


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str, title: str, source: str, purpose: str, manifest: list[dict[str, str]]) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    manifest.append({"file": name, "title": title, "source": source, "purpose": purpose})


def read_prediction(path: str) -> np.ndarray:
    frame = pd.read_csv(ROOT / path)
    if "prediction" in frame.columns:
        return frame["prediction"].to_numpy(dtype=np.float64)
    return frame.iloc[:, -1].to_numpy(dtype=np.float64)


def plot_score_ladder(manifest: list[dict[str, str]]) -> None:
    stages = [
        ("Beta2\nRidge", 0.09638),
        ("Beta3\nSpearman", 0.10003),
        ("Beta4\nSHAP-XGB", 0.10043),
        ("Beta5\nInteraction", 0.10302),
        ("Beta6.2\nSup-AE", 0.10303),
        ("Beta7\nMLP", 0.10550),
    ]
    labels = [s[0] for s in stages]
    scores = [s[1] for s in stages]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(labels, scores, marker="o", linewidth=2.4, color="#2563eb")
    ax.fill_between(labels, scores, min(scores) - 0.001, color="#bfdbfe", alpha=0.45)
    for idx, value in enumerate(scores):
        ax.text(idx, value + 0.00025, f"{value:.5f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Private Score Ladder")
    ax.set_ylabel("Kaggle private Pearson")
    ax.set_ylim(min(scores) - 0.001, max(scores) + 0.0015)
    ax.grid(axis="y", alpha=0.25)
    savefig(
        "score_ladder.png",
        "Private score ladder",
        "docs/main_baseline_results.md",
        "Show how each successful signal family improved the selected private score.",
        manifest,
    )


def submission_points() -> pd.DataFrame:
    rows = [
        ("Beta2 Ridge", "Ridge", 0.05160, 0.09638),
        ("Beta3 Spearman", "Ridge/Spearman", 0.05456, 0.10003),
        ("Beta4 SHAP XGB", "SHAP/XGB", 0.05634, 0.10043),
        ("Beta4 MLP", "MLP", 0.05959, 0.09729),
        ("Beta5 Interaction", "Interaction", 0.06362, 0.10302),
        ("Beta5B int240", "Interaction", 0.06797, 0.10256),
        ("Beta6 AE w0.15", "AE", 0.06488, 0.10169),
        ("Beta6.2 Sup-AE", "AE", 0.06454, 0.10303),
        ("Beta7 MLP mean", "MLP", 0.06588, 0.10411),
        ("Beta7 MLP final", "MLP", 0.06553, 0.10550),
        ("Beta7 MLP heavy", "MLP", 0.06930, 0.10304),
        ("Sprint-A w0.05", "Sprint", 0.06637, 0.10514),
        ("Sprint-A w0.15", "Sprint", 0.06766, 0.10437),
        ("Sprint-B mean", "Sprint", 0.06573, 0.10376),
        ("Sprint-B seed4026", "Sprint", 0.06944, 0.10321),
        ("Sprint-B seed2526", "Sprint", 0.06397, 0.10205),
    ]
    return pd.DataFrame(rows, columns=["candidate", "family", "public", "private"])


def plot_public_private(manifest: list[dict[str, str]]) -> None:
    data = submission_points()
    colors = {
        "Ridge": "#2563eb",
        "Ridge/Spearman": "#0891b2",
        "SHAP/XGB": "#16a34a",
        "Interaction": "#9333ea",
        "AE": "#64748b",
        "MLP": "#dc2626",
        "Sprint": "#f59e0b",
    }
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for family, group in data.groupby("family"):
        ax.scatter(group["public"], group["private"], label=family, s=58, alpha=0.85, color=colors.get(family))
    final = data[data["candidate"] == "Beta7 MLP final"].iloc[0]
    ax.scatter([final["public"]], [final["private"]], s=150, facecolors="none", edgecolors="black", linewidths=1.8)
    ax.annotate("selected", (final["public"], final["private"]), xytext=(8, 8), textcoords="offset points")
    ax.set_title("Public vs Private Scores")
    ax.set_xlabel("Public Pearson")
    ax.set_ylabel("Private Pearson")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    savefig(
        "public_private_scatter.png",
        "Public/private divergence",
        "Kaggle submission history summarized in docs/main_baseline_results.md",
        "Show that higher public scores often did not imply higher private scores.",
        manifest,
    )


def plot_final_pipeline(manifest: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axis("off")
    boxes = [
        ("Pearson/Spearman\nRidge stack", (0.10, 0.72)),
        ("SHAP-stable\nXGBoost", (0.35, 0.72)),
        ("Symbolic\nInteraction Ridge", (0.60, 0.72)),
        ("Supervised\nAE8 Ridge", (0.35, 0.36)),
        ("Supervised\nMLP signal", (0.60, 0.36)),
        ("Final\nprediction", (0.84, 0.54)),
    ]
    for text, (x, y) in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eff6ff", "edgecolor": "#1d4ed8", "linewidth": 1.2},
        )
    arrows = [
        ((0.19, 0.72), (0.27, 0.72)),
        ((0.44, 0.72), (0.52, 0.72)),
        ((0.69, 0.72), (0.78, 0.57)),
        ((0.44, 0.36), (0.78, 0.51)),
        ((0.69, 0.36), (0.78, 0.53)),
        ((0.22, 0.67), (0.31, 0.41)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#334155"})
    ax.text(0.50, 0.12, "Final = 0.925 * beta6_2 + 0.075 * AdamW MLP seed2026", ha="center", fontsize=11)
    savefig(
        "final_pipeline.png",
        "Final ensemble pipeline",
        "docs/main_report_outline.md",
        "Summarize the selected model as layered complementary signals.",
        manifest,
    )


def plot_feature_funnel(manifest: list[dict[str, str]]) -> None:
    labels = ["Raw features", "Core pool", "Pairwise candidates", "Selected interactions", "Structured MLP input"]
    values = [785, 40, 5460, 120, 160]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=["#2563eb", "#0891b2", "#9333ea", "#16a34a", "#dc2626"])
    ax.set_yscale("log")
    ax.set_title("Feature Engineering Funnel")
    ax.set_ylabel("Count (log scale)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08, str(value), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    savefig(
        "feature_funnel.png",
        "Feature engineering funnel",
        "runs/01_main/beta5_interactions/",
        "Show the compression and expansion path from raw features to structured inputs.",
        manifest,
    )


def plot_interaction_distribution(manifest: list[dict[str, str]]) -> None:
    path = ROOT / "runs" / "01_main" / "beta5_interactions" / "selected_interaction_definitions.csv"
    frame = pd.read_csv(path)
    counts = frame["operation"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#16a34a")
    ax.set_title("Selected Interaction Operator Distribution")
    ax.set_ylabel("Selected feature count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", va="bottom", fontsize=9)
    savefig(
        "interaction_operator_distribution.png",
        "Interaction operator distribution",
        "runs/01_main/beta5_interactions/selected_interaction_definitions.csv",
        "Show which symbolic operators survived selection.",
        manifest,
    )


def plot_component_correlation(manifest: list[dict[str, str]]) -> None:
    candidates = {
        "top50_ridge": "runs/01_main/pearson_topk/valid_predictions/valid_top50_ridge.csv",
        "top100_ridge": "runs/01_main/pearson_topk/valid_predictions/valid_top100_ridge.csv",
        "spearman50": "runs/01_main/beta3_new_feature_signals/valid_predictions/valid_spearman_top50_ridge.csv",
        "shap_xgb": "runs/01_main/beta4_shap_stable/valid_predictions/valid_shap_stable_xgboost.csv",
        "interaction": "runs/01_main/beta5_interactions/valid_predictions/valid_ridge_interactions_only.csv",
        "sup_ae8": "runs/01_main/beta6_2_supervised_ae/valid_predictions/valid_ae8_supervised_mse005_ridge.csv",
    }
    preds: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    for name, path in candidates.items():
        full = ROOT / path
        if not full.exists():
            continue
        pred = read_prediction(path)
        preds[name] = pred
        lengths[name] = len(pred)
    if not preds:
        return
    target_len = max(set(lengths.values()), key=list(lengths.values()).count)
    aligned = {name: pred for name, pred in preds.items() if len(pred) == target_len}
    names = list(aligned)
    matrix = np.corrcoef(np.column_stack([aligned[name] for name in names]).T)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Validation Prediction Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(
        "component_correlation_heatmap.png",
        "Component prediction correlation heatmap",
        "runs/01_main/*/valid_predictions/",
        "Show that useful components are correlated but not identical.",
        manifest,
    )


def plot_holdout_private_delta(manifest: list[dict[str, str]]) -> None:
    rows = [
        ("Beta7 MLP w0.075", 0.002322, 0.10550 - 0.10303),
        ("Beta7 MLP w0.10", 0.002507, 0.10304 - 0.10303),
        ("Beta7.1 w0.0875", 0.002447, 0.10550 - 0.10303),
        ("Sprint-A w0.05", 0.000030, 0.10514 - 0.10550),
        ("Sprint-A w0.15", -0.000157, 0.10437 - 0.10550),
        ("Sprint-B mean", 0.001358, 0.10376 - 0.10550),
        ("Sprint-B seed4026", 0.004536, 0.10321 - 0.10550),
    ]
    frame = pd.DataFrame(rows, columns=["candidate", "holdout_delta", "private_delta"])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(0, color="#334155", linewidth=1)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.scatter(frame["holdout_delta"], frame["private_delta"], color="#9333ea", s=65)
    for _, row in frame.iterrows():
        ax.annotate(row["candidate"], (row["holdout_delta"], row["private_delta"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Holdout Delta vs Private Delta")
    ax.set_xlabel("Holdout Pearson delta")
    ax.set_ylabel("Private score delta")
    ax.grid(alpha=0.25)
    savefig(
        "holdout_private_delta.png",
        "Holdout/private delta comparison",
        "runs/01_main/*/candidate metrics and Kaggle submissions",
        "Show why local improvements were not always trusted late in the project.",
        manifest,
    )


def plot_signal_family_summary(manifest: list[dict[str, str]]) -> None:
    rows = [
        ("Pearson Ridge", "selected", 3),
        ("Spearman Ridge", "selected", 3),
        ("SHAP XGB", "selected", 2),
        ("Interaction Ridge", "selected", 3),
        ("Supervised AE", "tiny selected", 1),
        ("Supervised MLP", "selected, unstable", 2),
        ("Residual Ridge", "failed/unstable", -1),
        ("Medoid Ridge", "failed", -1),
        ("Wide interactions", "public trap", -2),
        ("Unsupervised AE", "failed", -1),
    ]
    frame = pd.DataFrame(rows, columns=["family", "status", "score"])
    colors = frame["score"].map(lambda x: "#16a34a" if x > 1 else "#64748b" if x > 0 else "#dc2626")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(frame["family"], frame["score"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    for y, row in enumerate(frame.itertuples()):
        x = row.score
        ha = "left" if x >= 0 else "right"
        offset = 0.08 if x >= 0 else -0.08
        ax.text(x + offset, y, row.status, va="center", ha=ha, fontsize=8)
    ax.set_title("Signal Family Outcome Summary")
    ax.set_xlabel("Qualitative transfer outcome")
    ax.set_xlim(-2.8, 3.8)
    ax.grid(axis="x", alpha=0.2)
    savefig(
        "signal_family_summary.png",
        "Signal family summary",
        "docs/main_task_retrospective.md",
        "Summarize selected, weak, failed, and public-trap directions.",
        manifest,
    )


def main() -> int:
    ensure_out_dir()
    manifest: list[dict[str, str]] = []
    plot_score_ladder(manifest)
    plot_public_private(manifest)
    plot_final_pipeline(manifest)
    plot_feature_funnel(manifest)
    plot_interaction_distribution(manifest)
    plot_component_correlation(manifest)
    plot_holdout_private_delta(manifest)
    plot_signal_family_summary(manifest)
    pd.DataFrame(manifest).to_csv(OUT_DIR / "figure_manifest.csv", index=False)
    print(f"Wrote {len(manifest)} figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
