from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drw_crypto.interaction_features import definitions_from_frame, transform_interactions  # noqa: E402
from drw_crypto.io import TRAIN_CANDIDATES, ensure_dir, find_first_existing, read_table, write_json  # noqa: E402
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402
from drw_crypto.pipeline import read_lines, save_valid_prediction, split_time_ordered  # noqa: E402
from drw_crypto.preprocessing import TabularPreprocessor, infer_target_column  # noqa: E402


RUN_DIR = ROOT / "runs" / "04_signal"
FIGURE_DIR = ROOT / "reports" / "figures" / "04_signal"
HISTORICAL_BETA7_W0075_HOLDOUT = 0.1215165150755724
FINAL_MODEL_NAME = "Beta7 final reconstructed"
FINAL_SIGNAL_NAME = "wide_adamw_lr001_seed2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictive signal interpretation artifacts.")
    parser.add_argument(
        "--beta7-config",
        default="configs/01_main_beta7_mlp_signal.yaml",
        help="Beta7 MLP signal config used for the single-branch MLP reconstruction.",
    )
    parser.add_argument(
        "--beta7-refine-config",
        default="configs/01_main_beta7_1_mlp_weight_refine.yaml",
        help="Beta7.1 config used to reconstruct beta6_2 current-best validation predictions.",
    )
    parser.add_argument("--force-retrain", action="store_true", help="Retrain the MLP signal even if cached output exists.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_prediction_csv(path: Path, expected_len: int | None = None) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "prediction" in frame.columns:
        pred = frame["prediction"].to_numpy(dtype=np.float64)
    else:
        pred = frame.iloc[:, -1].to_numpy(dtype=np.float64)
    if expected_len is not None and len(pred) != expected_len:
        return None
    return pred


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def build_holdout_input_matrices(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    beta7_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    prep_cfg = beta7_config["preprocessing"]
    input_cfg = beta7_config["input"]
    base_features = read_lines(ROOT / input_cfg["base_feature_pool_path"])
    missing = sorted(set(base_features) - set(train_part.columns))
    if missing:
        raise ValueError(f"Beta7 base feature pool contains unknown features: {missing[:10]}")

    interaction_defs = definitions_from_frame(pd.read_csv(ROOT / input_cfg["interaction_definitions_path"]))
    base_preprocessor = TabularPreprocessor(base_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_base_train = base_preprocessor.transform(train_part)
    x_base_valid = base_preprocessor.transform(valid_part)

    matrices_train: list[np.ndarray] = []
    matrices_valid: list[np.ndarray] = []
    if bool(input_cfg.get("include_base_features", True)):
        matrices_train.append(x_base_train)
        matrices_valid.append(x_base_valid)
    if bool(input_cfg.get("include_interactions", True)):
        matrices_train.append(transform_interactions(x_base_train, interaction_defs, eps=1e-6))
        matrices_valid.append(transform_interactions(x_base_valid, interaction_defs, eps=1e-6))

    x_train = np.hstack(matrices_train).astype(np.float32, copy=False)
    x_valid = np.hstack(matrices_valid).astype(np.float32, copy=False)
    if x_train.shape[1] != 160:
        raise ValueError(f"Expected 160 Beta7 structured features, got {x_train.shape[1]}")
    return x_train, x_valid


def reconstruct_or_load_mlp_signal(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    beta7_config: dict[str, Any],
    force_retrain: bool,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    beta7_module = load_script_module("beta7_mlp_signal_module", ROOT / "scripts" / "run_beta7_mlp_signal.py")
    cache_path = RUN_DIR / f"reconstructed_valid_{FINAL_SIGNAL_NAME}.csv"
    log_path = RUN_DIR / f"reconstructed_training_log_{FINAL_SIGNAL_NAME}.csv"
    expected_len = len(y_valid)

    if not force_retrain:
        cached = load_prediction_csv(cache_path, expected_len)
        if cached is not None:
            log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
            return cached, log, "cached_reconstructed"

    signal_path = ROOT / "runs" / "01_main" / "beta7_mlp_signal" / "valid_predictions" / f"valid_{FINAL_SIGNAL_NAME}.csv"
    existing = load_prediction_csv(signal_path, expected_len)
    if existing is not None and not force_retrain:
        log = pd.DataFrame()
        save_valid_prediction(cache_path, y_valid, existing)
        return existing, log, "existing_full_length"

    defaults = beta7_config["mlp_defaults"]
    model_cfg: dict[str, Any] | None = None
    for candidate in beta7_config["mlp_models"]:
        if str(candidate["name"]) == FINAL_SIGNAL_NAME:
            model_cfg = beta7_module.merged_model(defaults, candidate)
            break
    if model_cfg is None:
        raise ValueError(f"Could not find MLP model config {FINAL_SIGNAL_NAME!r}")

    x_train_raw, x_valid_raw = build_holdout_input_matrices(train_part, valid_part, beta7_config)
    scaler = beta7_module.MatrixStandardizer()
    x_train = scaler.fit_transform(x_train_raw)
    x_valid = scaler.transform(x_valid_raw)
    signal_valid, metric, log = beta7_module.train_mlp_early_stop(x_train, y_train, x_valid, y_valid, model_cfg)
    log.to_csv(log_path, index=False)
    save_valid_prediction(cache_path, y_valid, signal_valid)

    metric_frame = pd.DataFrame([{**metric, "signal": FINAL_SIGNAL_NAME, "source": "retrained_holdout"}])
    metric_frame.to_csv(RUN_DIR / f"reconstructed_metrics_{FINAL_SIGNAL_NAME}.csv", index=False)
    return signal_valid, log, "retrained_holdout"


def assign_signal_groups(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    order = np.argsort(frame["prediction"].to_numpy(dtype=np.float64), kind="mergesort")
    rank_position = np.empty(n, dtype=np.int64)
    rank_position[order] = np.arange(n, dtype=np.int64)
    decile = np.floor(rank_position * 10 / n).astype(np.int64) + 1
    decile = np.clip(decile, 1, 10)

    out = frame.copy()
    out["prediction_rank_pct"] = (rank_position + 1) / n
    out["prediction_decile"] = decile
    out["signal_group"] = np.where(decile == 1, "bottom_10", np.where(decile == 10, "top_10", "middle_80"))
    out["is_strong_signal"] = out["signal_group"].isin(["bottom_10", "top_10"])
    out["prediction_sign"] = np.sign(out["prediction"]).astype(np.int8)
    out["target_sign"] = np.sign(out["target"]).astype(np.int8)
    return out


def group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    order = ["bottom_10", "middle_80", "top_10"]
    rows: list[dict[str, Any]] = []
    for group in order:
        part = frame[frame["signal_group"] == group]
        rows.append(
            {
                "signal_group": group,
                "sample_count": int(len(part)),
                "prediction_mean": float(part["prediction"].mean()),
                "target_mean": float(part["target"].mean()),
                "target_median": float(part["target"].median()),
                "target_std": float(part["target"].std(ddof=0)),
                "target_q25": float(part["target"].quantile(0.25)),
                "target_q75": float(part["target"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def decile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decile, part in frame.groupby("prediction_decile", sort=True):
        sign_accuracy = float((part["prediction_sign"] == part["target_sign"]).mean())
        rows.append(
            {
                "prediction_decile": int(decile),
                "sample_count": int(len(part)),
                "prediction_min": float(part["prediction"].min()),
                "prediction_max": float(part["prediction"].max()),
                "prediction_mean": float(part["prediction"].mean()),
                "target_mean": float(part["target"].mean()),
                "target_median": float(part["target"].median()),
                "target_std": float(part["target"].std(ddof=0)),
                "sign_accuracy": sign_accuracy,
            }
        )
    return pd.DataFrame(rows)


def directional_summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("all", frame),
        ("strong_signal_top_bottom_10", frame[frame["is_strong_signal"]]),
        ("middle_80", frame[frame["signal_group"] == "middle_80"]),
        ("top_10", frame[frame["signal_group"] == "top_10"]),
        ("bottom_10", frame[frame["signal_group"] == "bottom_10"]),
    ]
    rows: list[dict[str, Any]] = []
    for name, part in groups:
        rows.append(
            {
                "group": name,
                "sample_count": int(len(part)),
                "sign_accuracy": float((part["prediction_sign"] == part["target_sign"]).mean()),
                "positive_prediction_rate": float((part["prediction_sign"] > 0).mean()),
                "positive_target_rate": float((part["target_sign"] > 0).mean()),
                "target_mean": float(part["target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_top_bottom_spread(grouped: pd.DataFrame) -> pd.DataFrame:
    bottom = grouped[grouped["signal_group"] == "bottom_10"].iloc[0]
    top = grouped[grouped["signal_group"] == "top_10"].iloc[0]
    return pd.DataFrame(
        [
            {
                "long_group": "top_10",
                "short_group": "bottom_10",
                "long_target_mean": top["target_mean"],
                "short_target_mean": bottom["target_mean"],
                "top_minus_bottom_target_spread": float(top["target_mean"] - bottom["target_mean"]),
                "long_prediction_mean": top["prediction_mean"],
                "short_prediction_mean": bottom["prediction_mean"],
                "sample_count_each_side_min": int(min(top["sample_count"], bottom["sample_count"])),
                "interpretation": "No-cost diagnostic only; not a tradable backtest.",
            }
        ]
    )


def make_metrics(frame: pd.DataFrame, grouped: pd.DataFrame, directional: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    pearson = pearson_corr(frame["target"].to_numpy(dtype=np.float64), frame["prediction"].to_numpy(dtype=np.float64))
    spearman = float(frame["target"].corr(frame["prediction"], method="spearman"))
    return pd.DataFrame(
        [
            {
                "model_name": FINAL_MODEL_NAME,
                "sample_count": int(len(frame)),
                "pearson": pearson,
                "spearman": spearman,
                "rmse": rmse(frame["target"].to_numpy(dtype=np.float64), frame["prediction"].to_numpy(dtype=np.float64)),
                "overall_sign_accuracy": float(directional[directional["group"] == "all"]["sign_accuracy"].iloc[0]),
                "strong_signal_sign_accuracy": float(
                    directional[directional["group"] == "strong_signal_top_bottom_10"]["sign_accuracy"].iloc[0]
                ),
                "top_target_mean": float(grouped[grouped["signal_group"] == "top_10"]["target_mean"].iloc[0]),
                "middle_target_mean": float(grouped[grouped["signal_group"] == "middle_80"]["target_mean"].iloc[0]),
                "bottom_target_mean": float(grouped[grouped["signal_group"] == "bottom_10"]["target_mean"].iloc[0]),
                "top_minus_bottom_target_spread": float(spread["top_minus_bottom_target_spread"].iloc[0]),
            }
        ]
    )


def plot_top_middle_bottom(grouped: pd.DataFrame) -> None:
    labels = ["Bottom 10%", "Middle 80%", "Top 10%"]
    values = grouped["target_mean"].to_numpy(dtype=np.float64)
    colors = ["#64748b", "#94a3b8", "#2563eb"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_ylabel("Mean target")
    ax.set_title("Realized Target by Prediction Group")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom" if value >= 0 else "top")
    ax.grid(axis="y", alpha=0.25)
    savefig("top_middle_bottom_target_mean.png")


def plot_deciles(deciles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(deciles["prediction_decile"], deciles["target_mean"], marker="o", linewidth=2.2, color="#2563eb")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_xticks(deciles["prediction_decile"])
    ax.set_xlabel("Prediction decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Mean target")
    ax.set_title("Mean Target by Prediction Decile")
    ax.grid(alpha=0.25)
    savefig("decile_target_mean.png")


def plot_directional(directional: pd.DataFrame) -> None:
    desired = ["all", "middle_80", "strong_signal_top_bottom_10"]
    plot_rows = directional[directional["group"].isin(desired)].copy()
    plot_rows["group"] = pd.Categorical(plot_rows["group"], categories=desired, ordered=True)
    plot_rows = plot_rows.sort_values("group")
    label_map = {
        "all": "All",
        "middle_80": "Middle 80%",
        "strong_signal_top_bottom_10": "Top/Bottom 10%",
    }
    labels = [label_map[item] for item in plot_rows["group"]]
    values = plot_rows["sign_accuracy"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values, color=["#64748b", "#94a3b8", "#2563eb"])
    ax.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.2, label="50% reference")
    ax.set_ylim(0.0, max(0.65, float(values.max()) + 0.04))
    ax.set_ylabel("Sign agreement")
    ax.set_title("Directional Agreement by Signal Strength")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.1%}", ha="center", va="bottom")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    savefig("directional_accuracy_summary.png")


def plot_summary(grouped: pd.DataFrame, deciles: pd.DataFrame, directional: pd.DataFrame, metrics: pd.DataFrame) -> None:
    metric = metrics.iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    labels = ["Bottom 10%", "Middle 80%", "Top 10%"]
    axes[0, 0].bar(labels, grouped["target_mean"], color=["#64748b", "#94a3b8", "#2563eb"])
    axes[0, 0].axhline(0, color="#334155", linewidth=1)
    axes[0, 0].set_title("Group Mean Target")
    axes[0, 0].set_ylabel("Mean target")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].plot(deciles["prediction_decile"], deciles["target_mean"], marker="o", color="#2563eb")
    axes[0, 1].axhline(0, color="#334155", linewidth=1)
    axes[0, 1].set_title("Decile Mean Target")
    axes[0, 1].set_xlabel("Prediction decile")
    axes[0, 1].set_ylabel("Mean target")
    axes[0, 1].grid(alpha=0.25)

    desired = ["all", "middle_80", "strong_signal_top_bottom_10"]
    plot_rows = directional[directional["group"].isin(desired)].copy()
    plot_rows["group"] = pd.Categorical(plot_rows["group"], categories=desired, ordered=True)
    plot_rows = plot_rows.sort_values("group")
    label_map = {
        "all": "All",
        "middle_80": "Middle 80%",
        "strong_signal_top_bottom_10": "Top/Bottom 10%",
    }
    labels = [label_map[item] for item in plot_rows["group"]]
    axes[1, 0].bar(labels, plot_rows["sign_accuracy"], color=["#64748b", "#94a3b8", "#2563eb"])
    axes[1, 0].axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("Directional Agreement")
    axes[1, 0].set_ylabel("Sign agreement")
    axes[1, 0].set_ylim(0.0, max(0.65, float(plot_rows["sign_accuracy"].max()) + 0.04))
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].axis("off")
    summary_text = "\n".join(
        [
            f"Pearson: {metric['pearson']:.4f}",
            f"Spearman: {metric['spearman']:.4f}",
            f"Top-bottom target spread: {metric['top_minus_bottom_target_spread']:.4f}",
            f"Overall sign agreement: {metric['overall_sign_accuracy']:.1%}",
            f"Strong-signal sign agreement: {metric['strong_signal_sign_accuracy']:.1%}",
            "No-cost diagnostic only; not a backtest.",
        ]
    )
    axes[1, 1].text(
        0.02,
        0.74,
        summary_text,
        fontsize=12,
        va="top",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    axes[1, 1].set_title("Signal Interpretation Summary")
    savefig("signal_interpretation_summary.png")


def main() -> int:
    args = parse_args()
    ensure_dir(RUN_DIR)
    ensure_dir(FIGURE_DIR)

    beta7_config = load_config(args.beta7_config)
    refine_config = load_config(args.beta7_refine_config)
    data_cfg = refine_config["data"]
    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    if train_path is None:
        raise FileNotFoundError("Missing train file under data/raw.")

    train_df = read_table(train_path)
    target_col = infer_target_column(train_df, None, data_cfg.get("target_col"))
    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)

    beta71_module = load_script_module("beta7_1_refine_module", ROOT / "scripts" / "run_beta7_1_mlp_weight_refine.py")
    beta6_valid = beta71_module.reconstruct_beta6_2_valid(
        train_part,
        valid_part,
        y_train,
        y_valid,
        refine_config["current_best"],
        refine_config["preprocessing"],
    )
    save_valid_prediction(RUN_DIR / "reconstructed_valid_beta6_2_current_best.csv", y_valid, beta6_valid)

    signal_valid, signal_log, signal_source = reconstruct_or_load_mlp_signal(
        train_part,
        valid_part,
        y_train,
        y_valid,
        beta7_config,
        bool(args.force_retrain),
    )
    final_pred = 0.925 * beta6_valid + 0.075 * signal_valid
    save_valid_prediction(RUN_DIR / "reconstructed_valid_beta7_final.csv", y_valid, final_pred)

    split_at = len(train_part)
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(split_at, split_at + len(y_valid), dtype=np.int64),
            "local_valid_index": np.arange(len(y_valid), dtype=np.int64),
            "chronological_segment": "holdout_80_100",
            "model_name": FINAL_MODEL_NAME,
            "prediction": final_pred,
            "target": y_valid,
        }
    )
    frame = assign_signal_groups(frame)
    frame.to_csv(RUN_DIR / "signal_interpretation_table.csv", index=False)

    grouped = group_summary(frame)
    deciles = decile_summary(frame)
    directional = directional_summary(frame)
    spread = make_top_bottom_spread(grouped)
    metrics = make_metrics(frame, grouped, directional, spread)

    grouped.to_csv(RUN_DIR / "top_middle_bottom_summary.csv", index=False)
    deciles.to_csv(RUN_DIR / "decile_target_summary.csv", index=False)
    directional.to_csv(RUN_DIR / "directional_accuracy_summary.csv", index=False)
    spread.to_csv(RUN_DIR / "theoretical_top_bottom_spread.csv", index=False)
    metrics.to_csv(RUN_DIR / "metrics_signal_interpretation.csv", index=False)

    final_holdout = float(metrics["pearson"].iloc[0])
    reconstruction = pd.DataFrame(
        [
            {
                "model_name": FINAL_MODEL_NAME,
                "validation_rows": int(len(frame)),
                "sample_index_start": int(frame["sample_index"].min()),
                "sample_index_end": int(frame["sample_index"].max()),
                "beta6_2_holdout_pearson": pearson_corr(y_valid, beta6_valid),
                "mlp_signal_source": signal_source,
                "mlp_signal_holdout_pearson": pearson_corr(y_valid, signal_valid),
                "final_formula": "0.925 * beta6_2_current_best + 0.075 * wide_adamw_lr001_seed2026",
                "final_holdout_pearson": final_holdout,
                "historical_beta7_w0075_holdout_pearson": HISTORICAL_BETA7_W0075_HOLDOUT,
                "delta_vs_historical": final_holdout - HISTORICAL_BETA7_W0075_HOLDOUT,
            }
        ]
    )
    reconstruction.to_csv(RUN_DIR / "reconstruction_check.csv", index=False)

    plot_top_middle_bottom(grouped)
    plot_deciles(deciles)
    plot_directional(directional)
    plot_summary(grouped, deciles, directional, metrics)

    write_json(
        RUN_DIR / "signal_interpretation_summary.json",
        {
            "model_name": FINAL_MODEL_NAME,
            "validation_rows": int(len(frame)),
            "sample_index_start": int(frame["sample_index"].min()),
            "sample_index_end": int(frame["sample_index"].max()),
            "pearson": final_holdout,
            "spearman": float(metrics["spearman"].iloc[0]),
            "rmse": float(metrics["rmse"].iloc[0]),
            "overall_sign_accuracy": float(metrics["overall_sign_accuracy"].iloc[0]),
            "strong_signal_sign_accuracy": float(metrics["strong_signal_sign_accuracy"].iloc[0]),
            "top_minus_bottom_target_spread": float(metrics["top_minus_bottom_target_spread"].iloc[0]),
            "mlp_signal_source": signal_source,
            "signal_training_epochs": int(signal_log["epoch"].max()) if not signal_log.empty and "epoch" in signal_log else None,
            "artifact_dir": str(RUN_DIR.relative_to(ROOT)),
            "figure_dir": str(FIGURE_DIR.relative_to(ROOT)),
        },
    )

    print("Signal interpretation artifacts written to:")
    print(f"  {RUN_DIR}")
    print(f"  {FIGURE_DIR}")
    print(reconstruction.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
