from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drw_crypto.io import (  # noqa: E402
    SAMPLE_SUBMISSION_CANDIDATES,
    TEST_CANDIDATES,
    TRAIN_CANDIDATES,
    ensure_dir,
    find_first_existing,
    read_table,
    write_json,
)
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402
from drw_crypto.pipeline import (  # noqa: E402
    choose_prediction_column,
    make_submission,
    read_lines,
    safe_name,
    save_valid_prediction,
    split_time_ordered,
    weighted_ridge_prediction,
    write_selected_features,
)
from drw_crypto.preprocessing import (  # noqa: E402
    TabularPreprocessor,
    infer_feature_columns,
    infer_id_column,
    infer_target_column,
)
from drw_crypto.validation import purged_group_time_series_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Beta4-C CPU MLP experiments.")
    parser.add_argument("--config", default="configs/01_main_beta4_mlp.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip purged CV and only run 80/20 holdout.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unique_features(paths: list[str]) -> list[str]:
    features: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for feature in read_lines(ROOT / path):
            if feature not in seen:
                seen.add(feature)
                features.append(feature)
    return features


def setup_torch(cfg: dict[str, Any]) -> Any:
    import torch

    threads = int(cfg.get("torch_num_threads", 0))
    if threads > 0:
        torch.set_num_threads(threads)
    return torch


def seed_everything(torch: Any, seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


class MLPRegressor:
    def __init__(self, torch: Any, input_dim: int, hidden_sizes: list[int], dropout: float):
        nn = torch.nn
        layers: list[Any] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)


def pearson_loss(torch: Any, pred: Any, target: Any) -> Any:
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = torch.sqrt(torch.sum(pred_centered * pred_centered) * torch.sum(target_centered * target_centered) + 1e-8)
    corr = torch.sum(pred_centered * target_centered) / denom
    return 1.0 - corr


def train_mlp_early_stop(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_valid: np.ndarray,
    y_valid_raw: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = setup_torch(cfg)
    seed_everything(torch, seed)
    device = torch.device("cpu")

    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if not np.isfinite(y_std) or y_std == 0.0:
        y_std = 1.0
    y_train = ((y_train_raw - y_mean) / y_std).astype("float32")
    y_valid = ((y_valid_raw - y_mean) / y_std).astype("float32")

    model = MLPRegressor(
        torch,
        input_dim=x_train.shape[1],
        hidden_sizes=[int(value) for value in cfg["hidden_sizes"]],
        dropout=float(cfg.get("dropout", 0.0)),
    ).model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.01)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    mse = torch.nn.MSELoss()
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train.astype("float32", copy=False)),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 4096)),
        shuffle=True,
        generator=generator,
    )
    x_valid_tensor = torch.from_numpy(x_valid.astype("float32", copy=False)).to(device)

    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_pred_raw: np.ndarray | None = None
    patience_left = int(cfg.get("patience", 12))
    max_epochs = int(cfg.get("max_epochs", 80))
    mse_weight = float(cfg.get("mse_weight", 0.6))
    pearson_weight = float(cfg.get("pearson_weight", 0.4))
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse_weight * mse(pred, yb) + pearson_weight * pearson_loss(torch, pred.view(-1), yb.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_norm = model(x_valid_tensor).cpu().numpy().reshape(-1)
        valid_raw = valid_norm.astype("float64") * y_std + y_mean
        score = pearson_corr(y_valid_raw, valid_raw)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_pred_raw = valid_raw
            patience_left = int(cfg.get("patience", 12))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    elapsed = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    assert best_pred_raw is not None
    return best_pred_raw, {
        "pearson": best_score,
        "rmse": rmse(y_valid_raw, best_pred_raw),
        "best_epoch": best_epoch,
        "train_seconds": elapsed,
        "y_mean": y_mean,
        "y_std": y_std,
        "max_epochs": max_epochs,
    }


def train_mlp_fixed_epochs(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_pred: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    epochs: int,
) -> np.ndarray:
    torch = setup_torch(cfg)
    seed_everything(torch, seed)
    device = torch.device("cpu")
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if not np.isfinite(y_std) or y_std == 0.0:
        y_std = 1.0
    y_train = ((y_train_raw - y_mean) / y_std).astype("float32")

    model = MLPRegressor(
        torch,
        input_dim=x_train.shape[1],
        hidden_sizes=[int(value) for value in cfg["hidden_sizes"]],
        dropout=float(cfg.get("dropout", 0.0)),
    ).model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.01)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    mse = torch.nn.MSELoss()
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train.astype("float32", copy=False)),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 4096)),
        shuffle=True,
        generator=generator,
    )
    mse_weight = float(cfg.get("mse_weight", 0.6))
    pearson_weight = float(cfg.get("pearson_weight", 0.4))
    for _ in range(max(1, int(epochs))):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse_weight * mse(pred, yb) + pearson_weight * pearson_loss(torch, pred.view(-1), yb.view(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_pred.astype("float32", copy=False)).to(device)).cpu().numpy().reshape(-1)
    return pred_norm.astype("float64") * y_std + y_mean


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    cv_cfg = config["cv"]
    mlp_cfg = config["mlp"]
    blend_cfg = config["blend"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])
    signal_submission_dir = ensure_dir(submission_dir / "signals")

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    test_path = find_first_existing(raw_dir, data_cfg.get("test_file"), TEST_CANDIDATES)
    sample_path = find_first_existing(raw_dir, data_cfg.get("sample_submission_file"), SAMPLE_SUBMISSION_CANDIDATES)
    if train_path is None or test_path is None:
        raise FileNotFoundError("Missing train/test files under data/raw.")

    train_df = read_table(train_path)
    if args.max_train_rows is not None:
        train_df = train_df.tail(int(args.max_train_rows)).reset_index(drop=True)
    test_df = read_table(test_path)
    sample_submission = read_table(sample_path) if sample_path is not None else None

    target_col = infer_target_column(train_df, test_df, data_cfg.get("target_col"))
    id_col = infer_id_column(train_df, test_df, sample_submission, target_col, data_cfg.get("id_col"))
    feature_cols = infer_feature_columns(
        train_df,
        target_col,
        id_col,
        data_cfg.get("drop_columns", []),
        bool(prep_cfg.get("numeric_only", True)),
    )
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    feature_sets: dict[str, list[str]] = {}
    for item in config["feature_sets"]:
        name = str(item["name"])
        features = unique_features([str(path) for path in item["paths"]])
        invalid = set(features) - set(feature_cols)
        if invalid:
            raise ValueError(f"Feature set {name} contains unknown features: {sorted(invalid)[:5]}")
        feature_sets[name] = features
        write_selected_features(run_dir / f"selected_features_{name}.txt", features)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    current_best_valid = weighted_ridge_prediction(
        train_part,
        y_train,
        valid_part,
        current_cfg["components"],
        prep_cfg,
        ROOT,
    )
    current_best_test = weighted_ridge_prediction(
        train_df,
        y_full,
        test_df,
        current_cfg["components"],
        prep_cfg,
        ROOT,
    )
    current_best_holdout = pearson_corr(y_valid, current_best_valid)

    cv_rows: list[dict[str, Any]] = []
    if bool(cv_cfg.get("enabled", True)) and not args.skip_cv:
        for feature_set, features in feature_sets.items():
            print(f"\n=== MLP purged CV feature_set={feature_set} features={len(features)} ===")
            splits = purged_group_time_series_splits(
                len(train_part),
                n_groups=int(cv_cfg.get("n_groups", 6)),
                gap=int(cv_cfg.get("gap", 1)),
            )
            for split in splits:
                fold_train = train_part.iloc[split.train_indices]
                fold_valid = train_part.iloc[split.valid_indices]
                fold_y_train = fold_train[target_col].to_numpy(dtype=np.float64)
                fold_y_valid = fold_valid[target_col].to_numpy(dtype=np.float64)
                preprocessor = TabularPreprocessor(features).fit(
                    fold_train,
                    missing_fill=str(prep_cfg.get("missing_fill", "median")),
                    standardize=bool(prep_cfg.get("standardize", True)),
                )
                x_fold_train = preprocessor.transform(fold_train)
                x_fold_valid = preprocessor.transform(fold_valid)
                pred, row = train_mlp_early_stop(
                    x_fold_train,
                    fold_y_train,
                    x_fold_valid,
                    fold_y_valid,
                    mlp_cfg,
                    seed=int(mlp_cfg.get("seed", 2026)) + split.fold,
                )
                cv_rows.append(
                    {
                        "feature_set": feature_set,
                        "fold": split.fold,
                        "valid_group": split.valid_group,
                        "purged_groups": ",".join(str(group) for group in split.purged_groups),
                        "train_rows": len(split.train_indices),
                        "valid_rows": len(split.valid_indices),
                        **row,
                    }
                )
                del preprocessor, x_fold_train, x_fold_valid, pred
                gc.collect()
            pd.DataFrame(cv_rows).to_csv(run_dir / "purged_mlp_fold_metrics.csv", index=False)

    metrics_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for feature_set, features in feature_sets.items():
        print(f"\n=== MLP holdout feature_set={feature_set} features={len(features)} ===")
        preprocessor = TabularPreprocessor(features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)
        valid_pred, metric = train_mlp_early_stop(
            x_train,
            y_train,
            x_valid,
            y_valid,
            mlp_cfg,
            seed=int(mlp_cfg.get("seed", 2026)),
        )
        save_valid_prediction(pred_dir / f"valid_{feature_set}_mlp.csv", y_valid, valid_pred)

        final_preprocessor = TabularPreprocessor(features).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = final_preprocessor.transform(train_df)
        x_test = final_preprocessor.transform(test_df)
        test_pred = train_mlp_fixed_epochs(
            x_full,
            y_full,
            x_test,
            mlp_cfg,
            seed=int(mlp_cfg.get("seed", 2026)),
            epochs=int(metric["best_epoch"]),
        )
        signal_submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        signal_submission.to_csv(signal_submission_dir / f"submission_{feature_set}_mlp.csv", index=False)

        cv_part = pd.DataFrame(cv_rows)
        cv_feature = cv_part[cv_part["feature_set"] == feature_set] if not cv_part.empty else pd.DataFrame()
        cv_mean = float(cv_feature["pearson"].mean()) if not cv_feature.empty else np.nan
        cv_min = float(cv_feature["pearson"].min()) if not cv_feature.empty else np.nan
        metrics_rows.append(
            {
                "feature_set": feature_set,
                "feature_count": len(features),
                "model": "mlp",
                "pearson": pearson_corr(y_valid, valid_pred),
                "rmse": rmse(y_valid, valid_pred),
                "best_epoch": int(metric["best_epoch"]),
                "train_seconds": float(metric["train_seconds"]),
                "purged_cv_mean_pearson": cv_mean,
                "purged_cv_min_pearson": cv_min,
                "holdout_delta_current_best": pearson_corr(y_valid, valid_pred) - current_best_holdout,
            }
        )

        for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
            base_weight = 1.0 - signal_weight
            blend_valid = base_weight * current_best_valid + signal_weight * valid_pred
            blend_test = base_weight * current_best_test + signal_weight * test_pred
            candidate = f"blend_current_w{safe_name(base_weight)}_{feature_set}_mlp_w{safe_name(signal_weight)}"
            submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
            path = submission_dir / f"submission_{candidate}.csv"
            submission.to_csv(path, index=False)
            holdout = pearson_corr(y_valid, blend_valid)
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "feature_set": feature_set,
                    "feature_count": len(features),
                    "model": "mlp",
                    "current_best_weight": base_weight,
                    "signal_weight": signal_weight,
                    "holdout_pearson": holdout,
                    "holdout_rmse": rmse(y_valid, blend_valid),
                    "holdout_delta_current_best": holdout - current_best_holdout,
                    "valid_prediction_std": float(np.std(blend_valid)),
                    "submission_prediction_std": float(np.std(blend_test)),
                    "std_ratio_current_best": float(np.std(blend_test) / np.std(current_best_test)),
                    **summarize_delta(blend_valid, current_best_valid, "valid"),
                    **summarize_delta(blend_test, current_best_test, "submission"),
                    "path": str(path.relative_to(ROOT)),
                }
            )

        del preprocessor, final_preprocessor, x_train, x_valid, x_full, x_test, valid_pred, test_pred
        gc.collect()

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["purged_cv_mean_pearson", "pearson"],
        ascending=[False, False],
    )
    metrics.to_csv(run_dir / "metrics_mlp.csv", index=False)
    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (candidates["holdout_delta_current_best"] >= -float(blend_cfg.get("holdout_tolerance", 0.003)))
        & (candidates["std_ratio_current_best"] <= float(blend_cfg.get("max_std_ratio_current_best", 1.2)))
    )
    candidates = candidates.sort_values(
        ["eligible", "holdout_pearson", "submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates.to_csv(run_dir / "candidate_metrics.csv", index=False)

    selected = candidates[candidates["eligible"]].head(int(blend_cfg.get("max_selected_submissions", 3))).copy()
    if selected.empty:
        selected = candidates.head(int(blend_cfg.get("max_selected_submissions", 3))).copy()
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    if not selected.empty:
        shutil.copyfile(ROOT / str(selected.iloc[0]["path"]), submission_dir / "submission_best.csv")

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best": current_cfg,
            "current_best_holdout_pearson": current_best_holdout,
            "feature_sets": {name: len(features) for name, features in feature_sets.items()},
            "best_metric_row": metrics.iloc[0].to_dict(),
            "best_candidate": selected.iloc[0].to_dict() if not selected.empty else None,
            "torch_note": "CPU-only MLP training.",
        },
    )

    print("\n=== MLP Metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== Selected Candidates ===")
    print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
