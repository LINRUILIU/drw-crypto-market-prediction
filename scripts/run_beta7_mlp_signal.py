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

from drw_crypto.interaction_features import definitions_from_frame, transform_interactions  # noqa: E402
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
    fit_ridge,
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


class MatrixStandardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MatrixStandardizer":
        self.mean_ = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale)] = 1.0
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("MatrixStandardizer must be fitted before transform.")
        out = x.astype(np.float32, copy=True)
        out -= self.mean_
        out /= self.scale_
        return out

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Beta7 supervised MLP signal experiments.")
    parser.add_argument("--config", default="configs/01_main_beta7_mlp_signal.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merged_model(defaults: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(model_cfg)
    return merged


def load_prediction_csv(path: Path, prediction_col: str, expected_len: int | None = None) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if prediction_col in frame.columns:
        pred = frame[prediction_col].to_numpy(dtype=np.float64)
    elif "prediction" in frame.columns:
        pred = frame["prediction"].to_numpy(dtype=np.float64)
    else:
        pred = frame.iloc[:, -1].to_numpy(dtype=np.float64)
    if expected_len is not None and len(pred) != expected_len:
        return None
    return pred


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


def make_optimizer(torch: Any, model: Any, cfg: dict[str, Any]) -> Any:
    optimizer_name = str(cfg.get("optimizer", "sgd")).lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.get("learning_rate", 0.01)),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("learning_rate", 0.001)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def standardize_target(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(y))
    scale = float(np.std(y))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return ((y - mean) / scale).astype(np.float32), mean, scale


def train_mlp_early_stop(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_valid: np.ndarray,
    y_valid_raw: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    torch = setup_torch(cfg)
    seed = int(cfg.get("seed", 2026))
    seed_everything(torch, seed)
    device = torch.device("cpu")

    y_train, y_mean, y_std = standardize_target(y_train_raw)
    model = MLPRegressor(
        torch,
        input_dim=x_train.shape[1],
        hidden_sizes=[int(value) for value in cfg["hidden_sizes"]],
        dropout=float(cfg.get("dropout", 0.0)),
    ).model.to(device)
    optimizer = make_optimizer(torch, model, cfg)
    mse = torch.nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train.astype("float32", copy=False)),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4096)),
        shuffle=True,
        generator=generator,
    )
    x_valid_tensor = torch.from_numpy(x_valid.astype("float32", copy=False)).to(device)
    mse_weight = float(cfg.get("mse_weight", 0.6))
    pearson_weight = float(cfg.get("pearson_weight", 0.4))
    max_epochs = int(cfg.get("max_epochs", 80))
    patience_left = int(cfg.get("patience", 12))
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_pred_raw: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        mse_sum = 0.0
        pearson_sum = 0.0
        train_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).view(-1)
            target = yb.view(-1)
            mse_loss = mse(pred, target)
            corr_loss = pearson_loss(torch, pred, target)
            loss = mse_weight * mse_loss + pearson_weight * corr_loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(xb)
            mse_sum += float(mse_loss.detach().cpu()) * len(xb)
            pearson_sum += float(corr_loss.detach().cpu()) * len(xb)
            train_count += len(xb)

        model.eval()
        with torch.no_grad():
            valid_norm = model(x_valid_tensor).cpu().numpy().reshape(-1)
        valid_pred_raw = valid_norm.astype(np.float64) * y_std + y_mean
        valid_pearson = pearson_corr(y_valid_raw, valid_pred_raw)
        valid_rmse = rmse(y_valid_raw, valid_pred_raw)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(train_count, 1),
                "train_mse_loss": mse_sum / max(train_count, 1),
                "train_pearson_loss": pearson_sum / max(train_count, 1),
                "valid_pearson": valid_pearson,
                "valid_rmse": valid_rmse,
            }
        )
        if valid_pearson > best_score:
            best_score = valid_pearson
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_pred_raw = valid_pred_raw
            patience_left = int(cfg.get("patience", 12))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    elapsed = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_pred_raw is None:
        raise RuntimeError("MLP training did not produce a validation prediction.")
    log = pd.DataFrame(rows)
    metric = {
        "pearson": best_score,
        "rmse": rmse(y_valid_raw, best_pred_raw),
        "best_epoch": best_epoch,
        "train_seconds": elapsed,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return best_pred_raw, metric, log


def train_mlp_fixed_epochs(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_pred: np.ndarray,
    cfg: dict[str, Any],
    epochs: int,
) -> np.ndarray:
    torch = setup_torch(cfg)
    seed = int(cfg.get("seed", 2026))
    seed_everything(torch, seed)
    device = torch.device("cpu")

    y_train, y_mean, y_std = standardize_target(y_train_raw)
    model = MLPRegressor(
        torch,
        input_dim=x_train.shape[1],
        hidden_sizes=[int(value) for value in cfg["hidden_sizes"]],
        dropout=float(cfg.get("dropout", 0.0)),
    ).model.to(device)
    optimizer = make_optimizer(torch, model, cfg)
    mse = torch.nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train.astype("float32", copy=False)),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
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
            pred = model(xb).view(-1)
            target = yb.view(-1)
            loss = mse_weight * mse(pred, target) + pearson_weight * pearson_loss(torch, pred, target)
            loss.backward()
            optimizer.step()

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_pred), int(cfg.get("batch_size", 4096))):
            batch = torch.from_numpy(x_pred[start : start + int(cfg.get("batch_size", 4096))].astype("float32", copy=False)).to(device)
            outputs.append(model(batch).cpu().numpy().reshape(-1))
    pred_norm = np.concatenate(outputs).astype(np.float64)
    return pred_norm * y_std + y_mean


def xgb_params(params: dict[str, Any], random_state: int) -> dict[str, Any]:
    return {
        "n_estimators": int(params.get("n_estimators", 800)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "max_depth": int(params.get("max_depth", 5)),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "objective": "reg:squarederror",
        "random_state": int(random_state),
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }


def train_xgb_valid_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> np.ndarray:
    import xgboost as xgb

    model_params = xgb_params(params, random_state)
    early_stopping_rounds = int(params.get("early_stopping_rounds", 80))
    if early_stopping_rounds > 0:
        model_params["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBRegressor(**model_params)
    try:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    except TypeError:
        model_params.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(**model_params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model.predict(x_valid)


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def signal_metrics(
    name: str,
    signal_type: str,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    y_valid: np.ndarray,
    current_best_valid: np.ndarray,
    current_best_test: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pearson = pearson_corr(y_valid, valid_pred)
    row: dict[str, Any] = {
        "signal": name,
        "signal_type": signal_type,
        "pearson": pearson,
        "rmse": rmse(y_valid, valid_pred),
        "holdout_delta_current_best": pearson - pearson_corr(y_valid, current_best_valid),
        "valid_prediction_std": float(np.std(valid_pred)),
        "submission_prediction_std": float(np.std(test_pred)),
        "valid_corr_current_best": pearson_corr(current_best_valid, valid_pred),
        "submission_corr_current_best": pearson_corr(current_best_test, test_pred),
    }
    if extra:
        row.update(extra)
    return row


def add_blend_candidate(
    rows: list[dict[str, Any]],
    signal_name: str,
    signal_type: str,
    signal_weight: float,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    current_best_valid: np.ndarray,
    current_best_test: np.ndarray,
    y_valid: np.ndarray,
    sample_submission: pd.DataFrame | None,
    test_df: pd.DataFrame,
    prediction_col: str,
    id_col: str | None,
    submission_dir: Path,
) -> None:
    current_weight = 1.0 - float(signal_weight)
    blend_valid = current_weight * current_best_valid + float(signal_weight) * valid_pred
    blend_test = current_weight * current_best_test + float(signal_weight) * test_pred
    candidate = f"blend_current_w{safe_name(current_weight)}_{signal_name}_w{safe_name(signal_weight)}"
    submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
    path = submission_dir / f"submission_{candidate}.csv"
    submission.to_csv(path, index=False)
    holdout = pearson_corr(y_valid, blend_valid)
    rows.append(
        {
            "candidate": candidate,
            "signal": signal_name,
            "signal_type": signal_type,
            "current_best_weight": current_weight,
            "signal_weight": float(signal_weight),
            "holdout_pearson": holdout,
            "holdout_rmse": rmse(y_valid, blend_valid),
            "holdout_delta_current_best": holdout - pearson_corr(y_valid, current_best_valid),
            "valid_prediction_std": float(np.std(blend_valid)),
            "submission_prediction_std": float(np.std(blend_test)),
            "std_ratio_current_best": float(np.std(blend_test) / np.std(current_best_test)),
            **summarize_delta(blend_valid, current_best_valid, "valid"),
            **summarize_delta(blend_test, current_best_test, "submission"),
            "path": str(path.relative_to(ROOT)),
        }
    )


def select_candidates(candidates: pd.DataFrame, max_selected: int, holdout_floor_delta: float) -> pd.DataFrame:
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty:
        return eligible
    selected_names: list[str] = []
    reasons: dict[str, str] = {}

    seed_mean = eligible[
        (eligible["signal_type"] == "ensemble")
        & (eligible["signal"].str.contains("seed_mean"))
        & (np.isclose(eligible["signal_weight"].astype(float), 0.025))
    ].sort_values(["holdout_pearson", "submission_corr_current_best"], ascending=[False, True])
    if not seed_mean.empty:
        row = seed_mean.iloc[0]
        selected_names.append(str(row["candidate"]))
        reasons[str(row["candidate"])] = "best_0p025_seed_mean"

    holdout_best = eligible[eligible["signal_weight"].astype(float) <= 0.075].sort_values(
        ["holdout_pearson", "submission_corr_current_best"],
        ascending=[False, True],
    )
    for _, row in holdout_best.iterrows():
        candidate = str(row["candidate"])
        if candidate not in selected_names:
            selected_names.append(candidate)
            reasons[candidate] = "holdout_best_weight_lte_0p075"
            break

    low_corr = eligible[eligible["holdout_delta_current_best"] >= float(holdout_floor_delta)].sort_values(
        ["submission_corr_current_best", "holdout_pearson"],
        ascending=[True, False],
    )
    for _, row in low_corr.iterrows():
        candidate = str(row["candidate"])
        if candidate not in selected_names:
            selected_names.append(candidate)
            reasons[candidate] = "lowest_corr_not_collapsed"
            break

    selected = candidates[candidates["candidate"].isin(selected_names)].copy()
    selected["selection_reason"] = selected["candidate"].map(reasons)
    return selected.set_index("candidate").loc[selected_names[: int(max_selected)]].reset_index()


def build_input_matrices(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_features: list[str],
    interaction_defs: list[Any],
    prep_cfg: dict[str, Any],
    input_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_preprocessor = TabularPreprocessor(base_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_base_train = base_preprocessor.transform(train_part)
    x_base_valid = base_preprocessor.transform(valid_part)

    full_preprocessor = TabularPreprocessor(base_features).fit(
        train_df,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_base_full = full_preprocessor.transform(train_df)
    x_base_test = full_preprocessor.transform(test_df)

    matrices_train: list[np.ndarray] = []
    matrices_valid: list[np.ndarray] = []
    matrices_full: list[np.ndarray] = []
    matrices_test: list[np.ndarray] = []
    if bool(input_cfg.get("include_base_features", True)):
        matrices_train.append(x_base_train)
        matrices_valid.append(x_base_valid)
        matrices_full.append(x_base_full)
        matrices_test.append(x_base_test)
    if bool(input_cfg.get("include_interactions", True)):
        matrices_train.append(transform_interactions(x_base_train, interaction_defs, eps=1e-6))
        matrices_valid.append(transform_interactions(x_base_valid, interaction_defs, eps=1e-6))
        matrices_full.append(transform_interactions(x_base_full, interaction_defs, eps=1e-6))
        matrices_test.append(transform_interactions(x_base_test, interaction_defs, eps=1e-6))
    return (
        np.hstack(matrices_train).astype(np.float32, copy=False),
        np.hstack(matrices_valid).astype(np.float32, copy=False),
        np.hstack(matrices_full).astype(np.float32, copy=False),
        np.hstack(matrices_test).astype(np.float32, copy=False),
    )


def reconstruct_beta5_valid(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    x_interaction_train: np.ndarray,
    x_interaction_valid: np.ndarray,
    current_cfg: dict[str, Any],
    prep_cfg: dict[str, Any],
    xgb_fallback_cfg: dict[str, Any],
) -> np.ndarray:
    beta4_cfg = current_cfg["beta4_base"]
    beta5_cfg = current_cfg["beta5_interaction"]
    beta3_valid = weighted_ridge_prediction(train_part, y_train, valid_part, beta4_cfg["beta3_components"], prep_cfg, ROOT)
    shap_valid = load_prediction_csv(ROOT / beta4_cfg["shap_stable_valid_prediction_path"], "prediction", expected_len=len(y_valid))
    if shap_valid is None:
        print("SHAP-stable valid prediction length mismatch; recomputing for current train scope.")
        shap_features = read_lines(ROOT / beta4_cfg["shap_stable_features_path"])
        shap_preprocessor = TabularPreprocessor(shap_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        shap_valid = train_xgb_valid_prediction(
            shap_preprocessor.transform(train_part),
            y_train,
            shap_preprocessor.transform(valid_part),
            y_valid,
            xgb_fallback_cfg,
            random_state=2026,
        )
    beta4_valid = float(beta4_cfg["base_weight"]) * beta3_valid + float(beta4_cfg["signal_weight"]) * shap_valid
    interaction_valid = load_prediction_csv(ROOT / beta5_cfg["valid_prediction_path"], "prediction", expected_len=len(y_valid))
    if interaction_valid is None:
        interaction_valid = fit_ridge(200000.0, x_interaction_train, y_train, x_interaction_valid)
    return (1.0 - float(beta5_cfg["weight"])) * beta4_valid + float(beta5_cfg["weight"]) * interaction_valid


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    input_cfg = config["input"]
    defaults = config["mlp_defaults"]
    blend_cfg = config["blend"]
    output_cfg = config["output"]
    xgb_fallback_cfg = config["xgboost_fallback"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    log_dir = ensure_dir(run_dir / "model_logs")
    signal_submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"] / "signals")
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])

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

    base_features = read_lines(ROOT / input_cfg["base_feature_pool_path"])
    invalid = sorted(set(base_features) - set(feature_cols))
    if invalid:
        raise ValueError(f"Beta7 base feature pool contains unknown features: {invalid[:10]}")
    interaction_defs = definitions_from_frame(pd.read_csv(ROOT / input_cfg["interaction_definitions_path"]))
    write_selected_features(run_dir / "base_feature_pool.txt", base_features)
    write_selected_features(run_dir / "interaction_feature_names.txt", [definition.name for definition in interaction_defs])

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    print("Generating 160-dimensional Beta5 structured inputs.")
    x_input_train_raw, x_input_valid_raw, x_input_full_raw, x_input_test_raw = build_input_matrices(
        train_part,
        valid_part,
        train_df,
        test_df,
        base_features,
        interaction_defs,
        prep_cfg,
        input_cfg,
    )
    if x_input_train_raw.shape[1] != 160:
        raise ValueError(f"Expected 160 input features, got {x_input_train_raw.shape[1]}.")

    train_scaler = MatrixStandardizer()
    x_train = train_scaler.fit_transform(x_input_train_raw)
    x_valid = train_scaler.transform(x_input_valid_raw)
    full_scaler = MatrixStandardizer()
    x_full = full_scaler.fit_transform(x_input_full_raw)
    x_test = full_scaler.transform(x_input_test_raw)

    print("Reconstructing current-best validation/test predictions.")
    x_interaction_train = x_input_train_raw[:, len(base_features) :]
    x_interaction_valid = x_input_valid_raw[:, len(base_features) :]
    beta5_valid = reconstruct_beta5_valid(
        train_part,
        valid_part,
        y_train,
        y_valid,
        x_interaction_train,
        x_interaction_valid,
        current_cfg,
        prep_cfg,
        xgb_fallback_cfg,
    )
    beta6_cfg = current_cfg["beta6_2_supervised_ae"]
    ae_valid = load_prediction_csv(ROOT / beta6_cfg["valid_prediction_path"], "prediction", expected_len=len(y_valid))
    current_best_reference = "beta6_2"
    if ae_valid is None:
        current_best_valid = beta5_valid
        current_best_reference = "beta5_fallback" if args.max_train_rows is not None else "beta5_fallback_missing_ae"
        print(f"Using {current_best_reference} validation reference.")
    else:
        ae_weight = float(beta6_cfg["weight"])
        current_best_valid = (1.0 - ae_weight) * beta5_valid + ae_weight * ae_valid

    current_best_test = load_prediction_csv(ROOT / current_cfg["test_submission_path"], prediction_col, expected_len=len(test_df))
    if current_best_test is None:
        raise FileNotFoundError(f"Could not load current-best submission: {current_cfg['test_submission_path']}")
    current_best_holdout = pearson_corr(y_valid, current_best_valid)

    model_valid_preds: dict[str, np.ndarray] = {}
    model_test_preds: dict[str, np.ndarray] = {}
    signal_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for raw_model_cfg in config["mlp_models"]:
        model_cfg = merged_model(defaults, raw_model_cfg)
        model_name = str(model_cfg["name"])
        print(f"\n=== MLP model {model_name} ===")
        valid_pred, metric, log = train_mlp_early_stop(x_train, y_train, x_valid, y_valid, model_cfg)
        log.to_csv(log_dir / f"training_log_{model_name}.csv", index=False)
        test_pred = train_mlp_fixed_epochs(x_full, y_full, x_test, model_cfg, int(metric["best_epoch"]))

        model_valid_preds[model_name] = valid_pred
        model_test_preds[model_name] = test_pred
        save_valid_prediction(pred_dir / f"valid_{model_name}.csv", y_valid, valid_pred)
        make_submission(sample_submission, test_df, test_pred, prediction_col, id_col).to_csv(
            signal_submission_dir / f"submission_{model_name}.csv",
            index=False,
        )
        signal_rows.append(
            signal_metrics(
                model_name,
                "single",
                valid_pred,
                test_pred,
                y_valid,
                current_best_valid,
                current_best_test,
                {
                    "family": str(model_cfg.get("family", "")),
                    "optimizer": str(model_cfg.get("optimizer", "sgd")).lower(),
                    "seed": int(model_cfg.get("seed", 2026)),
                    "hidden_sizes": json.dumps([int(value) for value in model_cfg["hidden_sizes"]]),
                    "dropout": float(model_cfg.get("dropout", 0.0)),
                    "learning_rate": float(model_cfg.get("learning_rate", 0.0)),
                    **metric,
                },
            )
        )
        for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
            add_blend_candidate(
                candidate_rows,
                model_name,
                "single",
                signal_weight,
                valid_pred,
                test_pred,
                current_best_valid,
                current_best_test,
                y_valid,
                sample_submission,
                test_df,
                prediction_col,
                id_col,
                submission_dir,
            )
        gc.collect()

    ensemble_rows: list[dict[str, Any]] = []
    for ensemble_cfg in config["ensemble_signals"]:
        ensemble_name = str(ensemble_cfg["name"])
        members = [str(member) for member in ensemble_cfg["members"]]
        missing = [member for member in members if member not in model_valid_preds]
        if missing:
            raise ValueError(f"Ensemble {ensemble_name} references missing models: {missing}")
        valid_pred = np.mean([model_valid_preds[member] for member in members], axis=0)
        test_pred = np.mean([model_test_preds[member] for member in members], axis=0)
        save_valid_prediction(pred_dir / f"valid_{ensemble_name}.csv", y_valid, valid_pred)
        make_submission(sample_submission, test_df, test_pred, prediction_col, id_col).to_csv(
            signal_submission_dir / f"submission_{ensemble_name}.csv",
            index=False,
        )
        row = signal_metrics(
            ensemble_name,
            "ensemble",
            valid_pred,
            test_pred,
            y_valid,
            current_best_valid,
            current_best_test,
            {"members": json.dumps(members), "member_count": len(members)},
        )
        signal_rows.append(row)
        ensemble_rows.append(row)
        for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
            add_blend_candidate(
                candidate_rows,
                ensemble_name,
                "ensemble",
                signal_weight,
                valid_pred,
                test_pred,
                current_best_valid,
                current_best_test,
                y_valid,
                sample_submission,
                test_df,
                prediction_col,
                id_col,
                submission_dir,
            )

    signal_metrics_df = pd.DataFrame(signal_rows).sort_values(
        ["pearson", "submission_corr_current_best"],
        ascending=[False, True],
    )
    signal_metrics_df[signal_metrics_df["signal_type"] == "single"].to_csv(run_dir / "mlp_model_metrics.csv", index=False)
    pd.DataFrame(ensemble_rows).sort_values(["pearson", "submission_corr_current_best"], ascending=[False, True]).to_csv(
        run_dir / "mlp_ensemble_metrics.csv",
        index=False,
    )

    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (candidates["holdout_delta_current_best"] >= float(blend_cfg["holdout_floor_delta"]))
        & (candidates["std_ratio_current_best"] >= float(blend_cfg["min_std_ratio_current_best"]))
        & (candidates["std_ratio_current_best"] <= float(blend_cfg["max_std_ratio_current_best"]))
    )
    candidates = candidates.sort_values(
        ["eligible", "holdout_pearson", "submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates.to_csv(run_dir / "blend_candidate_metrics.csv", index=False)
    selected = select_candidates(
        candidates,
        int(blend_cfg["max_selected_submissions"]),
        float(blend_cfg["holdout_floor_delta"]),
    )
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    if not selected.empty:
        shutil.copyfile(ROOT / str(selected.iloc[0]["path"]), submission_dir / "submission_best.csv")

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best": current_cfg,
            "current_best_reference": current_best_reference,
            "current_best_holdout_pearson": current_best_holdout,
            "input_feature_count": int(x_train.shape[1]),
            "model_count": len(config["mlp_models"]),
            "ensemble_count": len(config["ensemble_signals"]),
            "best_signal": signal_metrics_df.iloc[0].to_dict(),
            "selected_candidates": selected.to_dict(orient="records"),
        },
    )

    print("\n=== MLP Signal Metrics ===")
    print(signal_metrics_df.to_string(index=False))
    print("\n=== Selected Candidates ===")
    if selected.empty:
        print("No eligible candidates selected for submission.")
    else:
        print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
