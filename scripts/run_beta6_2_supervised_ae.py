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
    train_ridge_search,
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
    parser = argparse.ArgumentParser(description="Run Beta6.2 supervised AE experiments.")
    parser.add_argument("--config", default="configs/01_main_beta6_2_supervised_ae.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merged_variant(defaults: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(variant)
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


class AutoEncoder:
    def __init__(self, torch: Any, input_dim: int, hidden_sizes: list[int], bottleneck_dim: int, dropout: float):
        nn = torch.nn
        encoder_layers: list[Any] = []
        prev = input_dim
        for hidden in hidden_sizes:
            encoder_layers.append(nn.Linear(prev, int(hidden)))
            encoder_layers.append(nn.ReLU())
            if float(dropout) > 0:
                encoder_layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden)
        encoder_layers.append(nn.Linear(prev, int(bottleneck_dim)))
        decoder_layers: list[Any] = []
        prev = int(bottleneck_dim)
        for hidden in reversed(hidden_sizes):
            decoder_layers.append(nn.Linear(prev, int(hidden)))
            decoder_layers.append(nn.ReLU())
            if float(dropout) > 0:
                decoder_layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden)
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.head = nn.Linear(int(bottleneck_dim), 1)
        self.model = nn.Sequential(self.encoder, self.decoder)


def _maybe_noisy(torch: Any, x: Any, noise_std: float) -> Any:
    if float(noise_std) <= 0:
        return x
    return x + torch.randn_like(x) * float(noise_std)


def _forward_autoencoder(network: AutoEncoder, x: Any) -> tuple[Any, Any, Any]:
    latent = network.encoder(x)
    recon = network.decoder(latent)
    pred = network.head(latent).squeeze(-1)
    return latent, recon, pred


def _torch_pearson_loss(torch: Any, y_true: Any, y_pred: Any) -> Any:
    true_centered = y_true - torch.mean(y_true)
    pred_centered = y_pred - torch.mean(y_pred)
    denom = torch.sqrt(torch.sum(true_centered * true_centered) + 1e-12) * torch.sqrt(
        torch.sum(pred_centered * pred_centered) + 1e-12
    )
    return 1.0 - torch.sum(true_centered * pred_centered) / denom


def _standardize_target(y: np.ndarray) -> np.ndarray:
    y_float = y.astype(np.float32, copy=False)
    mean = float(np.mean(y_float))
    scale = float(np.std(y_float))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return ((y_float - mean) / scale).astype(np.float32, copy=False)


def train_autoencoder_early_stop(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[Any, Any, int, pd.DataFrame]:
    torch = setup_torch(cfg)
    seed_everything(torch, int(cfg.get("seed", 2026)))
    device = torch.device("cpu")

    valid_fraction = float(cfg.get("internal_validation_fraction", 0.1))
    split_at = int(len(x_train) * (1.0 - valid_fraction))
    split_at = min(max(split_at, 1), len(x_train) - 1)
    ae_train = x_train[:split_at]
    ae_valid = x_train[split_at:]

    target_loss_weight = float(cfg.get("target_loss_weight", 0.0))
    target_pearson_weight = float(cfg.get("target_pearson_weight", 0.0))
    use_target = target_loss_weight > 0.0 or target_pearson_weight > 0.0
    y_std = _standardize_target(y_train)
    y_ae_train = y_std[:split_at]
    y_ae_valid = y_std[split_at:]

    network = AutoEncoder(
        torch,
        input_dim=x_train.shape[1],
        hidden_sizes=[int(value) for value in cfg["hidden_sizes"]],
        bottleneck_dim=int(cfg["bottleneck_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    network.encoder.to(device)
    network.decoder.to(device)
    network.head.to(device)
    params = list(network.encoder.parameters()) + list(network.decoder.parameters())
    if use_target:
        params += list(network.head.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=float(cfg.get("learning_rate", 0.001)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    mse = torch.nn.MSELoss()
    if use_target:
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(ae_train.astype("float32", copy=False)),
            torch.from_numpy(y_ae_train.astype("float32", copy=False)),
        )
    else:
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(ae_train.astype("float32", copy=False)))
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 2026)))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4096)),
        shuffle=True,
        generator=generator,
    )
    valid_tensor = torch.from_numpy(ae_valid.astype("float32", copy=False)).to(device)
    valid_target_tensor = torch.from_numpy(y_ae_valid.astype("float32", copy=False)).to(device)
    noise_std = float(cfg.get("input_noise_std", 0.0))

    max_epochs = int(cfg.get("max_epochs", 55))
    patience_left = int(cfg.get("patience", 7))
    best_loss = np.inf
    best_epoch = 0
    best_state: dict[str, dict[str, Any]] | None = None
    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        network.encoder.train()
        network.decoder.train()
        network.head.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_target_sum = 0.0
        train_pearson_sum = 0.0
        train_count = 0
        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device) if use_target else None
            xb_input = _maybe_noisy(torch, xb, noise_std)
            optimizer.zero_grad(set_to_none=True)
            _, recon, target_pred = _forward_autoencoder(network, xb_input)
            recon_loss = mse(recon, xb)
            if use_target and yb is not None:
                target_loss = mse(target_pred, yb)
                pearson_loss = _torch_pearson_loss(torch, yb, target_pred)
            else:
                target_loss = recon_loss * 0.0
                pearson_loss = recon_loss * 0.0
            loss = recon_loss + target_loss_weight * target_loss + target_pearson_weight * pearson_loss
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(xb)
            train_recon_sum += float(recon_loss.detach().cpu()) * len(xb)
            train_target_sum += float(target_loss.detach().cpu()) * len(xb)
            train_pearson_sum += float(pearson_loss.detach().cpu()) * len(xb)
            train_count += len(xb)

        network.encoder.eval()
        network.decoder.eval()
        network.head.eval()
        with torch.no_grad():
            _, valid_recon, valid_target_pred = _forward_autoencoder(network, valid_tensor)
            valid_recon_loss = mse(valid_recon, valid_tensor)
            if use_target:
                valid_target_loss = mse(valid_target_pred, valid_target_tensor)
                valid_pearson_loss = _torch_pearson_loss(torch, valid_target_tensor, valid_target_pred)
            else:
                valid_target_loss = valid_recon_loss * 0.0
                valid_pearson_loss = valid_recon_loss * 0.0
            valid_loss_tensor = (
                valid_recon_loss
                + target_loss_weight * valid_target_loss
                + target_pearson_weight * valid_pearson_loss
            )
            valid_loss = float(valid_loss_tensor.detach().cpu())
        train_loss = train_loss_sum / max(train_count, 1)
        rows.append(
            {
                "epoch": epoch,
                "train_objective": train_loss,
                "train_recon_mse": train_recon_sum / max(train_count, 1),
                "train_target_mse": train_target_sum / max(train_count, 1),
                "train_target_pearson_loss": train_pearson_sum / max(train_count, 1),
                "valid_objective": valid_loss,
                "valid_recon_mse": float(valid_recon_loss.detach().cpu()),
                "valid_target_mse": float(valid_target_loss.detach().cpu()),
                "valid_target_pearson_loss": float(valid_pearson_loss.detach().cpu()),
            }
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                "encoder": {key: value.detach().cpu().clone() for key, value in network.encoder.state_dict().items()},
                "decoder": {key: value.detach().cpu().clone() for key, value in network.decoder.state_dict().items()},
                "head": {key: value.detach().cpu().clone() for key, value in network.head.state_dict().items()},
            }
            patience_left = int(cfg.get("patience", 7))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    elapsed = time.perf_counter() - start
    if best_state is not None:
        network.encoder.load_state_dict(best_state["encoder"])
        network.decoder.load_state_dict(best_state["decoder"])
        network.head.load_state_dict(best_state["head"])
    log = pd.DataFrame(rows)
    log["best_epoch"] = best_epoch
    log["train_seconds_total"] = elapsed
    return torch, network, best_epoch, log


def train_autoencoder_fixed_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
    epochs: int,
) -> tuple[Any, Any]:
    torch = setup_torch(cfg)
    seed_everything(torch, int(cfg.get("seed", 2026)))
    device = torch.device("cpu")
    target_loss_weight = float(cfg.get("target_loss_weight", 0.0))
    target_pearson_weight = float(cfg.get("target_pearson_weight", 0.0))
    use_target = target_loss_weight > 0.0 or target_pearson_weight > 0.0
    y_std = _standardize_target(y_train)
    network = AutoEncoder(
        torch,
        input_dim=x_train.shape[1],
        hidden_sizes=[int(value) for value in cfg["hidden_sizes"]],
        bottleneck_dim=int(cfg["bottleneck_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    network.encoder.to(device)
    network.decoder.to(device)
    network.head.to(device)
    params = list(network.encoder.parameters()) + list(network.decoder.parameters())
    if use_target:
        params += list(network.head.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=float(cfg.get("learning_rate", 0.001)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    mse = torch.nn.MSELoss()
    if use_target:
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_train.astype("float32", copy=False)),
            torch.from_numpy(y_std.astype("float32", copy=False)),
        )
    else:
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train.astype("float32", copy=False)))
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 2026)))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4096)),
        shuffle=True,
        generator=generator,
    )
    noise_std = float(cfg.get("input_noise_std", 0.0))
    for _ in range(max(1, int(epochs))):
        network.encoder.train()
        network.decoder.train()
        network.head.train()
        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device) if use_target else None
            xb_input = _maybe_noisy(torch, xb, noise_std)
            optimizer.zero_grad(set_to_none=True)
            _, recon, target_pred = _forward_autoencoder(network, xb_input)
            recon_loss = mse(recon, xb)
            if use_target and yb is not None:
                target_loss = mse(target_pred, yb)
                pearson_loss = _torch_pearson_loss(torch, yb, target_pred)
            else:
                target_loss = recon_loss * 0.0
                pearson_loss = recon_loss * 0.0
            loss = recon_loss + target_loss_weight * target_loss + target_pearson_weight * pearson_loss
            loss.backward()
            optimizer.step()
    return torch, network


def encode_autoencoder(torch: Any, network: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    device = torch.device("cpu")
    network.encoder.to(device)
    network.encoder.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            batch = torch.from_numpy(x[start : start + int(batch_size)].astype("float32", copy=False)).to(device)
            outputs.append(network.encoder(batch).cpu().numpy().astype(np.float32))
    return np.vstack(outputs)


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


def add_candidate(
    rows: list[dict[str, Any]],
    variant_name: str,
    alpha: float,
    signal_weight: float,
    signal_valid: np.ndarray,
    signal_test: np.ndarray,
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
    blend_valid = current_weight * current_best_valid + float(signal_weight) * signal_valid
    blend_test = current_weight * current_best_test + float(signal_weight) * signal_test
    candidate = f"blend_current_w{safe_name(current_weight)}_{variant_name}_ridge_a{safe_name(alpha)}_w{safe_name(signal_weight)}"
    submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
    path = submission_dir / f"submission_{candidate}.csv"
    submission.to_csv(path, index=False)
    holdout = pearson_corr(y_valid, blend_valid)
    rows.append(
        {
            "candidate": candidate,
            "variant": variant_name,
            "alpha": float(alpha),
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


def select_candidates(candidates: pd.DataFrame, max_selected: int) -> pd.DataFrame:
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty:
        return eligible
    selected_names: list[str] = []
    reasons: dict[str, str] = {}
    best = eligible.sort_values(["holdout_pearson", "submission_corr_current_best"], ascending=[False, True]).iloc[0]
    selected_names.append(str(best["candidate"]))
    reasons[str(best["candidate"])] = "holdout_best"
    near = eligible[
        eligible["holdout_pearson"] >= float(best["holdout_pearson"]) - 0.001
    ].sort_values(["submission_corr_current_best", "holdout_pearson"], ascending=[True, False])
    for _, row in near.iterrows():
        candidate = str(row["candidate"])
        if candidate not in selected_names:
            selected_names.append(candidate)
            reasons[candidate] = "lower_corr_near_best"
            break
    conservative = eligible.assign(weight_gap=(eligible["signal_weight"].astype(float) - 0.05).abs()).sort_values(
        ["weight_gap", "holdout_pearson"],
        ascending=[True, False],
    )
    for _, row in conservative.iterrows():
        candidate = str(row["candidate"])
        if candidate not in selected_names:
            selected_names.append(candidate)
            reasons[candidate] = "conservative_weight"
            break
    selected = eligible[eligible["candidate"].isin(selected_names)].copy()
    selected["selection_reason"] = selected["candidate"].map(reasons)
    return selected.set_index("candidate").loc[selected_names[: int(max_selected)]].reset_index()


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    beta4_cfg = current_cfg["beta4_base"]
    beta5_cfg = current_cfg["beta5_interaction"]
    input_cfg = config["ae_input"]
    defaults = config["autoencoder_defaults"]
    ridge_cfg = config["ridge"]
    blend_cfg = config["blend"]
    xgb_fallback_cfg = config["xgboost_fallback"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    latent_dir = ensure_dir(run_dir / "latent_features")
    log_dir = ensure_dir(run_dir / "ae_logs")
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
        raise ValueError(f"AE base feature pool contains unknown features: {invalid[:10]}")
    interaction_defs = definitions_from_frame(pd.read_csv(ROOT / input_cfg["interaction_definitions_path"]))
    write_selected_features(run_dir / "base_feature_pool.txt", base_features)
    write_selected_features(run_dir / "interaction_feature_names.txt", [definition.name for definition in interaction_defs])

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    print("Generating AE input matrices.")
    base_preprocessor = TabularPreprocessor(base_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_base_train = base_preprocessor.transform(train_part)
    x_base_valid = base_preprocessor.transform(valid_part)
    x_base_full = base_preprocessor.transform(train_df)
    x_base_test = base_preprocessor.transform(test_df)
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
    x_input_train_raw = np.hstack(matrices_train).astype(np.float32, copy=False)
    x_input_valid_raw = np.hstack(matrices_valid).astype(np.float32, copy=False)
    x_input_full_raw = np.hstack(matrices_full).astype(np.float32, copy=False)
    x_input_test_raw = np.hstack(matrices_test).astype(np.float32, copy=False)
    del matrices_train, matrices_valid, matrices_full, matrices_test, x_base_train, x_base_valid, x_base_full, x_base_test
    gc.collect()

    print("Reconstructing current-best validation/test predictions.")
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
        del shap_preprocessor
        gc.collect()
    beta4_valid = float(beta4_cfg["base_weight"]) * beta3_valid + float(beta4_cfg["signal_weight"]) * shap_valid
    interaction_valid = load_prediction_csv(ROOT / beta5_cfg["valid_prediction_path"], "prediction", expected_len=len(y_valid))
    if interaction_valid is None:
        interaction_valid = fit_ridge(200000.0, x_input_train_raw[:, len(base_features) :], y_train, x_input_valid_raw[:, len(base_features) :])
    current_best_valid = (1.0 - float(beta5_cfg["weight"])) * beta4_valid + float(beta5_cfg["weight"]) * interaction_valid
    current_best_test = load_prediction_csv(ROOT / current_cfg["test_submission_path"], prediction_col, expected_len=len(test_df))
    if current_best_test is None:
        raise FileNotFoundError(f"Could not load current-best submission: {current_cfg['test_submission_path']}")
    current_best_holdout = pearson_corr(y_valid, current_best_valid)

    ae_scaler = MatrixStandardizer()
    x_ae_train = ae_scaler.fit_transform(x_input_train_raw)
    x_ae_valid = ae_scaler.transform(x_input_valid_raw)
    full_scaler = MatrixStandardizer()
    x_ae_full = full_scaler.fit_transform(x_input_full_raw)
    x_ae_test = full_scaler.transform(x_input_test_raw)

    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for raw_variant in config["autoencoder_variants"]:
        variant = merged_variant(defaults, raw_variant)
        variant_name = str(variant["name"])
        print(f"\n=== AE variant {variant_name} ===")
        torch, network, best_epoch, ae_log = train_autoencoder_early_stop(x_ae_train, y_train, variant)
        ae_log.to_csv(log_dir / f"ae_training_log_{variant_name}.csv", index=False)
        z_train = encode_autoencoder(torch, network, x_ae_train, int(variant["batch_size"]))
        z_valid = encode_autoencoder(torch, network, x_ae_valid, int(variant["batch_size"]))
        del network
        gc.collect()

        torch_full, network_full = train_autoencoder_fixed_epochs(x_ae_full, y_full, variant, best_epoch)
        z_full = encode_autoencoder(torch_full, network_full, x_ae_full, int(variant["batch_size"]))
        z_test = encode_autoencoder(torch_full, network_full, x_ae_test, int(variant["batch_size"]))
        del network_full
        gc.collect()

        latent_cols = [f"{variant_name}_ae_{idx}" for idx in range(z_train.shape[1])]
        pd.DataFrame(z_valid, columns=latent_cols).to_csv(latent_dir / f"valid_latent_{variant_name}.csv", index=False)

        latent_scaler = MatrixStandardizer()
        z_train_scaled = latent_scaler.fit_transform(z_train)
        z_valid_scaled = latent_scaler.transform(z_valid)
        full_latent_scaler = MatrixStandardizer()
        z_full_scaled = full_latent_scaler.fit_transform(z_full)
        z_test_scaled = full_latent_scaler.transform(z_test)
        best, valid_pred, ridge_grid = train_ridge_search(
            z_train_scaled,
            y_train,
            z_valid_scaled,
            y_valid,
            [float(alpha) for alpha in config["ridge"]["alphas"]],
        )
        ridge_grid.to_csv(run_dir / f"ridge_alpha_search_{variant_name}.csv", index=False)
        test_pred = fit_ridge(float(best["alpha"]), z_full_scaled, y_full, z_test_scaled)
        save_valid_prediction(pred_dir / f"valid_{variant_name}_ridge.csv", y_valid, valid_pred)
        signal_submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        signal_submission.to_csv(signal_submission_dir / f"submission_{variant_name}_ridge.csv", index=False)

        signal_pearson = pearson_corr(y_valid, valid_pred)
        metric_rows.append(
            {
                "variant": variant_name,
                "bottleneck_dim": int(variant["bottleneck_dim"]),
                "hidden_sizes": json.dumps([int(value) for value in variant["hidden_sizes"]]),
                "input_noise_std": float(variant.get("input_noise_std", 0.0)),
                "target_loss_weight": float(variant.get("target_loss_weight", 0.0)),
                "target_pearson_weight": float(variant.get("target_pearson_weight", 0.0)),
                "seed": int(variant.get("seed", 2026)),
                "best_epoch": int(best_epoch),
                "best_recon_valid_mse": float(ae_log["valid_recon_mse"].min()),
                "alpha": float(best["alpha"]),
                "pearson": signal_pearson,
                "rmse": rmse(y_valid, valid_pred),
                "holdout_delta_current_best": signal_pearson - current_best_holdout,
                "valid_corr_current_best": pearson_corr(current_best_valid, valid_pred),
                "submission_corr_current_best": pearson_corr(current_best_test, test_pred),
            }
        )
        for signal_weight in [float(value) for value in config["blend"]["signal_weights"]]:
            add_candidate(
                candidate_rows,
                variant_name,
                float(best["alpha"]),
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
        del z_train, z_valid, z_full, z_test, z_train_scaled, z_valid_scaled, z_full_scaled, z_test_scaled
        gc.collect()

    metrics = pd.DataFrame(metric_rows).sort_values(["pearson", "submission_corr_current_best"], ascending=[False, True])
    metrics.to_csv(run_dir / "signal_metrics.csv", index=False)
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
    selected = select_candidates(candidates, int(blend_cfg["max_selected_submissions"]))
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    if not selected.empty:
        shutil.copyfile(ROOT / str(selected.iloc[0]["path"]), submission_dir / "submission_best.csv")

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best": current_cfg,
            "current_best_holdout_pearson": current_best_holdout,
            "ae_input_feature_count": x_input_train_raw.shape[1],
            "variant_count": len(config["autoencoder_variants"]),
            "best_signal": metrics.iloc[0].to_dict(),
            "selected_candidates": selected.to_dict(orient="records"),
        },
    )

    print("\n=== AE Variant Signal Metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== Selected Candidates ===")
    if selected.empty:
        print("No eligible candidates selected for submission.")
    else:
        print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
