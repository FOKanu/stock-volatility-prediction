#!/usr/bin/env python3
"""Volatility regression preprocessing pipeline for AAPL sequence models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_CONFIG = {
    "profile": "full",
    "asset_type": "stock",
    "symbols_v1": ["AAPL"],
    "min_trading_days": 756,
    "max_zero_volume_pct": 0.20,
    "financial_status_normal_only": True,
    "price_col": "Adj Close",
    "target_type": "abs_return",
    "target_horizon_days": 1,
    "feature_cols": [
        "log_return",
        "abs_return",
        "squared_return",
        "intraday_range",
        "range_ratio",
        "volatility_5",
        "volatility_21",
        "log_volume",
        "volume_surge",
        "spy_volatility_21",
    ],
    "lookback_days": 60,
    "warmup_days": 21,
    "train_end": "2017-12-31",
    "val_start": "2018-01-01",
    "val_end": "2019-12-31",
    "test_start": "2020-01-01",
    "test_end": "2020-04-01",
    "random_seed": 42,
}

PIPELINE_CONFIG_MINIMAL = {
    **PIPELINE_CONFIG,
    "profile": "minimal",
    "feature_cols": [
        "log_return",
        "abs_return",
        "volatility_21",
        "intraday_range",
        "log_volume",
    ],
}

PIPELINE_CONFIG_PLUS5 = {
    **PIPELINE_CONFIG,
    "profile": "plus5",
    "feature_cols": [
        *PIPELINE_CONFIG["feature_cols"],
        "return_ma_21",
        "rsi_14",
        "volatility_ratio_5_21",
        "lag_abs_return_1",
        "spy_log_return",
    ],
}

PROFILES = {
    "full": PIPELINE_CONFIG,
    "minimal": PIPELINE_CONFIG_MINIMAL,
    "plus5": PIPELINE_CONFIG_PLUS5,
}


def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    if Path.cwd().name == "scripts":
        candidates.insert(0, Path.cwd().parent)
    for path in candidates:
        if (path / "Stock Market Dataset").exists():
            return path
    raise FileNotFoundError("Could not locate project root (Stock Market Dataset/ missing)")


def load_price_data(symbol: str, stocks_dir: Path, etfs_dir: Path, asset_type: str = "stock") -> pd.DataFrame:
    folder = stocks_dir if asset_type == "stock" else etfs_dir
    path = folder / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> dict:
    ohlcv = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    null_count = int(df[ohlcv].isnull().sum().sum())
    dup_dates = int(df["Date"].duplicated().sum())
    zero_vol = int((df["Volume"] == 0).sum())
    if dup_dates > 0:
        raise ValueError(f"{symbol}: {dup_dates} duplicate dates")
    return {
        "symbol": symbol,
        "rows": len(df),
        "null_count": null_count,
        "zero_volume_days": zero_vol,
        "zero_volume_pct": zero_vol / max(len(df), 1),
        "start_date": df["Date"].min(),
        "end_date": df["Date"].max(),
    }


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    ohlcv = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    out = df.dropna(subset=ohlcv).copy()
    return out.sort_values("Date").reset_index(drop=True)


def engineer_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out[price_col] / out[price_col].shift(1))
    out["abs_return"] = out["log_return"].abs()
    out["squared_return"] = out["log_return"] ** 2
    out["log_volume"] = np.log1p(out["Volume"])
    out["intraday_range"] = (out["High"] - out["Low"]) / out["Close"]
    range_ma_21 = out["intraday_range"].rolling(21).mean()
    out["range_ratio"] = out["intraday_range"] / range_ma_21
    out["volatility_5"] = out["log_return"].rolling(5).std()
    out["volatility_21"] = out["log_return"].rolling(21).std()
    out["return_ma_21"] = out["log_return"].rolling(21).mean()

    delta = out["log_return"].diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    out["volatility_ratio_5_21"] = out["volatility_5"] / out["volatility_21"].replace(0.0, np.nan)
    out["lag_abs_return_1"] = out["abs_return"].shift(1)

    volume_ma_21 = out["log_volume"].rolling(21).mean()
    out["volume_surge"] = out["log_volume"] - volume_ma_21
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def load_spy_vol_features(stocks_dir: Path, etfs_dir: Path, price_col: str = "Adj Close") -> pd.DataFrame:
    spy = clean_ohlcv(load_price_data("SPY", stocks_dir, etfs_dir, asset_type="etf"))
    spy = engineer_features(spy, price_col=price_col)
    return spy[["Date", "volatility_21", "log_return"]].rename(
        columns={"volatility_21": "spy_volatility_21", "log_return": "spy_log_return"}
    )


def merge_spy_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(spy_df, on="Date", how="inner")


def forward_realized_vol(log_returns: pd.Series, horizon: int) -> pd.Series:
    """Std of the next `horizon` daily log returns (exclusive of same-day return)."""
    arr = log_returns.to_numpy(dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr) - horizon):
        window = arr[i + 1 : i + 1 + horizon]
        if np.isnan(window).any():
            continue
        out[i] = float(np.std(window))
    return pd.Series(out, index=log_returns.index)


def forward_mean_return(log_returns: pd.Series, horizon: int) -> pd.Series:
    """Mean of the next `horizon` daily log returns (exclusive of same-day return)."""
    arr = log_returns.to_numpy(dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr) - horizon):
        window = arr[i + 1 : i + 1 + horizon]
        if np.isnan(window).any():
            continue
        out[i] = float(np.mean(window))
    return pd.Series(out, index=log_returns.index)


def add_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    horizon = int(cfg["target_horizon_days"])
    target_type = cfg["target_type"]

    if target_type == "abs_return":
        out["target"] = out["log_return"].shift(-horizon).abs()
        out["forward_vol"] = np.nan
        return out

    if target_type == "vol_regime_21":
        out["forward_vol"] = forward_realized_vol(out["log_return"], horizon)
        out["target"] = np.nan
        return out

    if target_type == "vol_regime3_21":
        out["forward_vol"] = forward_realized_vol(out["log_return"], horizon)
        out["forward_ret"] = forward_mean_return(out["log_return"], horizon)
        out["target"] = np.nan
        return out

    raise ValueError(f"Unsupported target_type: {target_type!r}")


def apply_regime_threshold(model_df: pd.DataFrame, train_end: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """Binarize forward_vol into high/low regime using the train-split median."""
    train_mask = model_df["Date"] <= train_end
    threshold = float(model_df.loc[train_mask, "forward_vol"].median())
    out = model_df.copy()
    out["target"] = (out["forward_vol"] >= threshold).astype(np.float32)
    return out, threshold


def apply_regime3_thresholds(
    model_df: pd.DataFrame, train_end: pd.Timestamp, crisis_quantile: float = 0.90
) -> tuple[pd.DataFrame, dict]:
    """Three latent states from forward vol + forward mean return (train thresholds).

    0 = bull   (non-crisis vol, non-negative forward mean return)
    1 = bear   (non-crisis vol, negative forward mean return)
    2 = crisis (forward vol >= train `crisis_quantile`, regardless of return sign)
    """
    train_mask = model_df["Date"] <= train_end
    crisis_threshold = float(model_df.loc[train_mask, "forward_vol"].quantile(crisis_quantile))
    out = model_df.copy()
    crisis = out["forward_vol"] >= crisis_threshold
    bear = (~crisis) & (out["forward_ret"] < 0)
    out["target"] = np.where(crisis, 2.0, np.where(bear, 1.0, 0.0)).astype(np.float32)

    train_targets = out.loc[train_mask, "target"]
    info = {
        "crisis_quantile": crisis_quantile,
        "crisis_threshold": crisis_threshold,
        "train_class_rates": {
            "bull": float((train_targets == 0).mean()),
            "bear": float((train_targets == 1).mean()),
            "crisis": float((train_targets == 2).mean()),
        },
    }
    return out, info


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    lookback: int = 60,
    warmup: int = 21,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    dates = df["Date"]

    start_idx = max(lookback - 1, warmup)
    X_list, y_list, date_list = [], [], []

    for i in range(start_idx, len(df)):
        window = values[i - lookback + 1 : i + 1]
        if window.shape[0] != lookback or np.isnan(window).any():
            continue
        if np.isnan(targets[i]):
            continue
        X_list.append(window)
        y_list.append(targets[i])
        date_list.append(dates.iloc[i])

    return np.array(X_list), np.array(y_list), pd.DatetimeIndex(date_list)


def fit_transform_zscore(X_train, X_other_list):
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std == 0] = 1.0

    def transform(X):
        return (X - mean) / std

    return transform(X_train), [transform(x) for x in X_other_list], mean, std


def artifact_paths(
    processed_dir: Path, symbol: str, profile: str = "full", target_type: str = "abs_return"
) -> dict[str, Path]:
    sym = symbol.lower()
    suffix = "" if profile == "full" else f"_{profile}"
    target_suffix = "" if target_type == "abs_return" else f"_{target_type}"
    stem = f"{sym}{target_suffix}"
    if sym == "aapl":
        config_name = f"pipeline_config{target_suffix}{suffix}.json"
        scaler_name = f"scaler{target_suffix}{suffix}.json"
    else:
        config_name = f"pipeline_config_{stem}{suffix}.json"
        scaler_name = f"scaler_{stem}{suffix}.json"
    return {
        "sequences": processed_dir / f"{stem}_sequences{suffix}.npz",
        "config": processed_dir / config_name,
        "scaler": processed_dir / scaler_name,
        "model_frame": processed_dir / f"{stem}_model_frame{suffix}.csv",
    }


def select_eligible_symbols(
    cfg: dict,
    stocks_dir: Path,
    etfs_dir: Path,
    meta_path: Path,
    symbol_scope: str = "configured",
) -> list[str]:
    meta = pd.read_csv(meta_path)
    if cfg["asset_type"] == "stock":
        meta = meta[meta["ETF"] == "N"].copy()
    else:
        meta = meta[meta["ETF"] == "Y"].copy()
    if cfg["financial_status_normal_only"]:
        meta = meta[meta["Financial Status"] == "N"].copy()

    folder = stocks_dir if cfg["asset_type"] == "stock" else etfs_dir
    if symbol_scope == "all":
        requested = sorted(str(s) for s in meta["Symbol"].dropna().unique())
    else:
        requested = cfg["symbols_v1"]

    eligible = [s for s in requested if (folder / f"{s}.csv").exists()]
    if symbol_scope == "configured":
        missing = set(requested) - set(eligible)
        if missing:
            raise FileNotFoundError(f"Configured symbols missing CSV files: {missing}")
    if not eligible:
        raise RuntimeError("No eligible symbols found after metadata/file checks.")
    return eligible


def load_and_clean_symbols(
    cfg: dict,
    eligible_symbols: list[str],
    stocks_dir: Path,
    etfs_dir: Path,
    strict: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    cleaned_frames: dict[str, pd.DataFrame] = {}
    symbol_metrics: dict[str, dict] = {}
    skipped: list[str] = []
    for symbol in eligible_symbols:
        raw = load_price_data(symbol, stocks_dir, etfs_dir, asset_type=cfg["asset_type"])
        metrics = validate_ohlcv(raw, symbol)
        if metrics["rows"] < cfg["min_trading_days"]:
            msg = f"{symbol}: only {metrics['rows']} rows < min {cfg['min_trading_days']}"
            if strict:
                raise ValueError(msg)
            skipped.append(msg)
            continue
        if metrics["zero_volume_pct"] > cfg["max_zero_volume_pct"]:
            msg = (
                f"{symbol}: zero-volume pct {metrics['zero_volume_pct']:.1%} "
                f"> threshold {cfg['max_zero_volume_pct']:.0%}"
            )
            if strict:
                raise ValueError(msg)
            skipped.append(msg)
            continue
        cleaned = clean_ohlcv(raw)
        cleaned["Symbol"] = symbol
        cleaned_frames[symbol] = cleaned
        symbol_metrics[symbol] = metrics
    if not cleaned_frames:
        raise RuntimeError("No symbols passed quality filters.")
    if skipped:
        print(f"Skipped {len(skipped)} symbols failing quality filters.")
    return cleaned_frames, symbol_metrics


def filter_history_outliers(
    cleaned_frames: dict[str, pd.DataFrame],
    method: str = "none",
) -> tuple[dict[str, pd.DataFrame], dict]:
    lengths = {s: len(df) for s, df in cleaned_frames.items()}
    if method == "none" or len(lengths) < 4:
        return cleaned_frames, {"method": method, "removed": [], "upper_bound": None}

    vals = np.array(list(lengths.values()), dtype=float)
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    kept = {s: df for s, df in cleaned_frames.items() if len(df) <= upper}
    removed = sorted(s for s in cleaned_frames if s not in kept)
    if not kept:
        raise RuntimeError("History outlier filter removed all symbols.")
    return kept, {"method": method, "removed": removed, "upper_bound": float(upper)}


def run_pipeline(
    cfg: dict,
    cleaned_frames: dict[str, pd.DataFrame],
    eligible_symbols: list[str],
    processed_dir: Path,
    stocks_dir: Path,
    etfs_dir: Path,
) -> dict:
    feature_cols = cfg["feature_cols"]
    spy_feat = load_spy_vol_features(stocks_dir, etfs_dir, cfg["price_col"])
    x_chunks = {"train": [], "val": [], "test": []}
    y_chunks = {"train": [], "val": [], "test": []}
    model_frames: list[pd.DataFrame] = []
    persistence_abs_err: list[np.ndarray] = []

    train_end = pd.Timestamp(cfg["train_end"])
    val_start = pd.Timestamp(cfg["val_start"])
    val_end = pd.Timestamp(cfg["val_end"])
    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"])

    for symbol in eligible_symbols:
        feat = engineer_features(cleaned_frames[symbol], price_col=cfg["price_col"])
        feat = merge_spy_features(feat, spy_feat)
        feat = add_target(feat, cfg)
        drop_cols = feature_cols + ["Date", "Symbol"]
        if cfg["target_type"] == "vol_regime_21":
            drop_cols.append("forward_vol")
        elif cfg["target_type"] == "vol_regime3_21":
            drop_cols.extend(["forward_vol", "forward_ret"])
        else:
            drop_cols.append("target")
        model_df = feat.dropna(subset=drop_cols).reset_index(drop=True)
        if model_df.empty:
            continue
        model_frames.append(model_df)

    if not model_frames:
        raise RuntimeError("No symbols produced usable model frames.")

    model_df = pd.concat(model_frames, ignore_index=True)
    regime_threshold = None
    regime3_info = None
    if cfg["target_type"] == "vol_regime_21":
        model_df, regime_threshold = apply_regime_threshold(model_df, train_end)
        model_frames = [
            model_df[model_df["Symbol"] == symbol].reset_index(drop=True)
            for symbol in eligible_symbols
            if symbol in model_df["Symbol"].values
        ]
    elif cfg["target_type"] == "vol_regime3_21":
        model_df, regime3_info = apply_regime3_thresholds(model_df, train_end)
        model_frames = [
            model_df[model_df["Symbol"] == symbol].reset_index(drop=True)
            for symbol in eligible_symbols
            if symbol in model_df["Symbol"].values
        ]

    persistence_abs_err: list[np.ndarray] = []
    for symbol_df in model_frames:
        if symbol_df.empty:
            continue
        X_sym, y_sym, seq_dates = build_sequences(
            symbol_df,
            feature_cols=feature_cols,
            lookback=cfg["lookback_days"],
            warmup=cfg["warmup_days"],
        )
        if len(y_sym) == 0:
            continue

        if cfg["target_type"] == "abs_return":
            abs_idx = feature_cols.index("abs_return")
            persistence_abs_err.append(np.abs(X_sym[:, -1, abs_idx] - y_sym))
        splits = {
            "train": seq_dates <= train_end,
            "val": (seq_dates >= val_start) & (seq_dates <= val_end),
            "test": (seq_dates >= test_start) & (seq_dates <= test_end),
        }
        for split_name, mask in splits.items():
            if mask.any():
                x_chunks[split_name].append(X_sym[mask])
                y_chunks[split_name].append(y_sym[mask])

    if not any(y_chunks[k] for k in y_chunks):
        raise RuntimeError("No symbols produced usable sequences.")

    model_df = pd.concat(model_frames, ignore_index=True)
    train_mask = model_df["Date"] <= train_end
    export_cfg = {
        **cfg,
        "n_symbols": len(eligible_symbols),
        "symbols": eligible_symbols,
        "train_target_mean": float(model_df.loc[train_mask, "target"].mean()),
        "train_target_median": float(model_df.loc[train_mask, "target"].median()),
    }
    if regime_threshold is not None:
        export_cfg["regime_threshold"] = regime_threshold
        export_cfg["train_high_regime_rate"] = float(model_df.loc[train_mask, "target"].mean())
    if regime3_info is not None:
        export_cfg["regime3"] = regime3_info

    profile = cfg.get("profile", "full")
    print(f"\n=== Profile: {profile} ({len(feature_cols)} features) ===")
    print(f"Model frame shape: {model_df.shape}")
    print(f"Date range: {model_df['Date'].min().date()} → {model_df['Date'].max().date()}")
    if cfg["target_type"] == "abs_return":
        print(f"Target mean |log return| (all): {model_df['target'].mean():.6f}")
        print(f"Target median (train): {export_cfg['train_target_median']:.6f}")
    elif cfg["target_type"] == "vol_regime3_21":
        rates = regime3_info["train_class_rates"]
        print(
            f"Train class rates: bull={rates['bull']:.3f} bear={rates['bear']:.3f} "
            f"crisis={rates['crisis']:.3f}"
        )
        print(f"Crisis threshold (train q{regime3_info['crisis_quantile']:.2f} fwd vol): "
              f"{regime3_info['crisis_threshold']:.6f}")
    else:
        print(f"Target high-regime rate (all): {model_df['target'].mean():.3f}")
        print(f"Target high-regime rate (train): {export_cfg['train_target_mean']:.3f}")
        print(f"Regime threshold (train median fwd vol): {regime_threshold:.6f}")
    print(f"Symbols used: {len(eligible_symbols)}")

    def concat_split(chunks: list[np.ndarray], is_x: bool) -> np.ndarray:
        if chunks:
            return np.concatenate(chunks, axis=0)
        if is_x:
            return np.empty((0, cfg["lookback_days"], len(feature_cols)), dtype=np.float32)
        return np.empty((0,), dtype=np.float32)

    X = {k: concat_split(v, is_x=True) for k, v in x_chunks.items()}
    y = {k: concat_split(v, is_x=False) for k, v in y_chunks.items()}

    print("Split summary:")
    for split_name in ("train", "val", "test"):
        n = y[split_name].shape[0]
        t_mean = float(np.mean(y[split_name])) if n > 0 else float("nan")
        print(
            f"  {split_name:5s}: n={n:6d}  "
            f"target_mean={t_mean:.6f}"
        )

    X_train_norm, [X_val_norm, X_test_norm], feat_mean, feat_std = fit_transform_zscore(
        X["train"], [X["val"], X["test"]]
    )
    X_norm = {"train": X_train_norm, "val": X_val_norm, "test": X_test_norm}
    scaler = {"mean": feat_mean.tolist(), "std": feat_std.tolist(), "feature_cols": feature_cols}

    dataset_id = (
        eligible_symbols[0]
        if len(eligible_symbols) == 1
        else ("stocks_universe" if cfg["asset_type"] == "stock" else "etfs_universe")
    )
    paths = artifact_paths(processed_dir, dataset_id, profile=profile, target_type=cfg["target_type"])
    np.savez_compressed(
        paths["sequences"],
        X_train=X_norm["train"],
        y_train=y["train"],
        X_val=X_norm["val"],
        y_val=y["val"],
        X_test=X_norm["test"],
        y_test=y["test"],
        feature_cols=np.array(feature_cols),
    )
    with open(paths["config"], "w") as f:
        json.dump(export_cfg, f, indent=2)
    with open(paths["scaler"], "w") as f:
        json.dump(scaler, f, indent=2)
    model_df.to_csv(paths["model_frame"], index=False)

    print("Exported:")
    for path in paths.values():
        print(f"  {path}")

    persistence_mae = (
        float(np.mean(np.concatenate(persistence_abs_err))) if persistence_abs_err else float("nan")
    )
    if cfg["target_type"] == "abs_return":
        print(f"Persistence baseline MAE (today |r| → tomorrow |r|): {persistence_mae:.6f}")

    return {
        "cfg": export_cfg,
        "feature_cols": feature_cols,
        "model_df": model_df,
        "X_norm": X_norm,
        "y": y,
        "paths": paths,
        "persistence_mae": persistence_mae,
    }


def validate_pipeline_result(result: dict) -> None:
    cfg = result["cfg"]
    feature_cols = result["feature_cols"]
    X_norm = result["X_norm"]
    y = result["y"]
    train_end = pd.Timestamp(cfg["train_end"])
    val_start = pd.Timestamp(cfg["val_start"])
    val_end = pd.Timestamp(cfg["val_end"])
    test_start = pd.Timestamp(cfg["test_start"])
    target_type = cfg["target_type"]

    checks = {
        "train_samples > 0": X_norm["train"].shape[0] > 0,
        "val_samples > 0": X_norm["val"].shape[0] > 0,
        "test_samples > 0": X_norm["test"].shape[0] > 0,
        "no_nan_in_train_X": not np.isnan(X_norm["train"]).any(),
        "no_nan_in_train_y": not np.isnan(y["train"]).any(),
        "lookback_matches_config": X_norm["train"].shape[1] == cfg["lookback_days"],
        "n_features_matches_config": X_norm["train"].shape[2] == len(feature_cols),
        "train_end_before_val_start": train_end < val_start,
        "val_end_before_test_start": val_end < test_start,
        "train_target_mean_documented": "train_target_mean" in cfg,
    }

    if target_type == "abs_return":
        checks["target_type_is_abs_return"] = True
        checks["y_is_continuous"] = y["train"].max() > 0.05 and len(np.unique(y["train"])) > 10
    elif target_type == "vol_regime_21":
        checks["target_type_is_vol_regime"] = True
        checks["y_is_binary"] = set(np.unique(y["train"]).tolist()).issubset({0.0, 1.0})
        checks["regime_threshold_documented"] = "regime_threshold" in cfg
    elif target_type == "vol_regime3_21":
        checks["target_type_is_vol_regime3"] = True
        checks["y_is_ternary"] = set(np.unique(y["train"]).tolist()).issubset({0.0, 1.0, 2.0})
        checks["regime3_thresholds_documented"] = "regime3" in cfg
    else:
        raise ValueError(f"Unsupported target_type for validation: {target_type!r}")

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Pipeline validation failed: {failed}")
    print("All pipeline checks passed.")


TARGET_HORIZON_BY_TYPE = {
    "abs_return": 1,
    "vol_regime_21": 21,
    "vol_regime3_21": 21,
}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Vol regression preprocessing")
    parser.add_argument(
        "--profile",
        choices=[*PROFILES.keys(), "all"],
        default="full",
        help="Preprocessing profile (default: full)",
    )
    parser.add_argument(
        "--symbol-scope",
        choices=["configured", "all"],
        default="configured",
        help="Use configured symbols_v1 or all eligible symbols from metadata",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Optional cap for number of eligible symbols (after filters)",
    )
    parser.add_argument(
        "--history-outlier-filter",
        choices=["none", "iqr_upper"],
        default="none",
        help="Filter symbols with unusually long histories (upper-tail outliers)",
    )
    parser.add_argument(
        "--asset-type",
        choices=["stock", "etf"],
        default="stock",
        help="Universe asset type (stock or etf metadata filter)",
    )
    parser.add_argument(
        "--target-type",
        choices=["abs_return", "vol_regime_21", "vol_regime3_21"],
        default="abs_return",
        help="Label to export in sequence bundles",
    )
    parser.add_argument(
        "--symbols-v1",
        nargs="+",
        default=None,
        help="Override configured symbols_v1 (e.g. SPY for market lane)",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    data_dir = project_root / "Stock Market Dataset"
    stocks_dir = data_dir / "stocks"
    etfs_dir = data_dir / "etfs"
    meta_path = data_dir / "symbols_valid_meta.csv"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root:  {project_root}")
    print(f"Processed dir: {processed_dir}")

    base_cfg = {
        **PIPELINE_CONFIG,
        "asset_type": args.asset_type,
        "target_type": args.target_type,
        "target_horizon_days": TARGET_HORIZON_BY_TYPE[args.target_type],
    }
    if args.symbols_v1:
        base_cfg["symbols_v1"] = args.symbols_v1
    symbol_scope = args.symbol_scope
    if args.symbols_v1:
        symbol_scope = "configured"
    eligible_symbols = select_eligible_symbols(
        base_cfg,
        stocks_dir,
        etfs_dir,
        meta_path,
        symbol_scope=symbol_scope,
    )
    if args.max_symbols is not None:
        eligible_symbols = eligible_symbols[: args.max_symbols]
    eligible_path = processed_dir / "eligible_symbols.json"
    print(f"Eligible symbols (pre-filter): {len(eligible_symbols)}")

    strict_filters = symbol_scope == "configured"
    cleaned_frames, _metrics = load_and_clean_symbols(
        base_cfg,
        eligible_symbols,
        stocks_dir,
        etfs_dir,
        strict=strict_filters,
    )
    cleaned_frames, outlier_info = filter_history_outliers(
        cleaned_frames, method=args.history_outlier_filter
    )
    eligible_symbols = [s for s in eligible_symbols if s in cleaned_frames]
    with open(eligible_path, "w") as f:
        json.dump(eligible_symbols, f, indent=2)
    print(f"Eligible symbols (final): {len(eligible_symbols)}")
    if outlier_info["removed"]:
        print(
            f"History outlier filter removed {len(outlier_info['removed'])} symbols "
            f"(upper_bound={outlier_info['upper_bound']:.1f} rows)"
        )

    profiles = list(PROFILES) if args.profile == "all" else [args.profile]
    for profile_name in profiles:
        cfg = {
            **PROFILES[profile_name],
            "asset_type": args.asset_type,
            "target_type": args.target_type,
            "target_horizon_days": TARGET_HORIZON_BY_TYPE[args.target_type],
            "symbol_scope": symbol_scope,
            "history_outlier_filter": args.history_outlier_filter,
        }
        if args.symbols_v1:
            cfg["symbols_v1"] = args.symbols_v1
        result = run_pipeline(
            cfg,
            cleaned_frames,
            eligible_symbols,
            processed_dir,
            stocks_dir,
            etfs_dir,
        )
        validate_pipeline_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
