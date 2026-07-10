#!/usr/bin/env python3
"""Train J01 vol_risk candidates with the QLIKE objective (parallel artifacts).

Saves predictions as ``{arch}_qlike_{job}_{universe}_{tag}_tf_predictions.json``
so results appear next to the Huber-trained runs in compare_job_models.py
without touching the existing notebooks or artifacts.

Usage:
    python scripts/train_qlike_variants.py --universe stocks --architectures gru garchgru
"""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _reexec_with_cuda_libs() -> None:
    """Re-exec with LD_LIBRARY_PATH pointing at pip-installed NVIDIA libs.

    The dynamic linker reads LD_LIBRARY_PATH at process start, so setting it
    after Python launches (as bootstrap_notebook does) is too late for
    TensorFlow's dlopen of CUDA libraries. Jupyter kernels get the path from
    kernel.json; plain script runs need this re-exec.
    """
    if os.environ.get("_CUDA_LIBS_REEXECED") or os.name == "nt":
        return
    nvidia_root = Path(site.getsitepackages()[0]) / "nvidia"
    if not nvidia_root.exists():
        return
    lib_dirs = sorted({str(so.parent.resolve()) for so in nvidia_root.glob("*/lib/*.so*")})
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in existing for d in lib_dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    os.environ["_CUDA_LIBS_REEXECED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


_reexec_with_cuda_libs()


def train_one(architecture: str, universe: str, dataset_tag: str, epochs: int, patience: int) -> dict:
    from tf_data import load_sequence_bundle
    from tf_notebook_utils import (
        as_1d_predictions,
        asymmetric_error_metrics,
        bootstrap_notebook,
        build_regressor,
        configure_tf_gpu,
        make_tf_datasets,
        persistence_mae,
    )

    root, models_dir, _, tf = bootstrap_notebook(42)
    device_info = configure_tf_gpu(tf, enable_mixed_precision=True)

    prefix = "etfs_universe" if universe == "etfs" else "stocks_universe"
    stem = f"{prefix}_{dataset_tag}" if dataset_tag != "full" else prefix
    bundle = load_sequence_bundle(sequences_name=f"{stem}_sequences_plus5.npz")

    X_train = bundle.X_train.astype(np.float32)
    y_train = bundle.y_train.astype(np.float32)
    X_val = bundle.X_val.astype(np.float32)
    y_val = bundle.y_val.astype(np.float32)
    X_test = bundle.X_test.astype(np.float32)
    y_test = bundle.y_test.astype(np.float32)

    batch_size = 512 if device_info["using_gpu"] else 1024
    lr = 1e-3
    arch_kwargs = {}
    if architecture == "garchgru":
        arch_kwargs["return_idx"] = bundle.feature_cols.index("log_return")

    train_ds, val_ds = make_tf_datasets(X_train, y_train, X_val, y_val, batch_size=batch_size)
    model = build_regressor(
        architecture, X_train.shape[1], X_train.shape[-1], lr=lr, loss="qlike", **arch_kwargs
    )

    print(
        f"[{architecture}/{universe}] training with QLIKE | "
        f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,} | "
        f"gpu={device_info['using_gpu']} epochs<={epochs} patience={patience}"
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", mode="min", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2
    )

    val_pred = as_1d_predictions(model.predict(X_val, batch_size=batch_size, verbose=0))
    test_pred = as_1d_predictions(model.predict(X_test, batch_size=batch_size, verbose=0))

    val_mae = float(np.mean(np.abs(val_pred - y_val)))
    val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    test_mae = float(np.mean(np.abs(test_pred - y_test)))
    test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))
    asym_val = asymmetric_error_metrics(y_val, val_pred)
    asym_crisis = asymmetric_error_metrics(y_test, test_pred)

    persistence_mae_val = persistence_mae(X_val, y_val, bundle.feature_cols, bundle.scaler)
    persistence_mae_test = persistence_mae(X_test, y_test, bundle.feature_cols, bundle.scaler)
    train_mean = float(np.mean(y_train))
    train_mean_mae_val = float(np.mean(np.abs(train_mean - y_val)))
    train_mean_mae_test = float(np.mean(np.abs(train_mean - y_test)))

    arch_tag = f"{architecture}_qlike"
    model_tag = f"_{dataset_tag}" if dataset_tag != "full" else ""
    artifact_stem = f"{arch_tag}_vol_risk_{universe}{model_tag}_tf"

    baseline_path = models_dir / f"baseline_vol_risk_{universe}{model_tag}_tf_predictions.json"
    baseline_comparison = {"baseline_trained": False}
    if baseline_path.exists():
        with open(baseline_path) as f:
            b = json.load(f)
        baseline_comparison = {
            "baseline_trained": True,
            "baseline_artifact": baseline_path.name,
            "baseline_val_mae": b.get("val_mae"),
            "baseline_crisis_mae": b.get("crisis_mae"),
            "beats_baseline_val_mae": val_mae < b.get("val_mae", float("inf")),
            "beats_baseline_crisis_mae": test_mae < b.get("crisis_mae", float("inf")),
            "delta_val_mae": float(val_mae - b.get("val_mae", val_mae)),
            "delta_crisis_mae": float(test_mae - b.get("crisis_mae", test_mae)),
        }

    best_model_path = models_dir / f"{artifact_stem}_best.keras"
    model.save(best_model_path)

    hist = history.history
    payload = {
        "job_id": "01",
        "job_slug": "vol_risk",
        "architecture": arch_tag,
        "universe": universe,
        "model": f"{artifact_stem}_tensorflow",
        "run_mode": "fast_dev" if dataset_tag != "full" else "full",
        "dataset_tag": dataset_tag,
        "n_symbols": bundle.cfg.get("n_symbols"),
        "framework": "tensorflow",
        "target_type": bundle.cfg["target_type"],
        "n_features": len(bundle.feature_cols),
        "lookback_days": bundle.cfg["lookback_days"],
        "feature_cols": bundle.feature_cols,
        "best_epoch": int(np.argmin(hist["val_mae"]) + 1),
        "best_val_mae": float(np.min(hist["val_mae"])),
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "crisis_mae": test_mae,
        "test_rmse": test_rmse,
        "asymmetric_errors": {"val": asym_val, "crisis": asym_crisis},
        "baselines": {
            "persistence_mae_val": persistence_mae_val,
            "persistence_mae_test": persistence_mae_test,
            "train_mean_mae_val": train_mean_mae_val,
            "train_mean_mae_test": train_mean_mae_test,
        },
        "baseline_comparison": baseline_comparison,
        "train_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "loss": "qlike",
            "architecture": architecture,
            "device": device_info,
        },
        "history": [
            {
                "epoch": i + 1,
                "train_loss": float(hist["loss"][i]),
                "val_loss": float(hist["val_loss"][i]),
                "val_mae": float(hist["val_mae"][i]),
            }
            for i in range(len(hist["loss"]))
        ],
        "predictions": {
            "val": {"y_true": y_val.tolist(), "y_pred": val_pred.tolist()},
            "test": {"y_true": y_test.tolist(), "y_pred": test_pred.tolist()},
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    pred_path = models_dir / f"{artifact_stem}_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[{architecture}/{universe}] QLIKE done | val MAE {val_mae:.6f} | "
        f"crisis MAE {test_mae:.6f} | crisis under-MAE {asym_crisis['under_mae']:.6f} "
        f"(rate {asym_crisis['under_rate']:.1%})"
    )
    print(f"Saved {pred_path.name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", nargs="+", default=["stocks", "etfs"])
    parser.add_argument("--architectures", nargs="+", default=["gru", "garchgru"])
    parser.add_argument("--dataset-tag", default="dev200")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    for universe in args.universe:
        for arch in args.architectures:
            train_one(arch, universe, args.dataset_tag, args.epochs, args.patience)


if __name__ == "__main__":
    main()
