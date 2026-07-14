#!/usr/bin/env python3
"""Train J01 vol_risk candidates with the QLIKE objective (parallel artifacts).

Saves predictions as ``{arch}_qlike_{job}_{universe}_{tag}_tf_predictions.json``
so results appear next to the Huber-trained runs without touching the existing
notebooks or artifacts.

Full-universe runs use lazy memmap arrays and chunked prediction (no .astype()
on complete tensors).

Usage:
    # Dev200 smoke test
    python scripts/train_qlike_variants.py --universe stocks --architectures garchgru

    # Full-universe GARCH-GRU (disk-backed, ~hours on GPU)
    python scripts/train_qlike_variants.py --universe stocks --architectures garchgru \\
        --dataset-tag full --epochs 100 --patience 8
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


def _artifact_stem(architecture: str, universe: str, dataset_tag: str) -> tuple[str, str]:
    arch_tag = f"{architecture}_qlike"
    model_tag = f"_{dataset_tag}" if dataset_tag != "full" else ""
    artifact_stem = f"{arch_tag}_vol_risk_{universe}{model_tag}_tf"
    return arch_tag, artifact_stem


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
        predict_in_batches,
    )

    root, models_dir, _, tf = bootstrap_notebook(42)
    device_info = configure_tf_gpu(tf, enable_mixed_precision=True)

    prefix = "etfs_universe" if universe == "etfs" else "stocks_universe"
    stem = f"{prefix}_{dataset_tag}" if dataset_tag != "full" else prefix
    bundle = load_sequence_bundle(sequences_name=f"{stem}_sequences_plus5.npz")

    # Keep disk-backed full-universe arrays lazy; .astype() would OOM WSL.
    X_train, y_train = bundle.X_train, bundle.y_train
    X_val, y_val = bundle.X_val, bundle.y_val
    X_test, y_test = bundle.X_test, bundle.y_test

    batch_size = 512 if device_info["using_gpu"] else 1024
    lr = 1e-3
    arch_kwargs = {}
    if architecture == "garchgru":
        arch_kwargs["return_idx"] = bundle.feature_cols.index("log_return")

    train_ds, val_ds = make_tf_datasets(X_train, y_train, X_val, y_val, batch_size=batch_size)
    model = build_regressor(
        architecture, X_train.shape[1], X_train.shape[-1], lr=lr, loss="qlike", **arch_kwargs
    )

    arch_tag, artifact_stem = _artifact_stem(architecture, universe, dataset_tag)
    best_model_path = models_dir / f"{artifact_stem}_best.keras"
    epoch_ckpt_path = models_dir / f"{artifact_stem}_epoch{{epoch:03d}}.keras"

    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    print(
        f"[{architecture}/{universe}] training with QLIKE | dataset={dataset_tag} | "
        f"storage={bundle.storage_format} | symbols={bundle.cfg.get('n_symbols')} | "
        f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,} | "
        f"steps/epoch={steps_per_epoch:,} batch={batch_size} | "
        f"gpu={device_info['using_gpu']} epochs<={epochs} patience={patience}"
    )
    if device_info["using_gpu"]:
        print(
            "GPU note: first epoch may take 10–30+ min while TF JIT-compiles CUDA kernels."
        )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", mode="min", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(epoch_ckpt_path),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
    ]

    device = "/GPU:0" if device_info["using_gpu"] else "/CPU:0"
    with tf.device(device):
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2
        )

    val_pred = as_1d_predictions(predict_in_batches(model, X_val, batch_size))
    test_pred = as_1d_predictions(predict_in_batches(model, X_test, batch_size))

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

    model_tag = f"_{dataset_tag}" if dataset_tag != "full" else ""
    baseline_path = models_dir / f"baseline_vol_risk_{universe}{model_tag}_tf_predictions.json"
    huber_path = models_dir / f"{architecture}_vol_risk_{universe}{model_tag}_tf_predictions.json"
    baseline_comparison = {"baseline_trained": False}
    if baseline_path.exists():
        with open(baseline_path) as f:
            b = json.load(f)
        beats_val = val_mae < b.get("val_mae", float("inf"))
        beats_crisis = test_mae < b.get("crisis_mae", float("inf"))
        baseline_comparison = {
            "baseline_trained": True,
            "baseline_artifact": baseline_path.name,
            "baseline_val_mae": b.get("val_mae"),
            "baseline_crisis_mae": b.get("crisis_mae"),
            "beats_baseline_val_mae": beats_val,
            "beats_baseline_crisis_mae": beats_crisis,
            "delta_val_mae": float(val_mae - b.get("val_mae", val_mae)),
            "delta_crisis_mae": float(test_mae - b.get("crisis_mae", test_mae)),
            "promotion_eligible": beats_val and beats_crisis,
        }

    huber_comparison = {"huber_trained": False}
    if huber_path.exists():
        with open(huber_path) as f:
            h = json.load(f)
        huber_comparison = {
            "huber_trained": True,
            "huber_artifact": huber_path.name,
            "huber_val_mae": h.get("val_mae"),
            "huber_crisis_mae": h.get("crisis_mae"),
            "beats_huber_val_mae": val_mae < h.get("val_mae", float("inf")),
            "beats_huber_crisis_mae": test_mae < h.get("crisis_mae", float("inf")),
            "delta_val_mae_vs_huber": float(val_mae - h.get("val_mae", val_mae)),
            "delta_crisis_mae_vs_huber": float(test_mae - h.get("crisis_mae", test_mae)),
        }

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
        "huber_comparison": huber_comparison,
        "train_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "loss": "qlike",
            "architecture": architecture,
            "device": device_info,
            "storage_format": bundle.storage_format,
            "sequences_name": f"{stem}_sequences_plus5.npz",
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
        f"[{architecture}/{universe}] QLIKE done | best_epoch={payload['best_epoch']} | "
        f"val MAE {val_mae:.6f} | crisis MAE {test_mae:.6f} | "
        f"crisis under-MAE {asym_crisis['under_mae']:.6f} "
        f"(rate {asym_crisis['under_rate']:.1%})"
    )
    if baseline_comparison.get("baseline_trained"):
        print(
            f"Vs full baseline: val {'beats' if baseline_comparison['beats_baseline_val_mae'] else 'loses'} "
            f"({val_mae:.6f} vs {baseline_comparison['baseline_val_mae']:.6f}) | "
            f"crisis {'beats' if baseline_comparison['beats_baseline_crisis_mae'] else 'loses'} "
            f"({test_mae:.6f} vs {baseline_comparison['baseline_crisis_mae']:.6f}) | "
            f"promotion_eligible={baseline_comparison.get('promotion_eligible')}"
        )
    if huber_comparison.get("huber_trained"):
        print(
            f"Vs Huber GARCH-GRU: crisis {'beats' if huber_comparison['beats_huber_crisis_mae'] else 'loses'} "
            f"({test_mae:.6f} vs {huber_comparison['huber_crisis_mae']:.6f})"
        )
    print(f"Saved {pred_path.name} and {best_model_path.name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", nargs="+", default=["stocks", "etfs"])
    parser.add_argument("--architectures", nargs="+", default=["gru", "garchgru"])
    parser.add_argument("--dataset-tag", default="dev200")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else (100 if args.dataset_tag == "full" else 30)
    patience = args.patience if args.patience is not None else 8

    for universe in args.universe:
        for arch in args.architectures:
            train_one(arch, universe, args.dataset_tag, epochs, patience)


if __name__ == "__main__":
    main()
