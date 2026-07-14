#!/usr/bin/env python3
"""Gracefully stop an in-flight Keras fit in a live Jupyter kernel and save artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from jupyter_client import BlockingKernelClient


def _stream_output(msg: dict) -> None:
    msg_type = msg["msg_type"]
    content = msg["content"]
    if msg_type == "stream":
        print(content.get("text", ""), end="")
    elif msg_type == "error":
        print("".join(content.get("traceback", [])), file=sys.stderr)
    elif msg_type == "execute_result":
        text = content.get("data", {}).get("text/plain")
        if text:
            print(text)


def execute(client: BlockingKernelClient, code: str, timeout: float = 600.0) -> str:
    msg_id = client.execute(code)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = client.get_iopub_msg(timeout=1.0)
        except Exception:
            if not client.is_alive():
                raise RuntimeError("Kernel died while executing code.")
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        _stream_output(msg)
        if msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle":
            break
    else:
        raise TimeoutError(f"Timed out after {timeout:.0f}s executing:\n{code}")
    return msg_id


def wait_until_idle(client: BlockingKernelClient, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg_id = client.execute("1")
        while time.time() < deadline:
            try:
                msg = client.get_iopub_msg(timeout=2.0)
            except Exception:
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            if msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle":
                return
        time.sleep(2.0)
    raise TimeoutError(f"Kernel still busy after {timeout:.0f}s.")


SAVE_CODE_TEMPLATE = r'''
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tf_notebook_utils import (
    as_1d_predictions,
    asymmetric_error_metrics,
    persistence_mae,
    predict_in_batches,
)

ARTIFACT_STEM = globals().get("ARTIFACT_STEM") or "{artifact_stem}"
LOSS = globals().get("LOSS") or "{loss}"

if "history" in globals() and history is not None and getattr(history, "history", None):
    best_epoch = int(np.argmin(history.history["val_mae"]) + 1)
    best_val_mae = float(np.min(history.history["val_mae"]))
    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "val_mae": history.history["val_mae"],
        }
    )
    history_records = hist_df.to_dict(orient="records")
else:
    # Interrupt may return before `history = model.fit(...)` assignment, but
    # EarlyStopping(restore_best_weights=True) should already have restored weights.
    best_epoch = None
    best_val_mae = float("nan")
    history_records = []
    ckpt_stem = ARTIFACT_STEM
    epoch_ckpts = sorted(MODELS_DIR.glob(f"{ckpt_stem}_epoch*.keras"))
    if epoch_ckpts:
        best_epoch = int(epoch_ckpts[-1].stem.rsplit("_epoch", 1)[-1])
        print(f"WARN: history missing; inferred best_epoch={best_epoch} from latest checkpoint.")
    else:
        print("WARN: history missing; saving restored in-memory weights and recomputing metrics.")

val_pred = as_1d_predictions(predict_in_batches(model, X_val, BATCH_SIZE))
test_pred = as_1d_predictions(predict_in_batches(model, X_test, BATCH_SIZE))

val_mae = float(np.mean(np.abs(val_pred - y_val)))
val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
test_mae = float(np.mean(np.abs(test_pred - y_test)))
test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))

persistence_mae_val = persistence_mae(X_val, y_val, bundle.feature_cols, bundle.scaler)
persistence_mae_test = persistence_mae(X_test, y_test, bundle.feature_cols, bundle.scaler)
train_mean = float(np.mean(y_train))
train_mean_mae_val = float(np.mean(np.abs(train_mean - y_val)))
train_mean_mae_test = float(np.mean(np.abs(train_mean - y_test)))
asym_val = asymmetric_error_metrics(y_val, val_pred)
asym_crisis = asymmetric_error_metrics(y_test, test_pred)

artifact_stem = ARTIFACT_STEM
model_architecture = globals().get("MODEL_ARCHITECTURE") or ("garchgru_qlike" if LOSS == "qlike" else ARCHITECTURE)
best_model_path = MODELS_DIR / f"{artifact_stem}_best.keras"
final_model_path = MODELS_DIR / f"{artifact_stem}_final.keras"
pred_path = MODELS_DIR / f"{artifact_stem}_predictions.json"

model.save(best_model_path)
model.save(final_model_path)

baseline_pred_path = MODELS_DIR / f"baseline_{JOB_SLUG}_{UNIVERSE}{MODEL_TAG}_tf_predictions.json"
baseline_comparison = {"baseline_trained": False}
if baseline_pred_path.exists():
    with open(baseline_pred_path) as f:
        baseline_payload = json.load(f)
    baseline_comparison = {
        "baseline_trained": True,
        "baseline_artifact": baseline_pred_path.name,
        "baseline_val_mae": baseline_payload.get("val_mae"),
        "baseline_crisis_mae": baseline_payload.get("crisis_mae"),
        "beats_baseline_val_mae": val_mae < baseline_payload.get("val_mae", float("inf")),
        "beats_baseline_crisis_mae": test_mae < baseline_payload.get("crisis_mae", float("inf")),
        "delta_val_mae": float(val_mae - baseline_payload.get("val_mae", val_mae)),
        "delta_crisis_mae": float(test_mae - baseline_payload.get("crisis_mae", test_mae)),
    }

huber_comparison = {"huber_trained": False}
if LOSS == "qlike":
    huber_pred_path = MODELS_DIR / f"{ARCHITECTURE}_{JOB_SLUG}_{UNIVERSE}{MODEL_TAG}_tf_predictions.json"
    if huber_pred_path.exists():
        with open(huber_pred_path) as f:
            huber_payload = json.load(f)
        huber_comparison = {
            "huber_trained": True,
            "huber_artifact": huber_pred_path.name,
            "huber_val_mae": huber_payload.get("val_mae"),
            "huber_crisis_mae": huber_payload.get("crisis_mae"),
            "beats_huber_val_mae": val_mae < huber_payload.get("val_mae", float("inf")),
            "beats_huber_crisis_mae": test_mae < huber_payload.get("crisis_mae", float("inf")),
            "delta_val_mae": float(val_mae - huber_payload.get("val_mae", val_mae)),
            "delta_crisis_mae": float(test_mae - huber_payload.get("crisis_mae", test_mae)),
        }

payload = {
    "job_id": JOB_ID,
    "job_slug": JOB_SLUG,
    "architecture": model_architecture,
    "universe": UNIVERSE,
    "model": f"{ARCHITECTURE}_{JOB_SLUG}_{UNIVERSE}{MODEL_TAG}_tensorflow",
    "run_mode": RUN_MODE,
    "dataset_tag": DATASET_TAG,
    "n_symbols": bundle.cfg.get("n_symbols"),
    "framework": "tensorflow",
    "target_type": bundle.cfg["target_type"],
    "n_features": len(bundle.feature_cols),
    "lookback_days": bundle.cfg["lookback_days"],
    "feature_cols": bundle.feature_cols,
    "best_epoch": best_epoch,
    "best_val_mae": best_val_mae,
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
    "huber_comparison": huber_comparison if LOSS == "qlike" else {"huber_trained": False},
    "train_config": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "loss": LOSS,
        "architecture": ARCHITECTURE,
        "device": device_info,
        "sequences_name": sequences_name,
        "storage_format": getattr(bundle, "storage_format", "npz"),
        "stopped_early": True,
    },
    "history": history_records,
    "predictions": {
        "val": {"y_true": np.asarray(y_val).tolist(), "y_pred": val_pred.tolist()},
        "test": {"y_true": np.asarray(y_test).tolist(), "y_pred": test_pred.tolist()},
    },
    "trained_at": datetime.now(timezone.utc).isoformat(),
}

with open(pred_path, "w") as f:
    json.dump(payload, f, indent=2)

print("SAFE_SAVE_OK")
print(f"best_epoch={best_epoch} best_val_mae={best_val_mae:.6f} val_mae={val_mae:.6f} crisis_mae={test_mae:.6f}")
print(f"symbols={bundle.cfg.get('n_symbols')} storage={getattr(bundle, 'storage_format', 'npz')}")
print(f"saved_best={best_model_path}")
print(f"saved_final={final_model_path}")
print(f"saved_predictions={{pred_path}}")
'''


def backup_existing(models_dir: Path, stem: str) -> Path | None:
    targets = [
        models_dir / f"{stem}_best.keras",
        models_dir / f"{stem}_final.keras",
        models_dir / f"{stem}_predictions.json",
    ]
    existing = [p for p in targets if p.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = models_dir / f".backup_{stem}_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    print(f"Backed up {len(existing)} prior artifact(s) to {backup_dir}")
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-file",
        default="/run/user/1000/jupyter/runtime/kernel-v335268558e43ed416b44db5280caa50975d972582.json",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=7200.0,
        help="Seconds to wait for the current epoch/fit to finish after stop request.",
    )
    parser.add_argument(
        "--artifact-stem",
        default="garchgru_qlike_vol_risk_stocks_tf",
    )
    parser.add_argument(
        "--kernel-pid",
        type=int,
        default=None,
        help="Optional ipykernel PID to SIGINT when the kernel is busy training.",
    )
    parser.add_argument(
        "--loss",
        default="qlike",
        help="Loss label to embed when the kernel globals are missing/stale.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for kernel heartbeat before proceeding anyway.",
    )
    args = parser.parse_args()

    models_dir = Path(__file__).resolve().parents[1] / "data" / "models"
    backup_existing(models_dir, args.artifact_stem)

    client = BlockingKernelClient()
    client.load_connection_file(args.connection_file)
    client.start_channels()
    try:
        client.wait_for_ready(timeout=args.ready_timeout)
        kernel_idle = True
    except RuntimeError:
        kernel_idle = False
        print(
            f"Kernel heartbeat not ready within {args.ready_timeout:.0f}s "
            "(likely mid-epoch). Will request interrupt."
        )

    if not kernel_idle and args.kernel_pid:
        print(f"Sending SIGINT to kernel pid {args.kernel_pid} to end after current batch...")
        os.kill(args.kernel_pid, signal.SIGINT)
        time.sleep(5.0)
        for _ in range(int(args.wait_timeout // 5)):
            try:
                client.wait_for_ready(timeout=5)
                kernel_idle = True
                break
            except RuntimeError:
                time.sleep(5.0)
        if not kernel_idle:
            raise TimeoutError(
                f"Kernel pid {args.kernel_pid} still busy after SIGINT + {args.wait_timeout:.0f}s."
            )

    print("Probing kernel state...")
    execute(
        client,
        """
import json
state = {
    "has_model": "model" in globals(),
    "has_history": "history" in globals(),
    "has_bundle": "bundle" in globals(),
    "stop_training": getattr(globals().get("model"), "stop_training", None),
    "n_symbols": globals().get("bundle").cfg.get("n_symbols") if "bundle" in globals() else None,
    "train_rows": len(globals().get("X_train")) if "X_train" in globals() else None,
    "storage": getattr(globals().get("bundle"), "storage_format", None) if "bundle" in globals() else None,
}
print("KERNEL_STATE", json.dumps(state))
""",
        timeout=300,
    )

    if kernel_idle:
        print("Requesting graceful stop (model.stop_training = True)...")
        try:
            execute(
                client,
                """
if "model" not in globals():
    raise RuntimeError("No in-memory model found in kernel.")
model.stop_training = True
print("STOP_REQUESTED")
""",
                timeout=120,
            )
            print(f"Waiting up to {args.wait_timeout:.0f}s for fit to finish and restore best weights...")
            wait_until_idle(client, timeout=args.wait_timeout)
        except TimeoutError:
            if args.kernel_pid:
                print("Stop flag did not settle quickly; sending SIGINT.")
                os.kill(args.kernel_pid, signal.SIGINT)
                wait_until_idle(client, timeout=args.wait_timeout)
            else:
                raise
    print("Kernel idle. Saving best-epoch artifacts from in-memory model/history...")

    save_code = SAVE_CODE_TEMPLATE.format(
        artifact_stem=args.artifact_stem,
        loss=args.loss,
    )
    execute(client, save_code, timeout=args.wait_timeout)
    client.stop_channels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
