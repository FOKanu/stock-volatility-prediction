#!/usr/bin/env python3
"""Restore notebook train/evaluate outputs from saved *_predictions.json (no retrain)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_iteration2_jobs import JOB_TASK_KIND, ML_DIR, ROOT

NOTEBOOK_RE = re.compile(
    r"^(?:(?P<baseline>00a_baseline)|(?P<num>\d{2})(?P<letter>[a-f])_(?P<arch>gru|tcn|lstm|cnn1d|xgb|garchgru))_"
    r"(?P<job_slug>vol_risk|vol_regime|rel_strength|etf_opportunity)_"
    r"(?P<universe>stocks|etfs|market)$"
)


def parse_notebook_stem(stem: str) -> dict[str, str]:
    m = NOTEBOOK_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognized notebook name: {stem}")
    if m.group("baseline"):
        return {
            "architecture": "baseline",
            "job_slug": m.group("job_slug"),
            "universe": m.group("universe"),
        }
    return {
        "architecture": m.group("arch"),
        "job_slug": m.group("job_slug"),
        "universe": m.group("universe"),
    }


def predictions_path(
    models_dir: Path,
    architecture: str,
    job_slug: str,
    universe: str,
    dataset_tag: str,
) -> Path:
    return models_dir / f"{architecture}_{job_slug}_{universe}_{dataset_tag}_tf_predictions.json"


def _fig_to_output(fig: plt.Figure) -> dict:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {"image/png": data},
        "metadata": {},
    }


def _text_output(text: str) -> dict:
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text.splitlines(keepends=True),
    }


def _history_to_keras_format(history_rows: list[dict]) -> dict[str, list[float]]:
    if not history_rows:
        return {"loss": [0.0], "val_loss": [0.0], "val_mae": [0.0]}
    keys = history_rows[0].keys()
    out: dict[str, list[float]] = {k: [] for k in keys if k != "epoch"}
    for row in history_rows:
        for k in out:
            out[k].append(float(row[k]))
    if "loss" not in out and "train_loss" in out:
        out["loss"] = out["train_loss"]
    return out


def train_summary_text(payload: dict) -> str:
    arch = payload.get("architecture", "?")
    universe = payload.get("universe", "?")
    task = JOB_TASK_KIND.get(payload.get("job_slug", ""), "regression")
    lines = [
        f"[REPLAYED FROM SAVED ARTIFACT — no training run]\n",
        f"Architecture: {arch} | Universe: {universe} | Task: {task}\n",
        f"Artifact: {payload.get('_artifact_name', 'predictions.json')}\n",
        f"Trained at: {payload.get('trained_at', 'n/a')}\n",
    ]
    if task == "classification":
        lines.extend(
            [
                f"Val macro-F1:  {payload.get('val_macro_f1', float('nan')):.4f}\n",
                f"Crisis macro-F1: {payload.get('crisis_macro_f1', float('nan')):.4f}\n",
                f"Val accuracy:  {payload.get('val_accuracy', float('nan')):.4f}\n",
            ]
        )
    else:
        lines.extend(
            [
                f"Best epoch: {payload.get('best_epoch', 'n/a')} | "
                f"Best val MAE: {payload.get('best_val_mae', float('nan')):.6f}\n",
                f"Val  MAE: {payload.get('val_mae', float('nan')):.6f} | "
                f"RMSE: {payload.get('val_rmse', float('nan')):.6f}\n",
                f"Crisis MAE: {payload.get('crisis_mae', float('nan')):.6f} | "
                f"RMSE: {payload.get('test_rmse', float('nan')):.6f}\n",
            ]
        )
        bc = payload.get("baseline_comparison") or {}
        if bc.get("baseline_trained"):
            beats_v = bc.get("beats_baseline_val_mae")
            beats_c = bc.get("beats_baseline_crisis_mae")
            lines.append(
                f"Vs baseline: val MAE {'beats' if beats_v else 'loses to'} | "
                f"crisis MAE {'beats' if beats_c else 'loses to'}\n"
            )
    return "".join(lines)


def replay_regression_outputs(payload: dict, model_label: str) -> list[dict]:
    outputs: list[dict] = []
    history = _history_to_keras_format(payload.get("history") or [])
    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["loss"]) + 1),
            "train_loss": history["loss"],
            "val_loss": history.get("val_loss", history["loss"]),
            "val_mae": history.get("val_mae", history["loss"]),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(hist_df["epoch"], hist_df["val_mae"], color="#2980b9")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation MAE")
    fig.suptitle(f"{model_label} — training (replayed)", y=1.02)
    plt.tight_layout()
    outputs.append(_fig_to_output(fig))

    y_val = np.asarray(payload["predictions"]["val"]["y_true"], dtype=np.float32)
    y_test = np.asarray(payload["predictions"]["test"]["y_true"], dtype=np.float32)
    val_pred = np.asarray(payload["predictions"]["val"]["y_pred"], dtype=np.float32)
    test_pred = np.asarray(payload["predictions"]["test"]["y_pred"], dtype=np.float32)

    for y_true, y_pred, title in (
        (y_val, val_pred, f"{model_label} — validation"),
        (y_test, test_pred, f"{model_label} — COVID test"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=18, edgecolors="none")
        lim = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
        axes[0].plot([0, lim], [0, lim], "r--", lw=1)
        axes[0].set_xlabel("Actual |log return|")
        axes[0].set_ylabel("Predicted |log return|")
        axes[0].set_title("Predicted vs actual")
        resid = y_pred - y_true
        axes[1].hist(resid, bins=30, color="#7f8c8d", edgecolor="white")
        axes[1].axvline(0, color="red", ls="--", lw=1)
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual distribution")
        fig.suptitle(title + " (replayed)", y=1.02)
        plt.tight_layout()
        outputs.append(_fig_to_output(fig))

    baselines = payload.get("baselines") or {}
    ref = payload.get("reference_baselines") or baselines
    table = pd.DataFrame(
        [
            {
                "Model": model_label,
                "Val MAE": payload.get("val_mae"),
                "Crisis MAE": payload.get("crisis_mae"),
                "Val RMSE": payload.get("val_rmse"),
            },
            {
                "Model": "Persistence (val)",
                "Val MAE": ref.get("persistence_mae_val"),
                "Crisis MAE": ref.get("persistence_mae_test"),
                "Val RMSE": np.nan,
            },
            {
                "Model": "Train-mean (test)",
                "Val MAE": ref.get("train_mean_mae_val"),
                "Crisis MAE": ref.get("train_mean_mae_test"),
                "Val RMSE": np.nan,
            },
        ]
    )
    outputs.append(_text_output(table.to_string(index=False) + "\n"))
    return outputs


def replay_classification_outputs(payload: dict, model_label: str) -> list[dict]:
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    outputs: list[dict] = []
    history = _history_to_keras_format(payload.get("history") or [])
    score_key = "val_auc" if "val_auc" in history else "val_mae"
    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["loss"]) + 1),
            "train_loss": history["loss"],
            "val_loss": history.get("val_loss", history["loss"]),
            "val_score": history.get(score_key, history["loss"]),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(hist_df["epoch"], hist_df["val_score"], color="#2980b9")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val AUC" if score_key == "val_auc" else "Val score")
    axes[1].set_title("Validation score")
    fig.suptitle(f"{model_label} — vol regime (replayed)", y=1.02)
    plt.tight_layout()
    outputs.append(_fig_to_output(fig))

    for split, title in (("val", "validation"), ("test", "COVID test")):
        y_true = np.asarray(payload["predictions"][split]["y_true"])
        y_pred = np.asarray(payload["predictions"][split]["y_pred"])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(y_true.astype(int), y_pred.astype(int)),
            display_labels=["low vol", "high vol"],
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{model_label} — {title} (replayed)")
        plt.tight_layout()
        outputs.append(_fig_to_output(fig))

    table = pd.DataFrame(
        [
            {
                "Model": model_label,
                "Val macro-F1": payload.get("val_macro_f1"),
                "Crisis macro-F1": payload.get("crisis_macro_f1"),
                "Val accuracy": payload.get("val_accuracy"),
            },
        ]
    )
    outputs.append(_text_output(table.to_string(index=False) + "\n"))
    return outputs


def model_label(payload: dict) -> str:
    arch = payload.get("architecture", "?")
    universe = payload.get("universe", "?")
    if arch == "baseline":
        return f"Baseline ({universe})"
    return f"{str(arch).upper()} ({universe})"


def build_outputs(payload: dict) -> tuple[list[dict], list[dict]]:
    """Return (train_cell_outputs, evaluate_cell_outputs)."""
    payload = dict(payload)
    task = JOB_TASK_KIND.get(payload.get("job_slug", ""), "regression")
    label = model_label(payload)
    train_out = [_text_output(train_summary_text(payload))]
    if task == "classification":
        eval_out = replay_classification_outputs(payload, label)
    else:
        eval_out = replay_regression_outputs(payload, label)
    return train_out, eval_out


def find_code_cell_index(nb: dict, needle: str) -> int | None:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if needle in "".join(cell.get("source", [])):
            return i
    return None


def inject_notebook_outputs(
    notebook_path: Path,
    payload: dict,
    *,
    dry_run: bool = False,
) -> bool:
    pred_name = payload.get("_artifact_name", "")
    if not notebook_path.exists():
        print(f"Skip (missing notebook): {notebook_path}")
        return False

    nb = json.loads(notebook_path.read_text())
    train_idx = find_code_cell_index(nb, "model = build_regressor")
    if train_idx is None:
        train_idx = find_code_cell_index(nb, "model = build_classifier")
    if train_idx is None:
        train_idx = find_code_cell_index(nb, "model = train_baseline_model")
    if train_idx is None:
        train_idx = find_code_cell_index(nb, "train_hist_gradient_boosting")
    eval_idx = find_code_cell_index(nb, "hist_df = pd.DataFrame")

    if train_idx is None or eval_idx is None:
        print(f"Skip (train/eval cells not found): {notebook_path}")
        return False

    train_out, eval_out = build_outputs(payload)

    if dry_run:
        print(
            f"Would inject into {notebook_path.relative_to(ROOT)} "
            f"(train cell {train_idx}, eval cell {eval_idx}) <- {pred_name}"
        )
        return True

    nb["cells"][train_idx]["outputs"] = train_out
    nb["cells"][train_idx]["execution_count"] = 1
    nb["cells"][eval_idx]["outputs"] = eval_out
    nb["cells"][eval_idx]["execution_count"] = 1
    notebook_path.write_text(json.dumps(nb, indent=1))
    print(f"Replayed {notebook_path.relative_to(ROOT)} <- {pred_name}")
    return True


def discover_notebooks(
    job_slug: str | None,
    universe: str | None,
) -> list[Path]:
    paths: list[Path] = []
    for job_dir in sorted(ML_DIR.glob("J*_*")):
        if job_slug and f"_{job_slug}" not in job_dir.name:
            continue
        for nb_path in sorted(job_dir.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in str(nb_path):
                continue
            if universe and f"/{universe}/" not in str(nb_path):
                continue
            try:
                parse_notebook_stem(nb_path.stem)
            except ValueError:
                continue
            paths.append(nb_path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay saved metrics/plots into notebooks without retraining"
    )
    parser.add_argument("--notebook", type=Path, help="Single notebook to replay")
    parser.add_argument("--job-slug", help="Replay all notebooks for a job slug")
    parser.add_argument("--universe", choices=["stocks", "etfs", "market"])
    parser.add_argument("--dataset-tag", default="dev200")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "data" / "models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir
    if not models_dir.is_dir():
        raise SystemExit(f"Models dir not found: {models_dir}")

    targets: list[Path]
    if args.notebook:
        targets = [args.notebook.resolve()]
    elif args.job_slug:
        targets = discover_notebooks(args.job_slug, args.universe)
    else:
        raise SystemExit("Provide --notebook or --job-slug")

    ok = 0
    missing = 0
    for nb_path in targets:
        meta = parse_notebook_stem(nb_path.stem)
        pred_path = predictions_path(
            models_dir,
            meta["architecture"],
            meta["job_slug"],
            meta["universe"],
            args.dataset_tag,
        )
        if not pred_path.exists():
            print(f"Skip (no predictions): {pred_path.name}")
            missing += 1
            continue
        with open(pred_path) as f:
            payload = json.load(f)
        payload["_artifact_name"] = pred_path.name
        if inject_notebook_outputs(nb_path, payload, dry_run=args.dry_run):
            ok += 1

    print(f"\nReplayed: {ok} | Missing predictions: {missing}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
