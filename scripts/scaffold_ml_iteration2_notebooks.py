#!/usr/bin/env python3
"""Generate ML Iteration 2 notebook tree from the 09a template."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from kernel_config import KERNEL_METADATA, KERNEL_PICKER_HINT
from ml_iteration2_jobs import (
    ARCHITECTURE_LABELS,
    JOB_PRIMARY_METRIC,
    JOB_TASK_KIND,
    ML_DIR,
    NotebookSpec,
    all_notebook_specs_including_baselines,
    filter_notebook_specs,
    notebook_path,
    target_artifact_suffix,
    universe_prefix,
)

TEMPLATE_PATH = ML_DIR / "J01_vol_risk" / "stocks" / "01a_gru_vol_risk_stocks.ipynb"

NOTEBOOK_KERNEL_METADATA = KERNEL_METADATA


def _apply_kernel_metadata(nb: dict) -> dict:
    nb.setdefault("metadata", {})
    nb["metadata"]["kernelspec"] = dict(NOTEBOOK_KERNEL_METADATA["kernelspec"])
    nb["metadata"]["language_info"] = {
        **nb["metadata"].get("language_info", {}),
        **NOTEBOOK_KERNEL_METADATA["language_info"],
    }
    return nb


def _md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def _title_cell(spec: NotebookSpec) -> str:
    arch_label = ARCHITECTURE_LABELS[spec.architecture]
    name = "BASELINE" if spec.is_baseline else spec.architecture.upper()
    train_first = (
        "\n\n> **Train this notebook first** in each job folder. Models `01a`–`01e` report whether they beat this baseline."
        if spec.is_baseline
        else ""
    )
    return f"""# {spec.job_num if not spec.is_baseline else '00a'} — {name} · {spec.job_title} · {spec.universe}

**Job:** `J{spec.job_id}_{spec.job_slug}` | **Phase:** {spec.phase} | **Status:** {spec.status}

**Architecture:** {arch_label}{train_first}

**Project path (WSL):** `~/ai-workspace/experiments/stock-prediction`

{KERNEL_PICKER_HINT}

```bash
cd ~/ai-workspace/experiments/stock-prediction
source .venv/bin/activate
```
"""


def _config_cell(spec: NotebookSpec) -> str:
    prefix = universe_prefix(spec.universe)
    target_suffix = target_artifact_suffix(spec.target_type)
    return f'''# --- ML Iteration 2 job config (do not rename without updating scaffold) ---
JOB_ID = "{spec.job_id}"
JOB_SLUG = "{spec.job_slug}"
ARCHITECTURE = "{spec.architecture}"
UNIVERSE = "{spec.universe}"
ASSET_TYPE = "{spec.asset_type}"
TARGET_TYPE = "{spec.target_type}"
UNIVERSE_PREFIX = "{prefix}"
TARGET_SUFFIX = "{target_suffix}"

RUN_MODE = "fast_dev"  # "fast_dev" | "full"
FAST_DEV_MAX_SYMBOLS = 200
FORCE_REBUILD_DATA = False
FAST_DEV_EPOCHS = 30
FULL_EPOCHS = 100

processed_dir = None

if RUN_MODE == "fast_dev":
    DATASET_TAG = f"dev{{FAST_DEV_MAX_SYMBOLS}}"
    sequences_name = f"{{UNIVERSE_PREFIX}}_{{DATASET_TAG}}{{TARGET_SUFFIX}}_sequences_plus5.npz"
    config_name = f"pipeline_config_{{UNIVERSE_PREFIX}}_{{DATASET_TAG}}{{TARGET_SUFFIX}}_plus5.json"
    scaler_name = f"scaler_{{UNIVERSE_PREFIX}}_{{DATASET_TAG}}{{TARGET_SUFFIX}}_plus5.json"
    MODEL_TAG = f"_{{DATASET_TAG}}"
    TRAIN_EPOCHS = FAST_DEV_EPOCHS
else:
    DATASET_TAG = "full"
    sequences_name = f"{{UNIVERSE_PREFIX}}{{TARGET_SUFFIX}}_sequences_plus5.npz"
    config_name = f"pipeline_config_{{UNIVERSE_PREFIX}}{{TARGET_SUFFIX}}_plus5.json"
    scaler_name = f"scaler_{{UNIVERSE_PREFIX}}{{TARGET_SUFFIX}}_plus5.json"
    MODEL_TAG = ""
    TRAIN_EPOCHS = FULL_EPOCHS

print(
    f"JOB=J{{JOB_ID}}_{{JOB_SLUG}} | ARCH={{ARCHITECTURE}} | UNIVERSE={{UNIVERSE}} | "
    f"RUN_MODE={{RUN_MODE}} | epochs={{TRAIN_EPOCHS}} | sequences={{sequences_name}}"
)
'''


def _preprocess_cell(spec: NotebookSpec) -> str:
    if spec.universe == "market":
        max_symbols_line = 'prep_cmd.extend(["--symbols-v1", "SPY"])'
        asset_note = "SPY-only market regime lane"
    else:
        max_symbols_line = 'prep_cmd.extend(["--max-symbols", str(FAST_DEV_MAX_SYMBOLS)])'
        asset_note = f"{spec.universe} universe ({spec.asset_type})"

    return f'''FULL_UNIVERSE_ARTIFACTS = {{
    "sequences": processed_dir / f"{{UNIVERSE_PREFIX}}{{TARGET_SUFFIX}}_sequences_plus5.npz",
    "config": processed_dir / f"pipeline_config_{{UNIVERSE_PREFIX}}{{TARGET_SUFFIX}}_plus5.json",
    "scaler": processed_dir / f"scaler_{{UNIVERSE_PREFIX}}{{TARGET_SUFFIX}}_plus5.json",
}}

seq_path = processed_dir / sequences_name
cfg_path = processed_dir / config_name
scaler_path = processed_dir / scaler_name

artifacts_ready = seq_path.exists() and cfg_path.exists() and scaler_path.exists()


def _backup_full_universe_artifacts() -> Path | None:
    existing = [p for p in FULL_UNIVERSE_ARTIFACTS.values() if p.exists()]
    if not existing:
        return None
    backup_dir = Path(tempfile.mkdtemp(prefix="universe_full_backup_"))
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    print(f"Backed up {{len(existing)}} full-universe artifact(s) to {{backup_dir}}")
    return backup_dir


def _restore_full_universe_artifacts(backup_dir: Path | None) -> None:
    if backup_dir is None:
        return
    for path in backup_dir.iterdir():
        shutil.copy2(path, processed_dir / path.name)
    print("Restored full-universe artifacts after fast-dev preprocessing.")


def _promote_standard_to_dev_artifacts() -> None:
    promotions = [
        (FULL_UNIVERSE_ARTIFACTS["sequences"], seq_path),
        (FULL_UNIVERSE_ARTIFACTS["config"], cfg_path),
        (FULL_UNIVERSE_ARTIFACTS["scaler"], scaler_path),
    ]
    for src, dst in promotions:
        if not src.exists():
            raise FileNotFoundError(f"Expected preprocessing output missing: {{src}}")
        if dst.exists():
            dst.unlink()
        shutil.move(src, dst)
        print(f"Promoted -> {{dst.name}}")


prep_cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "run_preprocessing.py"),
    "--profile",
    "plus5",
    "--symbol-scope",
    "all",
    "--asset-type",
    ASSET_TYPE,
    "--target-type",
    TARGET_TYPE,
    "--history-outlier-filter",
    "none",
]

if RUN_MODE == "fast_dev":
    {max_symbols_line}

if artifacts_ready and not FORCE_REBUILD_DATA:
    print(f"Using existing {{RUN_MODE}} artifacts for {asset_note} (skip rebuild).")
    print(f"- {{seq_path}}")
elif RUN_MODE == "fast_dev":
    print(
        f"Running fast-dev preprocessing ({{FAST_DEV_MAX_SYMBOLS}} symbols, {{ASSET_TYPE}}) "
        f"-> {{DATASET_TAG}} artifacts..."
    )
    backup_dir = _backup_full_universe_artifacts()
    try:
        subprocess.run(prep_cmd, cwd=PROJECT_ROOT, check=True)
        _promote_standard_to_dev_artifacts()
    finally:
        _restore_full_universe_artifacts(backup_dir)
    print("Fast-dev preprocessing complete.")
else:
    print(f"Running preprocessing for full {{UNIVERSE}} universe (plus5, {{ASSET_TYPE}})...")
    subprocess.run(prep_cmd, cwd=PROJECT_ROOT, check=True)
    print("Preprocessing complete.")
'''


NOTEBOOK_IMPORT_BOOTSTRAP = '''import sys
from pathlib import Path
for _c in [Path.cwd(), *Path.cwd().parents]:
    _s = _c / "scripts"
    if (_s / "notebook_bootstrap.py").exists():
        sys.path.insert(0, str(_s))
        break
from notebook_bootstrap import ensure_scripts_path
ensure_scripts_path()

'''


def _baseline_comparison_block(task: str) -> str:
    if task == "classification":
        return '''
baseline_pred_path = MODELS_DIR / f"baseline_{JOB_SLUG}_{UNIVERSE}{MODEL_TAG}_tf_predictions.json"
baseline_comparison = {"baseline_trained": False}
if baseline_pred_path.exists():
    with open(baseline_pred_path) as f:
        baseline_payload = json.load(f)
    baseline_comparison = {
        "baseline_trained": True,
        "baseline_artifact": baseline_pred_path.name,
        "baseline_val_macro_f1": baseline_payload.get("val_macro_f1"),
        "baseline_crisis_macro_f1": baseline_payload.get("crisis_macro_f1"),
        "beats_baseline_val_macro_f1": val_macro_f1 > baseline_payload.get("val_macro_f1", -1.0),
        "beats_baseline_crisis_macro_f1": crisis_macro_f1 > baseline_payload.get("crisis_macro_f1", -1.0),
        "delta_val_macro_f1": float(val_macro_f1 - baseline_payload.get("val_macro_f1", val_macro_f1)),
        "delta_crisis_macro_f1": float(crisis_macro_f1 - baseline_payload.get("crisis_macro_f1", crisis_macro_f1)),
    }
    sign_v = "beats" if baseline_comparison["beats_baseline_val_macro_f1"] else "loses to"
    sign_c = "beats" if baseline_comparison["beats_baseline_crisis_macro_f1"] else "loses to"
    print(f"Vs trained baseline: model {sign_v} baseline on val macro-F1 ({val_macro_f1:.4f} vs {baseline_payload['val_macro_f1']:.4f})")
    print(f"Vs trained baseline: model {sign_c} baseline on crisis macro-F1 ({crisis_macro_f1:.4f} vs {baseline_payload['crisis_macro_f1']:.4f})")
else:
    print("Vs trained baseline: run 00a_baseline_* notebook first to enable comparison.")
'''
    return '''
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
    sign_v = "beats" if baseline_comparison["beats_baseline_val_mae"] else "loses to"
    sign_c = "beats" if baseline_comparison["beats_baseline_crisis_mae"] else "loses to"
    print(f"Vs trained baseline: model {sign_v} baseline on val MAE ({val_mae:.6f} vs {baseline_payload['val_mae']:.6f})")
    print(f"Vs trained baseline: model {sign_c} baseline on crisis MAE ({test_mae:.6f} vs {baseline_payload['crisis_mae']:.6f})")
else:
    print("Vs trained baseline: run 00a_baseline_* notebook first to enable comparison.")
'''


def _baseline_train_cell(spec: NotebookSpec) -> str:
    task = JOB_TASK_KIND[spec.job_slug]
    if task == "classification":
        return f'''{NOTEBOOK_IMPORT_BOOTSTRAP}from tf_notebook_utils import (
    train_baseline_model,
    as_1d_predictions,
    as_class_predictions,
    classification_metrics,
    make_single_epoch_history,
)

TASK = "{task}"
print(f"Training baseline ({{ARCHITECTURE}}) — Logistic on last-day features")
print(f"Task: {{TASK}}")
model = train_baseline_model(X_train, y_train, task=TASK, seed=SEED)
print(f"Baseline backend: {{getattr(model, 'backend', 'unknown')}}")

val_probs = as_1d_predictions(model.predict(X_val))
test_probs = as_1d_predictions(model.predict(X_test))
val_pred = as_class_predictions(val_probs)
test_pred = as_class_predictions(test_probs)

val_metrics = classification_metrics(y_val, val_pred)
crisis_metrics = classification_metrics(y_test, test_pred)
val_macro_f1 = val_metrics["macro_f1"]
crisis_macro_f1 = crisis_metrics["macro_f1"]
val_accuracy = val_metrics["accuracy"]
crisis_accuracy = crisis_metrics["accuracy"]
best_epoch = 1
best_val_macro_f1 = val_macro_f1
history = make_single_epoch_history(1.0 - val_macro_f1)
baseline_comparison = {{"is_baseline": True}}

print(f"Trained baseline Val macro-F1: {{val_macro_f1:.4f}} | accuracy: {{val_accuracy:.4f}}")
print(f"Trained baseline Crisis macro-F1: {{crisis_macro_f1:.4f}} | accuracy: {{crisis_accuracy:.4f}}")
'''

    persistence_block = ""
    if task == "regression" and spec.job_slug == "vol_risk":
        persistence_block = """
persist_val = persistence_predict(X_val, bundle.feature_cols, bundle.scaler)
persist_test = persistence_predict(X_test, bundle.feature_cols, bundle.scaler)
persistence_mae_val = float(np.mean(np.abs(persist_val - y_val)))
persistence_mae_test = float(np.mean(np.abs(persist_test - y_test)))
print("Reference (untrained) persistence:")
print(f"  Persistence MAE (val):  {persistence_mae_val:.6f}")
print(f"  Persistence MAE (test): {persistence_mae_test:.6f}")
ridge_beats_persist_val = val_mae < persistence_mae_val
ridge_beats_persist_test = test_mae < persistence_mae_test
print(
    f"Trained baseline vs persistence (val): "
    f"{'beats' if ridge_beats_persist_val else 'loses to'} ({val_mae:.6f} vs {persistence_mae_val:.6f})"
)
print(
    f"Trained baseline vs persistence (crisis): "
    f"{'beats' if ridge_beats_persist_test else 'loses to'} ({test_mae:.6f} vs {persistence_mae_test:.6f})"
)
"""
    return f'''{NOTEBOOK_IMPORT_BOOTSTRAP}from tf_notebook_utils import (
    persistence_mae,
    persistence_predict,
    train_baseline_model,
    as_1d_predictions,
    make_single_epoch_history,
)

TASK = "{task}"
BATCH_SIZE = 512
EPOCHS = 1
PATIENCE = 0
LR = 0.0

print(f"Training baseline ({{ARCHITECTURE}}) — Ridge/Logistic on last-day features")
print(f"Task: {{TASK}}")
model = train_baseline_model(X_train, y_train, task=TASK, seed=SEED)
print(f"Baseline backend: {{getattr(model, 'backend', 'unknown')}}")

val_pred = as_1d_predictions(model.predict(X_val))
test_pred = as_1d_predictions(model.predict(X_test))

val_mae = float(np.mean(np.abs(val_pred - y_val)))
val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
test_mae = float(np.mean(np.abs(test_pred - y_test)))
test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))
best_epoch = 1
best_val_mae = val_mae
history = make_single_epoch_history(val_mae)

train_mean = float(np.mean(y_train))
train_mean_mae_val = float(np.mean(np.abs(train_mean - y_val)))
train_mean_mae_test = float(np.mean(np.abs(train_mean - y_test)))
persistence_mae_val = persistence_mae(X_val, y_val, bundle.feature_cols, bundle.scaler)
persistence_mae_test = persistence_mae(X_test, y_test, bundle.feature_cols, bundle.scaler)

print(f"Trained baseline Val  MAE: {{val_mae:.6f}} | RMSE: {{val_rmse:.6f}}")
print(f"Trained baseline Crisis MAE: {{test_mae:.6f}} | RMSE: {{test_rmse:.6f}}")
print("Other references:")
print(f"  Train-mean MAE (val):   {{train_mean_mae_val:.6f}}")
print(f"  Train-mean MAE (test):  {{train_mean_mae_test:.6f}}")
{persistence_block}baseline_comparison = {{"is_baseline": True}}
'''


def _baseline_save_cell(spec: NotebookSpec) -> str:
    task = JOB_TASK_KIND[spec.job_slug]
    if task == "classification":
        return f'''artifact_stem = f"baseline_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}_tf"
best_model_path = MODELS_DIR / f"{{artifact_stem}}_best.joblib"
final_model_path = MODELS_DIR / f"{{artifact_stem}}_final.joblib"
pred_path = MODELS_DIR / f"{{artifact_stem}}_predictions.json"

model.save(best_model_path)
model.save(final_model_path)

payload = {{
    "job_id": JOB_ID,
    "job_slug": JOB_SLUG,
    "architecture": ARCHITECTURE,
    "universe": UNIVERSE,
    "is_baseline": True,
    "model": f"baseline_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}",
    "run_mode": RUN_MODE,
    "dataset_tag": DATASET_TAG,
    "n_symbols": bundle.cfg.get("n_symbols"),
    "framework": getattr(model, "backend", "baseline"),
    "target_type": bundle.cfg["target_type"],
    "n_features": len(bundle.feature_cols),
    "lookback_days": bundle.cfg["lookback_days"],
    "feature_cols": bundle.feature_cols,
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_val_macro_f1,
    "val_macro_f1": val_macro_f1,
    "val_accuracy": val_accuracy,
    "crisis_macro_f1": crisis_macro_f1,
    "crisis_accuracy": crisis_accuracy,
    "train_config": {{
        "architecture": ARCHITECTURE,
        "task": TASK,
        "sequences_name": sequences_name,
    }},
    "history": hist_df.to_dict(orient="records"),
    "predictions": {{
        "val": {{"y_true": y_val.tolist(), "y_pred": val_pred.tolist(), "y_prob": val_probs.tolist()}},
        "test": {{"y_true": y_test.tolist(), "y_pred": test_pred.tolist(), "y_prob": test_probs.tolist()}},
    }},
    "trained_at": datetime.now(timezone.utc).isoformat(),
}}

with open(pred_path, "w") as f:
    json.dump(payload, f, indent=2)

print("Saved baseline artifacts:")
print(f"- {{best_model_path}}")
print(f"- {{final_model_path}}")
print(f"- {{pred_path}}")
print("Train 02a–02e models next; they compare against this baseline automatically.")
'''

    extra_refs = ""
    if spec.job_slug == "vol_risk":
        extra_refs = """
    "reference_baselines": {
        "persistence_mae_val": persistence_mae_val,
        "persistence_mae_test": persistence_mae_test,
        "train_mean_mae_val": train_mean_mae_val,
        "train_mean_mae_test": train_mean_mae_test,
        "beats_persistence_val": bool(val_mae < persistence_mae_val),
        "beats_persistence_test": bool(test_mae < persistence_mae_test),
    },
"""
    return f'''artifact_stem = f"baseline_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}_tf"
best_model_path = MODELS_DIR / f"{{artifact_stem}}_best.joblib"
final_model_path = MODELS_DIR / f"{{artifact_stem}}_final.joblib"
pred_path = MODELS_DIR / f"{{artifact_stem}}_predictions.json"

model.save(best_model_path)
model.save(final_model_path)

payload = {{
    "job_id": JOB_ID,
    "job_slug": JOB_SLUG,
    "architecture": ARCHITECTURE,
    "universe": UNIVERSE,
    "is_baseline": True,
    "model": f"baseline_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}",
    "run_mode": RUN_MODE,
    "dataset_tag": DATASET_TAG,
    "n_symbols": bundle.cfg.get("n_symbols"),
    "framework": getattr(model, "backend", "baseline"),
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
{extra_refs}    "train_config": {{
        "architecture": ARCHITECTURE,
        "task": TASK,
        "sequences_name": sequences_name,
    }},
    "history": hist_df.to_dict(orient="records"),
    "predictions": {{
        "val": {{"y_true": y_val.tolist(), "y_pred": val_pred.tolist()}},
        "test": {{"y_true": y_test.tolist(), "y_pred": test_pred.tolist()}},
    }},
    "trained_at": datetime.now(timezone.utc).isoformat(),
}}

with open(pred_path, "w") as f:
    json.dump(payload, f, indent=2)

print("Saved baseline artifacts:")
print(f"- {{best_model_path}}")
print(f"- {{final_model_path}}")
print(f"- {{pred_path}}")
print("Train 01a–01e models next; they compare against this baseline automatically.")
'''


def _train_cell(spec: NotebookSpec) -> str:
    arch_label = ARCHITECTURE_LABELS[spec.architecture]
    task = JOB_TASK_KIND[spec.job_slug]
    comparison = _baseline_comparison_block(task)

    if task == "classification":
        if spec.architecture == "xgb":
            return f'''{NOTEBOOK_IMPORT_BOOTSTRAP}from tf_notebook_utils import (
    train_hist_gradient_boosting_classifier,
    as_1d_predictions,
    as_class_predictions,
    classification_metrics,
    make_single_epoch_history,
)

BATCH_SIZE = 512 if device_info["using_gpu"] else 1024
MAX_TREE_TRAIN = 80_000 if RUN_MODE == "fast_dev" else None
print(f"Training tree classifier ({{ARCHITECTURE}}) — {arch_label}")
if MAX_TREE_TRAIN:
    print(f"fast_dev: subsampling tree training to {{MAX_TREE_TRAIN:,}} rows (full val/test kept)")
model = train_hist_gradient_boosting_classifier(
    X_train, y_train, X_val, y_val, seed=SEED, max_train_samples=MAX_TREE_TRAIN
)
print(f"Tree backend: {{getattr(model, 'backend', 'unknown')}}")

val_probs = as_1d_predictions(model.predict(X_val, batch_size=BATCH_SIZE, verbose=0))
test_probs = as_1d_predictions(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0))
val_pred = as_class_predictions(val_probs)
test_pred = as_class_predictions(test_probs)

val_metrics = classification_metrics(y_val, val_pred)
crisis_metrics = classification_metrics(y_test, test_pred)
val_macro_f1 = val_metrics["macro_f1"]
crisis_macro_f1 = crisis_metrics["macro_f1"]
best_epoch = 1
best_val_macro_f1 = val_macro_f1
history = make_single_epoch_history(1.0 - val_macro_f1)

print(f"Val macro-F1: {{val_macro_f1:.4f}} | accuracy: {{val_metrics['accuracy']:.4f}}")
print(f"Crisis macro-F1: {{crisis_macro_f1:.4f}} | accuracy: {{crisis_metrics['accuracy']:.4f}}")
{comparison}'''

        return f'''{NOTEBOOK_IMPORT_BOOTSTRAP}from tf_notebook_utils import (
    build_classifier,
    make_tf_datasets,
    as_1d_predictions,
    as_class_predictions,
    classification_metrics,
)

BATCH_SIZE = 512 if device_info["using_gpu"] else 1024
EPOCHS = TRAIN_EPOCHS
PATIENCE = 15
LR = 1e-3

n_features = X_train.shape[-1]
lookback = X_train.shape[1]

train_ds, val_ds = make_tf_datasets(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE)
model = build_classifier(ARCHITECTURE, lookback, n_features, lr=LR)
model.summary()

steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
print(f"Run mode: {{RUN_MODE}} | Architecture: {{ARCHITECTURE}} | Epoch cap: {{EPOCHS}}")
print(f"Steps/epoch: {{steps_per_epoch:,}} | Batch size: {{BATCH_SIZE}}")
if device_info.get("using_gpu"):
    print(
        "GPU note: first epoch can take 10–30+ min on RTX 5090 while TF JIT-compiles CUDA kernels. "
        "This is normal — wait for epoch 1/30 output; later epochs are faster."
    )

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-5,
        verbose=1,
    ),
]

with tf.device("/GPU:0" if device_info["using_gpu"] else "/CPU:0"):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

val_probs = as_1d_predictions(model.predict(X_val, batch_size=BATCH_SIZE, verbose=0))
test_probs = as_1d_predictions(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0))
val_pred = as_class_predictions(val_probs)
test_pred = as_class_predictions(test_probs)

val_metrics = classification_metrics(y_val, val_pred)
crisis_metrics = classification_metrics(y_test, test_pred)
val_macro_f1 = val_metrics["macro_f1"]
crisis_macro_f1 = crisis_metrics["macro_f1"]

best_epoch = int(np.argmax(history.history["val_auc"]) + 1)
best_val_macro_f1 = val_macro_f1

print(f"Best epoch: {{best_epoch}} | Val macro-F1: {{val_macro_f1:.4f}}")
print(f"Crisis macro-F1: {{crisis_macro_f1:.4f}}")
{comparison}'''

    if spec.architecture == "xgb":
        return f'''{NOTEBOOK_IMPORT_BOOTSTRAP}from tf_notebook_utils import (
    make_tf_datasets,
    persistence_mae,
    train_hist_gradient_boosting,
    as_1d_predictions,
    make_single_epoch_history,
    asymmetric_error_metrics,
)

BATCH_SIZE = 512 if device_info["using_gpu"] else 1024
EPOCHS = TRAIN_EPOCHS
PATIENCE = 15
LR = 1e-3
LOSS = "absoluteerror"  # xgboost objective (reg:absoluteerror)

n_features = X_train.shape[-1]
lookback = X_train.shape[1]

MAX_TREE_TRAIN = 80_000 if RUN_MODE == "fast_dev" else None
print(f"Training tree model ({{ARCHITECTURE}}) — {arch_label}")
if MAX_TREE_TRAIN:
    print(f"fast_dev: subsampling tree training to {{MAX_TREE_TRAIN:,}} rows (full val/test kept)")
model = train_hist_gradient_boosting(
    X_train, y_train, X_val, y_val, seed=SEED, max_train_samples=MAX_TREE_TRAIN
)
print(f"Tree backend: {{getattr(model, 'backend', 'unknown')}}")

val_pred = as_1d_predictions(model.predict(X_val, batch_size=BATCH_SIZE, verbose=0))
test_pred = as_1d_predictions(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0))

val_mae = float(np.mean(np.abs(val_pred - y_val)))
val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
test_mae = float(np.mean(np.abs(test_pred - y_test)))
test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))

best_epoch = 1
best_val_mae = val_mae
history = make_single_epoch_history(val_mae)

persistence_mae_val = persistence_mae(X_val, y_val, bundle.feature_cols, bundle.scaler)
persistence_mae_test = persistence_mae(X_test, y_test, bundle.feature_cols, bundle.scaler)
train_mean = float(np.mean(y_train))
train_mean_mae_val = float(np.mean(np.abs(train_mean - y_val)))
train_mean_mae_test = float(np.mean(np.abs(train_mean - y_test)))

asym_val = asymmetric_error_metrics(y_val, val_pred)
asym_crisis = asymmetric_error_metrics(y_test, test_pred)

print(f"Val  MAE: {{val_mae:.6f}} | RMSE: {{val_rmse:.6f}}")
print(f"Crisis MAE (Test 2020 Q1): {{test_mae:.6f}} | RMSE: {{test_rmse:.6f}}")
print(
    f"Crisis asymmetry: under-MAE {{asym_crisis['under_mae']:.6f}} vs over-MAE "
    f"{{asym_crisis['over_mae']:.6f}} (under-prediction rate {{asym_crisis['under_rate']:.1%}})"
)
print("Baselines:")
print(f"  Persistence MAE (val):  {{persistence_mae_val:.6f}}")
print(f"  Persistence MAE (test): {{persistence_mae_test:.6f}}")
print(f"  Train-mean MAE (test):  {{train_mean_mae_test:.6f}}")
{comparison}
'''

    return f'''{NOTEBOOK_IMPORT_BOOTSTRAP}from tf_notebook_utils import (
    build_regressor,
    make_tf_datasets,
    persistence_mae,
    as_1d_predictions,
    asymmetric_error_metrics,
)

BATCH_SIZE = 512 if device_info["using_gpu"] else 1024
EPOCHS = TRAIN_EPOCHS
PATIENCE = 15
LR = 1e-3
LOSS = "huber"  # "huber" | "qlike" | "mse" — qlike penalizes vol underprediction harder

n_features = X_train.shape[-1]
lookback = X_train.shape[1]

ARCH_KWARGS = {{}}
if ARCHITECTURE == "garchgru":
    ARCH_KWARGS["return_idx"] = bundle.feature_cols.index("log_return")

train_ds, val_ds = make_tf_datasets(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE)
model = build_regressor(ARCHITECTURE, lookback, n_features, lr=LR, loss=LOSS, **ARCH_KWARGS)
model.summary()

steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
print(f"Run mode: {{RUN_MODE}} | Architecture: {{ARCHITECTURE}} | Epoch cap: {{EPOCHS}}")
print(f"Steps/epoch: {{steps_per_epoch:,}} | Batch size: {{BATCH_SIZE}}")
if device_info.get("using_gpu"):
    print(
        "GPU note: first epoch can take 10–30+ min on RTX 5090 while TF JIT-compiles CUDA kernels. "
        "This is normal — wait for epoch 1/30 output; later epochs are faster."
    )

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=PATIENCE,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-5,
        verbose=1,
    ),
]

with tf.device("/GPU:0" if device_info["using_gpu"] else "/CPU:0"):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

val_pred = as_1d_predictions(model.predict(X_val, batch_size=BATCH_SIZE, verbose=0))
test_pred = as_1d_predictions(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0))

val_mae = float(np.mean(np.abs(val_pred - y_val)))
val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
test_mae = float(np.mean(np.abs(test_pred - y_test)))
test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))

best_epoch = int(np.argmin(history.history["val_mae"]) + 1)
best_val_mae = float(np.min(history.history["val_mae"]))

persistence_mae_val = persistence_mae(X_val, y_val, bundle.feature_cols, bundle.scaler)
persistence_mae_test = persistence_mae(X_test, y_test, bundle.feature_cols, bundle.scaler)
train_mean = float(np.mean(y_train))
train_mean_mae_val = float(np.mean(np.abs(train_mean - y_val)))
train_mean_mae_test = float(np.mean(np.abs(train_mean - y_test)))

asym_val = asymmetric_error_metrics(y_val, val_pred)
asym_crisis = asymmetric_error_metrics(y_test, test_pred)

print(f"Best epoch: {{best_epoch}} | Best val MAE: {{best_val_mae:.6f}}")
print(f"Val  MAE: {{val_mae:.6f}} | RMSE: {{val_rmse:.6f}}")
print(f"Crisis MAE (Test 2020 Q1): {{test_mae:.6f}} | RMSE: {{test_rmse:.6f}}")
print(
    f"Crisis asymmetry: under-MAE {{asym_crisis['under_mae']:.6f}} vs over-MAE "
    f"{{asym_crisis['over_mae']:.6f}} (under-prediction rate {{asym_crisis['under_rate']:.1%}})"
)
print("Baselines:")
print(f"  Persistence MAE (val):  {{persistence_mae_val:.6f}}")
print(f"  Persistence MAE (test): {{persistence_mae_test:.6f}}")
print(f"  Train-mean MAE (test):  {{train_mean_mae_test:.6f}}")
{comparison}
'''


def _save_cell(spec: NotebookSpec) -> str:
    if spec.is_baseline:
        return _baseline_save_cell(spec)
    ext = "joblib" if spec.architecture in ("xgb", "baseline") else "keras"
    if JOB_TASK_KIND[spec.job_slug] == "classification":
        return f'''artifact_stem = f"{{ARCHITECTURE}}_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}_tf"
best_model_path = MODELS_DIR / f"{{artifact_stem}}_best.{ext}"
final_model_path = MODELS_DIR / f"{{artifact_stem}}_final.{ext}"
pred_path = MODELS_DIR / f"{{artifact_stem}}_predictions.json"

model.save(best_model_path)
model.save(final_model_path)

payload = {{
    "job_id": JOB_ID,
    "job_slug": JOB_SLUG,
    "architecture": ARCHITECTURE,
    "universe": UNIVERSE,
    "model": f"{{ARCHITECTURE}}_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}_tensorflow",
    "run_mode": RUN_MODE,
    "dataset_tag": DATASET_TAG,
    "n_symbols": bundle.cfg.get("n_symbols"),
    "framework": "tensorflow" if ARCHITECTURE != "xgb" else getattr(model, "backend", "tree"),
    "target_type": bundle.cfg["target_type"],
    "n_features": len(bundle.feature_cols),
    "lookback_days": bundle.cfg["lookback_days"],
    "feature_cols": bundle.feature_cols,
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_val_macro_f1,
    "val_macro_f1": val_macro_f1,
    "val_accuracy": val_metrics["accuracy"],
    "crisis_macro_f1": crisis_macro_f1,
    "crisis_accuracy": crisis_metrics["accuracy"],
    "baseline_comparison": baseline_comparison,
    "train_config": {{
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS if ARCHITECTURE != "xgb" else 1,
        "patience": PATIENCE if ARCHITECTURE != "xgb" else 0,
        "lr": LR if ARCHITECTURE != "xgb" else 0.0,
        "loss": "binary_crossentropy",
        "architecture": ARCHITECTURE,
        "device": device_info,
        "sequences_name": sequences_name,
    }},
    "history": hist_df.to_dict(orient="records"),
    "predictions": {{
        "val": {{"y_true": y_val.tolist(), "y_pred": val_pred.tolist(), "y_prob": val_probs.tolist()}},
        "test": {{"y_true": y_test.tolist(), "y_pred": test_pred.tolist(), "y_prob": test_probs.tolist()}},
    }},
    "trained_at": datetime.now(timezone.utc).isoformat(),
}}

with open(pred_path, "w") as f:
    json.dump(payload, f, indent=2)

print("Saved artifacts:")
print(f"- {{best_model_path}}")
print(f"- {{final_model_path}}")
print(f"- {{pred_path}}")
'''
    return f'''artifact_stem = f"{{ARCHITECTURE}}_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}_tf"
best_model_path = MODELS_DIR / f"{{artifact_stem}}_best.{ext}"
final_model_path = MODELS_DIR / f"{{artifact_stem}}_final.{ext}"
pred_path = MODELS_DIR / f"{{artifact_stem}}_predictions.json"

model.save(best_model_path)
model.save(final_model_path)

payload = {{
    "job_id": JOB_ID,
    "job_slug": JOB_SLUG,
    "architecture": ARCHITECTURE,
    "universe": UNIVERSE,
    "model": f"{{ARCHITECTURE}}_{{JOB_SLUG}}_{{UNIVERSE}}{{MODEL_TAG}}_tensorflow",
    "run_mode": RUN_MODE,
    "dataset_tag": DATASET_TAG,
    "n_symbols": bundle.cfg.get("n_symbols"),
    "framework": "tensorflow" if ARCHITECTURE != "xgb" else getattr(model, "backend", "tree"),
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
    "asymmetric_errors": {{
        "val": asym_val,
        "crisis": asym_crisis,
    }},
    "baselines": {{
        "persistence_mae_val": persistence_mae_val,
        "persistence_mae_test": persistence_mae_test,
        "train_mean_mae_val": train_mean_mae_val,
        "train_mean_mae_test": train_mean_mae_test,
    }},
    "baseline_comparison": baseline_comparison,
    "train_config": {{
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "loss": LOSS,
        "architecture": ARCHITECTURE,
        "device": device_info,
        "sequences_name": sequences_name,
    }},
    "history": hist_df.to_dict(orient="records"),
    "predictions": {{
        "val": {{"y_true": y_val.tolist(), "y_pred": val_pred.tolist()}},
        "test": {{"y_true": y_test.tolist(), "y_pred": test_pred.tolist()}},
    }},
    "trained_at": datetime.now(timezone.utc).isoformat(),
}}

with open(pred_path, "w") as f:
    json.dump(payload, f, indent=2)

print("Saved artifacts:")
print(f"- {{best_model_path}}")
print(f"- {{final_model_path}}")
print(f"- {{pred_path}}")
'''


def _evaluate_cell(spec: NotebookSpec) -> str:
    task = JOB_TASK_KIND[spec.job_slug]
    model_label = (
        f"Baseline ({spec.universe})"
        if spec.is_baseline
        else f"{spec.architecture.upper()} ({spec.universe})"
    )
    if task == "classification":
        return f'''hist_df = pd.DataFrame(
    {{
        "epoch": np.arange(1, len(history.history["loss"]) + 1),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "val_score": history.history.get(
            "val_auc",
            history.history.get("val_mae", [0.0]),
        ),
    }}
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
axes[1].set_ylabel("Val AUC" if "val_auc" in history.history else "Val score")
axes[1].set_title("Validation score")
fig.suptitle("{model_label} — vol regime training", y=1.02)
plt.tight_layout()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true.astype(int), y_pred.astype(int)),
        display_labels=["low vol", "high vol"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

plot_confusion(y_val, val_pred, "{model_label} — validation")
plot_confusion(y_test, test_pred, "{model_label} — COVID test")

from tf_notebook_utils import classification_metrics

majority_class = int(round(float(np.mean(y_train))))
majority_val = classification_metrics(
    y_val, np.full(len(y_val), majority_class, dtype=np.float32)
)
majority_crisis = classification_metrics(
    y_test, np.full(len(y_test), majority_class, dtype=np.float32)
)

table = pd.DataFrame(
    [
        {{
            "Model": "{model_label}",
            "Val macro-F1": val_macro_f1,
            "Crisis macro-F1": crisis_macro_f1,
            "Val accuracy": val_metrics["accuracy"],
        }},
        {{
            "Model": "Majority class (train)",
            "Val macro-F1": majority_val["macro_f1"],
            "Crisis macro-F1": majority_crisis["macro_f1"],
            "Val accuracy": majority_val["accuracy"],
        }},
    ]
)
display(table)
'''

    return f'''hist_df = pd.DataFrame(
    {{
        "epoch": np.arange(1, len(history.history["loss"]) + 1),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "val_mae": history.history["val_mae"],
    }}
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
fig.suptitle("{model_label} — training", y=1.02)
plt.tight_layout()
plt.show()

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
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
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

plot_predictions(y_val, val_pred, "{model_label} — validation")
plot_predictions(y_test, test_pred, "{model_label} — COVID test")

table = pd.DataFrame(
    [
        {{
            "Model": "{model_label}",
            "Val MAE": val_mae,
            "Crisis MAE": test_mae,
            "Val RMSE": val_rmse,
        }},
        {{
            "Model": "Persistence (val)",
            "Val MAE": persistence_mae_val,
            "Crisis MAE": persistence_mae_test,
            "Val RMSE": np.nan,
        }},
        {{
            "Model": "Train-mean (test)",
            "Val MAE": train_mean_mae_val,
            "Crisis MAE": train_mean_mae_test,
            "Val RMSE": np.nan,
        }},
    ]
)
display(table)
'''


def _scaffold_notebook(spec: NotebookSpec) -> dict:
    if spec.status == "scaffold":
        cells = [
            _md_cell(_title_cell(spec)),
            _md_cell(
                f"## Scaffold — implement in Phase {spec.phase}\n\n"
                f"This notebook is reserved for **{spec.job_title}** (`{spec.architecture}` on `{spec.universe}`).\n\n"
                f"- **Target type (planned):** `{spec.target_type}`\n"
                f"- **Champion metric:** `{spec.primary_metric}`\n"
                f"- **Blocked on:** preprocessing support for `{spec.target_type}`\n\n"
                + (
                    "When active, this is the **trainable baseline** for the job. Run it before `01a`–`01e`.\n\n"
                    if spec.is_baseline
                    else ""
                )
                + "Regenerate with `scripts/scaffold_ml_iteration2_notebooks.py` after preprocessing targets land."
            ),
            _code_cell(_config_cell(spec)),
        ]
        return _apply_kernel_metadata(
            {
            "cells": cells,
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        )

    template = json.loads(TEMPLATE_PATH.read_text())
    nb = copy.deepcopy(template)
    nb["cells"][0]["source"] = _title_cell(spec).splitlines(keepends=True)
    nb["cells"][2]["source"] = _config_cell(spec).splitlines(keepends=True)
    nb["cells"][5]["source"] = (
        f"## 2) Preprocess {spec.universe} ({spec.asset_type}, plus5 profile)\n\n"
        f"Reuse existing artifacts when available. **Fast-dev** builds `dev{{N}}` files via `--max-symbols`."
    ).splitlines(keepends=True)
    nb["cells"][6]["source"] = _preprocess_cell(spec).splitlines(keepends=True)
    if spec.is_baseline:
        nb["cells"][9]["source"] = (
            f"## 4) Train baseline model\n\n"
            f"{ARCHITECTURE_LABELS['baseline']}. Compare all other models in this folder against the saved metrics."
        ).splitlines(keepends=True)
        nb["cells"][10]["source"] = _baseline_train_cell(spec).splitlines(keepends=True)
        nb["cells"][14]["source"] = _baseline_save_cell(spec).splitlines(keepends=True)
    else:
        task = JOB_TASK_KIND[spec.job_slug]
        loss_note = (
            "Binary cross-entropy, early stopping on validation AUC."
            if task == "classification"
            else "Huber loss, early stopping on validation MAE."
        )
        nb["cells"][9]["source"] = (
            f"## 4) Build and train {spec.architecture.upper()}\n\n"
            f"{ARCHITECTURE_LABELS[spec.architecture]}. {loss_note}"
        ).splitlines(keepends=True)
        nb["cells"][10]["source"] = _train_cell(spec).splitlines(keepends=True)
        nb["cells"][14]["source"] = _save_cell(spec).splitlines(keepends=True)
    eval_note = (
        "Plot training curves, confusion matrices, and a baseline comparison table."
        if JOB_TASK_KIND[spec.job_slug] == "classification"
        else "Plot training curves, predicted-vs-actual diagnostics, and a baseline comparison table."
    )
    nb["cells"][11]["source"] = (
        f"## 5) Evaluate and visualize\n\n{eval_note}"
    ).splitlines(keepends=True)
    nb["cells"][12]["source"] = _evaluate_cell(spec).splitlines(keepends=True)
    # Clear executed outputs for a clean scaffold
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return _apply_kernel_metadata(nb)


def _orchestration_notebook() -> dict:
    return _apply_kernel_metadata(
        {
            "cells": [
                _md_cell(
                    "# 00a — Product Score Orchestration\n\n"
                    "Merge champion model outputs from each job into app-ready scores.\n\n"
                    "**Inputs:** `data/models/*_predictions.json` from J01–J04 champions\n\n"
                    "**Output:** composite `{risk_score, regime, rel_rank, coach_action}`"
                ),
                _code_cell(
                    '"""Load champion prediction JSONs and merge — implement after J01 bake-off."""\n'
                    "from pathlib import Path\n"
                    "import json\n\n"
                    "MODELS_DIR = Path('../../data/models')\n"
                    "CHAMPION_FILES = {\n"
                    '    "vol_risk_stocks": None,\n'
                    '    "vol_risk_etfs": None,\n'
                    '    "vol_regime_market": None,\n'
                    '    "rel_strength_stocks": None,\n'
                    '    "etf_opportunity_etfs": None,\n'
                    "}\n"
                    "print('Orchestration scaffold — wire champions after bake-offs.')\n"
                ),
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ML Iteration 2 notebooks from the J01 template."
    )
    parser.add_argument(
        "--job-id",
        help="Only regenerate notebooks for this job id (e.g. 02)",
    )
    parser.add_argument(
        "--job-slug",
        help="Only regenerate notebooks for this job slug (e.g. vol_regime)",
    )
    parser.add_argument(
        "--universe",
        choices=["stocks", "etfs", "market"],
        help="Further limit regeneration to one universe folder",
    )
    parser.add_argument(
        "--architecture",
        help="Only regenerate notebooks for one architecture (e.g. garchgru) — "
        "use to add new candidates without touching trained notebooks",
    )
    parser.add_argument(
        "--include-orchestration",
        action="store_true",
        help="Also rewrite 00_orchestration/00a_ensemble_product_scores.ipynb",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would be written without modifying files",
    )
    args = parser.parse_args()

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

    specs = filter_notebook_specs(
        all_notebook_specs_including_baselines(),
        job_id=args.job_id,
        job_slug=args.job_slug,
        universe=args.universe,
        architecture=args.architecture,
    )
    if not specs and not args.include_orchestration:
        raise SystemExit(
            "No notebooks matched filters. "
            "Check --job-id / --job-slug / --universe."
        )

    created = 0
    for spec in specs:
        path = notebook_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(f"Would write {path.relative_to(ML_DIR.parent)}")
            created += 1
            continue
        nb = _scaffold_notebook(spec)
        path.write_text(json.dumps(nb, indent=1))
        print(f"Wrote {path.relative_to(ML_DIR.parent)}")
        created += 1

    write_orch = args.include_orchestration or (
        not args.job_id and not args.job_slug and not args.universe and not args.architecture
    )
    if write_orch:
        orch_dir = ML_DIR / "00_orchestration"
        orch_dir.mkdir(parents=True, exist_ok=True)
        orch_path = orch_dir / "00a_ensemble_product_scores.ipynb"
        if args.dry_run:
            print(f"Would write {orch_path.relative_to(ML_DIR.parent)}")
        else:
            orch_path.write_text(json.dumps(_orchestration_notebook(), indent=1))
            print(f"Wrote {orch_path.relative_to(ML_DIR.parent)}")
        created += 1

    scope = []
    if args.job_slug:
        scope.append(f"job_slug={args.job_slug}")
    if args.job_id:
        scope.append(f"job_id={args.job_id}")
    if args.universe:
        scope.append(f"universe={args.universe}")
    scope_note = f" ({', '.join(scope)})" if scope else " (all jobs)"
    print(f"\nTotal notebooks: {created}{scope_note}")


if __name__ == "__main__":
    main()
