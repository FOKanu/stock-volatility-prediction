"""TensorFlow GRU notebook utilities (GPU-first, no PyTorch dependency)."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def _configure_cuda_library_path() -> list[str]:
    """Expose pip-installed NVIDIA libs to the dynamic linker (TF 2.21+ hermetic CUDA)."""
    if os.name == "nt":
        return []

    site_packages = Path(site.getsitepackages()[0])
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.exists():
        return []

    lib_dirs = sorted({str(so.parent.resolve()) for so in nvidia_root.glob("*/lib/*.so*")})
    if not lib_dirs:
        return []

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = lib_dirs + ([existing] if existing else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    return lib_dirs


def bootstrap_notebook(seed: int = 42):
    """Add scripts to path and return (project_root, models_dir, dataset_dir)."""
    from paths import find_project_root, resolve_dataset_dir

    root = find_project_root()
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    _configure_cuda_library_path()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    models_dir = root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = resolve_dataset_dir(root)
    return root, models_dir, dataset_dir, tf


def configure_tf_gpu(tf, enable_mixed_precision: bool = True) -> dict:
    """Enable GPU memory growth and optional mixed precision. Returns device summary."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    policy_name = "float32"
    if gpus and enable_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        policy_name = "mixed_float16"

    return {
        "gpus": [d.name for d in gpus],
        "using_gpu": bool(gpus),
        "mixed_precision": policy_name,
        "platform_note": (
            "Native Windows TensorFlow is CPU-only (TF >= 2.11). "
            "Select the WSL kernel 'Python (TF GPU WSL)' to train on your RTX 5090."
            if not gpus and os.name == "nt"
            else ""
        ),
    }


def make_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    shuffle_buffer: int = 100_000,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build prefetched tf.data pipelines for GPU training."""
    autotune = tf.data.AUTOTUNE

    def _dataset(X: np.ndarray, y: np.ndarray, train: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if train:
            ds = ds.shuffle(shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(autotune)

    return _dataset(X_train, y_train, True), _dataset(X_val, y_val, False)



def persistence_mae(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    scaler: dict,
) -> float:
    idx = feature_cols.index("abs_return")
    mean = scaler["mean"][idx]
    std = scaler["std"][idx]
    persist_pred = X[:, -1, idx] * std + mean
    return float(np.abs(persist_pred - y).mean())


def _compile_regressor(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_lstm_model(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.LSTM(64, dropout=0.2)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr)


def _tcn_residual_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
    l2 = tf.keras.regularizers.l2(1e-4)
    residual = x
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=False,
        kernel_regularizer=l2,
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SpatialDropout1D(dropout)(x)
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=False,
        kernel_regularizer=l2,
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SpatialDropout1D(dropout)(x)
    if int(residual.shape[-1]) != filters:
        residual = tf.keras.layers.Conv1D(
            filters, 1, padding="same", use_bias=False, kernel_regularizer=l2
        )(residual)
    return tf.keras.layers.Add()([x, residual])


def build_tcn_model(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(32, 1, padding="same", use_bias=False, kernel_regularizer=l2)(inputs)
    for dilation in (1, 2, 4, 8):
        x = _tcn_residual_block(x, 32, 2, dilation, 0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr)


def build_cnn1d_model(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr)


def build_gru_model(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    """2-layer GRU regressor matching the PyTorch universe model."""
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.GRU(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.GRU(64, dropout=0.2)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr)


def build_regressor(architecture: str, lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    builders = {
        "gru": build_gru_model,
        "tcn": build_tcn_model,
        "lstm": build_lstm_model,
        "cnn1d": build_cnn1d_model,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture {architecture!r}; expected one of {sorted(builders)}")
    return builders[architecture](lookback, n_features, lr)


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten (n, lookback, features) -> (n, lookback * features) for tree models."""
    n, lookback, n_features = X.shape
    return X.reshape(n, lookback * n_features)


def as_1d_predictions(pred: np.ndarray) -> np.ndarray:
    """Flatten model output to (n,) whether predict returned (n,) or (n, 1)."""
    return np.asarray(pred, dtype=np.float32).reshape(-1)


def make_single_epoch_history(val_mae: float):
    """Keras-compatible history stub for non-neural models (baseline, trees)."""

    class _History:
        def __init__(self, mae: float):
            self.history = {
                "loss": [0.0],
                "val_loss": [0.0],
                "val_mae": [float(mae)],
            }

    return _History(val_mae)


def last_timestep_features(X: np.ndarray) -> np.ndarray:
    """Use the most recent day in each sequence (n, features)."""
    return X[:, -1, :].astype(np.float32)


def persistence_predict(
    X: np.ndarray,
    feature_cols: list[str],
    scaler: dict,
) -> np.ndarray:
    """Predict tomorrow |return| as today's denormalized abs_return (no training)."""
    idx = feature_cols.index("abs_return")
    mean = scaler["mean"][idx]
    std = scaler["std"][idx]
    return (X[:, -1, idx] * std + mean).astype(np.float32)


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    seed: int = 42,
):
    """Trainable baseline: Ridge (regression) or LogisticRegression (classification) on last day."""
    from sklearn.linear_model import LogisticRegression, Ridge

    X_tr = last_timestep_features(X_train)

    if task == "classification":
        sk_model = LogisticRegression(max_iter=1000, random_state=seed)
        sk_model.fit(X_tr, y_train.astype(int))
        backend = "logistic_last_day"
    else:
        sk_model = Ridge(alpha=1.0, random_state=seed)
        sk_model.fit(X_tr, y_train)
        backend = "ridge_last_day"

    class _Wrapper:
        def predict(self, X: np.ndarray, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
            feats = last_timestep_features(X)
            pred = sk_model.predict(feats)
            return as_1d_predictions(pred)

        def save(self, path) -> None:
            import joblib

            joblib.dump({"model": sk_model, "backend": backend, "task": task}, path)

    wrapper = _Wrapper()
    wrapper.backend = backend
    wrapper.task = task
    wrapper.sk_model = sk_model
    return wrapper


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def _subsample_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(X) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max_samples, replace=False)
    return X[idx], y[idx]


def _compile_classifier(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _dense_classifier_head(x: tf.Tensor) -> tf.Tensor:
    return tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="target")(x)


def build_lstm_classifier(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.LSTM(64, dropout=0.2)(x)
    outputs = _dense_classifier_head(x)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr)


def build_gru_classifier(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.GRU(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.GRU(64, dropout=0.2)(x)
    outputs = _dense_classifier_head(x)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr)


def build_tcn_classifier(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(32, 1, padding="same", use_bias=False, kernel_regularizer=l2)(inputs)
    for dilation in (1, 2, 4, 8):
        x = _tcn_residual_block(x, 32, 2, dilation, 0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = _dense_classifier_head(x)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr)


def build_cnn1d_classifier(lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = _dense_classifier_head(x)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr)


def build_classifier(architecture: str, lookback: int, n_features: int, lr: float) -> tf.keras.Model:
    builders = {
        "gru": build_gru_classifier,
        "tcn": build_tcn_classifier,
        "lstm": build_lstm_classifier,
        "cnn1d": build_cnn1d_classifier,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture {architecture!r}; expected one of {sorted(builders)}")
    return builders[architecture](lookback, n_features, lr)


def as_class_predictions(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = np.asarray(pred, dtype=np.float32).reshape(-1)
    return (probs >= threshold).astype(np.float32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)
    metrics = {
        "accuracy": float(np.mean(y_true_i == y_pred_i)),
        "macro_f1": float(f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_i, y_pred))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_hist_gradient_boosting_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_train_samples: int | None = None,
):
    """Train a tree classifier; uses xgboost when installed, else sklearn HistGBC."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_tr, y_tr = _subsample_xy(X_train, y_train, max_train_samples, seed)
    X_va = flatten_sequences(X_val)
    X_tr = flatten_sequences(X_tr)
    y_tr_i = y_tr.astype(int)
    y_val_i = y_val.astype(int)

    print(
        f"Tree training rows: {len(X_tr):,} (features={X_tr.shape[1]:,}) | "
        f"val rows: {len(X_va):,}"
    )

    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=200 if max_train_samples else 300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=25,
            tree_method="hist",
        )
        model.fit(X_tr, y_tr_i, eval_set=[(X_va, y_val_i)], verbose=False)
        backend = "xgboost"
    except ImportError:
        print("xgboost not installed — using sklearn HistGradientBoostingClassifier (slower).")
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200 if max_train_samples else 300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=25,
        )
        model.fit(X_tr, y_tr_i, X_val=X_va, y_val=y_val_i)
        backend = "sklearn_hist_gbc"

    class _Wrapper:
        def predict(self, X: np.ndarray, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
            flat = flatten_sequences(X)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(flat)[:, 1]
            else:
                probs = model.predict(flat)
            return probs.astype(np.float32)

        def save(self, path) -> None:
            import joblib

            joblib.dump({"model": model, "backend": backend}, path)

    wrapper = _Wrapper()
    wrapper.backend = backend
    wrapper.sk_model = model
    return wrapper


def train_hist_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_train_samples: int | None = None,
):
    """Train a tree regressor; uses xgboost when installed, else sklearn HistGBR."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_tr, y_tr = _subsample_xy(X_train, y_train, max_train_samples, seed)
    X_va = flatten_sequences(X_val)
    X_tr = flatten_sequences(X_tr)

    print(
        f"Tree training rows: {len(X_tr):,} (features={X_tr.shape[1]:,}) | "
        f"val rows: {len(X_va):,}"
    )

    try:
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=200 if max_train_samples else 300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:absoluteerror",
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=25,
            tree_method="hist",
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_val)],
            verbose=False,
        )
        backend = "xgboost"
    except ImportError:
        print("xgboost not installed — using sklearn HistGradientBoostingRegressor (slower).")
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200 if max_train_samples else 300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=25,
        )
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_val)
        backend = "sklearn_hist_gbr"

    class _Wrapper:
        def predict(self, X: np.ndarray, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
            flat = flatten_sequences(X)
            return as_1d_predictions(model.predict(flat))

        def save(self, path) -> None:
            import joblib

            joblib.dump({"model": model, "backend": backend}, path)

    wrapper = _Wrapper()
    wrapper.backend = backend
    wrapper.sk_model = model
    return wrapper
