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
    train_sample_weight: np.ndarray | None = None,
    disk_block_rows: int = 65_536,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build prefetched pipelines without materializing memmaps in RAM."""
    autotune = tf.data.AUTOTUNE

    def _dataset(X: np.ndarray, y: np.ndarray, train: bool) -> tf.data.Dataset:
        if isinstance(X, np.memmap) or isinstance(y, np.memmap):
            n_rows = len(y)
            n_batches = sum(
                int(np.ceil((min(start + disk_block_rows, n_rows) - start) / batch_size))
                for start in range(0, n_rows, disk_block_rows)
            )

            def generator():
                rng = np.random.default_rng(42)
                block_starts = np.arange(0, n_rows, disk_block_rows)
                if train:
                    rng.shuffle(block_starts)
                for block_start in block_starts:
                    block_stop = min(int(block_start) + disk_block_rows, n_rows)
                    indices = np.arange(int(block_start), block_stop)
                    if train:
                        rng.shuffle(indices)
                    for offset in range(0, len(indices), batch_size):
                        batch_idx = indices[offset : offset + batch_size]
                        xb = np.asarray(X[batch_idx], dtype=np.float32)
                        yb = np.asarray(y[batch_idx], dtype=np.float32)
                        if train and train_sample_weight is not None:
                            wb = np.asarray(train_sample_weight[batch_idx], dtype=np.float32)
                            yield xb, yb, wb
                        else:
                            yield xb, yb

            x_spec = tf.TensorSpec(shape=(None, *X.shape[1:]), dtype=tf.float32)
            y_spec = tf.TensorSpec(shape=(None,), dtype=tf.float32)
            if train and train_sample_weight is not None:
                signature = (
                    x_spec,
                    y_spec,
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )
            else:
                signature = (x_spec, y_spec)
            ds = tf.data.Dataset.from_generator(generator, output_signature=signature)
            return ds.apply(tf.data.experimental.assert_cardinality(n_batches)).prefetch(autotune)

        tensors = (X, y)
        if train and train_sample_weight is not None:
            tensors = (X, y, train_sample_weight.astype(np.float32))
        ds = tf.data.Dataset.from_tensor_slices(tensors)
        if train:
            ds = ds.shuffle(shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(autotune)

    return _dataset(X_train, y_train, True), _dataset(X_val, y_val, False)


def predict_in_batches(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Predict over arrays or memmaps with only one contiguous batch resident."""
    outputs: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch = np.asarray(X[start:stop], dtype=np.float32)
        outputs.append(np.asarray(model.predict_on_batch(batch)))
    if not outputs:
        output_dim = int(model.output_shape[-1])
        return np.empty((0, output_dim), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def compute_class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    """Balanced class weights: n_samples / (n_classes * class_count)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for start in range(0, len(y), 1_000_000):
        counts += np.bincount(
            np.asarray(y[start : start + 1_000_000]).astype(int),
            minlength=n_classes,
        )
    safe_counts = np.maximum(counts, 1)
    weights = len(y) / (n_classes * safe_counts)
    return {cls: float(weight) for cls, weight in enumerate(weights)}


def class_weights_to_sample_weights(
    y: np.ndarray, class_weights: dict[int, float]
) -> np.ndarray:
    """Map class weights to a compact per-sample float32 vector."""
    lookup = np.array(
        [class_weights[c] for c in range(len(class_weights))], dtype=np.float32
    )
    weights = np.empty(len(y), dtype=np.float32)
    for start in range(0, len(y), 1_000_000):
        stop = min(start + 1_000_000, len(y))
        weights[start:stop] = lookup[np.asarray(y[start:stop]).astype(int)]
    return weights



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


@tf.keras.utils.register_keras_serializable(package="stock_prediction")
def qlike_loss(y_true, y_pred):
    """QLIKE loss on the variance scale for volatility targets.

    y_true / y_pred are on the |log return| (vol) scale; both are squared into
    variances. QLIKE is asymmetric: underpredicting volatility is penalized more
    than overpredicting it, which suits risk-control products.
    """
    eps = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    var_true = tf.square(y_true) + eps
    var_pred = tf.square(tf.nn.relu(y_pred)) + eps
    ratio = var_true / var_pred
    return ratio - tf.math.log(ratio) - 1.0


def _resolve_regression_loss(loss: str):
    losses = {
        "huber": tf.keras.losses.Huber(delta=1.0),
        "qlike": qlike_loss,
        "mse": tf.keras.losses.MeanSquaredError(),
    }
    if loss not in losses:
        raise ValueError(f"Unknown regression loss {loss!r}; expected one of {sorted(losses)}")
    return losses[loss]


def asymmetric_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Split MAE by error sign: underprediction (pred < true) is the risky side."""
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    under = err < 0
    over = ~under
    return {
        "under_mae": float(np.abs(err[under]).mean()) if under.any() else 0.0,
        "over_mae": float(np.abs(err[over]).mean()) if over.any() else 0.0,
        "under_rate": float(under.mean()),
    }


def _compile_regressor(model: tf.keras.Model, lr: float, loss: str = "huber") -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=_resolve_regression_loss(loss),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_lstm_model(lookback: int, n_features: int, lr: float, loss: str = "huber") -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.LSTM(64, dropout=0.2)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr, loss)


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


def build_tcn_model(lookback: int, n_features: int, lr: float, loss: str = "huber") -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(32, 1, padding="same", use_bias=False, kernel_regularizer=l2)(inputs)
    for dilation in (1, 2, 4, 8):
        x = _tcn_residual_block(x, 32, 2, dilation, 0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr, loss)


def build_cnn1d_model(lookback: int, n_features: int, lr: float, loss: str = "huber") -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr, loss)


def build_gru_model(lookback: int, n_features: int, lr: float, loss: str = "huber") -> tf.keras.Model:
    """2-layer GRU regressor matching the PyTorch universe model."""
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.GRU(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.GRU(64, dropout=0.2)(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32", name="target")(x)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr, loss)


@tf.keras.utils.register_keras_serializable(package="stock_prediction")
class GarchGruCell(tf.keras.layers.Layer):
    """GARCH(1,1)-style recurrent cell with learnable, constrained parameters.

    State h_t plays the role of conditional variance sigma_t^2:

        h_t = omega + alpha * x_{t-1}^2 + beta * h_{t-1}

    where x is the (standardized) return input. Constraints alpha, beta >= 0 and
    alpha + beta < 1 (covariance stationarity) are enforced with a softplus
    parameterization plus rescaling. A small GRU runs alongside the GARCH state so
    the model can also learn non-linear structure from the remaining features
    ("neural GARCH": the GRU can modulate the effective parameters per step).
    """

    def __init__(self, gru_units: int = 32, return_idx: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.gru_units = gru_units
        self.return_idx = return_idx
        self.gru_cell = tf.keras.layers.GRUCell(gru_units)
        # state: [garch_variance (1), gru_state (gru_units)]
        self.state_size = [1, gru_units]
        self.output_size = 1 + gru_units

    def build(self, input_shape):
        # Unconstrained GARCH parameters (softplus-transformed in call).
        # Inits target omega~0.05, alpha~0.10, beta~0.85: unconditional variance
        # omega/(1-alpha-beta) ~= 1, matching z-scored return inputs.
        self.raw_omega = self.add_weight(
            name="raw_omega", shape=(1,), initializer=tf.keras.initializers.Constant(-2.97)
        )
        self.raw_alpha = self.add_weight(
            name="raw_alpha", shape=(1,), initializer=tf.keras.initializers.Constant(-2.25)
        )
        self.raw_beta = self.add_weight(
            name="raw_beta", shape=(1,), initializer=tf.keras.initializers.Constant(0.29)
        )
        # Dynamic (neural-GARCH) modulation of alpha/beta from the GRU state.
        self.mod_dense = tf.keras.layers.Dense(
            2, activation="tanh", kernel_initializer="zeros", name="param_modulation"
        )
        self.gru_cell.build(input_shape)
        self.mod_dense.build((None, self.gru_units))
        super().build(input_shape)

    def garch_params(self):
        omega = tf.nn.softplus(self.raw_omega)
        alpha = tf.nn.softplus(self.raw_alpha)
        beta = tf.nn.softplus(self.raw_beta)
        # Rescale so alpha + beta < 1 (stationarity).
        total = alpha + beta
        scale = tf.where(total >= 0.999, 0.999 / total, tf.ones_like(total))
        return omega, alpha * scale, beta * scale

    def call(self, inputs, states):
        h_garch, h_gru = states
        inputs = tf.cast(inputs, h_gru.dtype)
        h_garch = tf.cast(h_garch, h_gru.dtype)

        gru_out, [h_gru_new] = self.gru_cell(inputs, [h_gru])

        omega, alpha, beta = self.garch_params()
        omega = tf.cast(omega, h_gru.dtype)
        alpha = tf.cast(alpha, h_gru.dtype)
        beta = tf.cast(beta, h_gru.dtype)

        # +-10% dynamic modulation of alpha/beta from sequence context.
        mod = self.mod_dense(gru_out) * 0.1
        alpha_t = alpha * (1.0 + mod[:, 0:1])
        beta_t = beta * (1.0 + mod[:, 1:2])

        x_ret = inputs[:, self.return_idx : self.return_idx + 1]
        h_garch_new = omega + alpha_t * tf.square(x_ret) + beta_t * h_garch

        garch_vol = tf.sqrt(h_garch_new + 1e-8)
        output = tf.concat([garch_vol, gru_out], axis=-1)
        return output, [h_garch_new, h_gru_new]

    def get_config(self):
        config = super().get_config()
        config.update({"gru_units": self.gru_units, "return_idx": self.return_idx})
        return config


def build_garchgru_model(
    lookback: int,
    n_features: int,
    lr: float,
    loss: str = "huber",
    return_idx: int = 0,
) -> tf.keras.Model:
    """GARCH-GRU hybrid: embedded GARCH(1,1) variance recursion + GRU features.

    return_idx must point at the log_return column of the feature tensor.
    """
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    cell = GarchGruCell(gru_units=32, return_idx=return_idx)
    x = tf.keras.layers.RNN(cell, return_sequences=False, name="garch_gru")(inputs)
    # Cell output = [garch_vol (1), gru_state (units)]; vol is refined by the GRU pathway.
    garch_vol = tf.keras.layers.Lambda(lambda t: t[:, :1], name="garch_vol")(x)
    gru_state = tf.keras.layers.Lambda(lambda t: t[:, 1:], name="gru_state")(x)
    # GARCH state evolves on the z-scored input scale; targets (raw |log return|)
    # live near 0.02, so calibrate with a learnable affine map initialized small.
    garch_vol = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.Constant(0.02),
        bias_initializer="zeros",
        name="vol_calibration",
    )(garch_vol)
    residual = tf.keras.layers.Dense(16, activation="relu")(gru_state)
    residual = tf.keras.layers.Dense(1, kernel_initializer="zeros", name="neural_residual")(
        residual
    )
    outputs = tf.keras.layers.Add(name="target_pre")([garch_vol, residual])
    outputs = tf.keras.layers.Activation("linear", dtype="float32", name="target")(outputs)
    return _compile_regressor(tf.keras.Model(inputs, outputs), lr, loss)


def build_regressor(
    architecture: str,
    lookback: int,
    n_features: int,
    lr: float,
    loss: str = "huber",
    **arch_kwargs,
) -> tf.keras.Model:
    builders = {
        "gru": build_gru_model,
        "tcn": build_tcn_model,
        "lstm": build_lstm_model,
        "cnn1d": build_cnn1d_model,
        "garchgru": build_garchgru_model,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture {architecture!r}; expected one of {sorted(builders)}")
    return builders[architecture](lookback, n_features, lr, loss=loss, **arch_kwargs)


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
    balanced: bool = False,
):
    """Train Ridge or LogisticRegression on last-day features."""
    from sklearn.linear_model import LogisticRegression, Ridge

    X_tr = last_timestep_features(X_train)

    if task == "classification":
        sk_model = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced" if balanced else None,
        )
        sk_model.fit(X_tr, y_train.astype(int))
        backend = "logistic_last_day"
    else:
        sk_model = Ridge(alpha=1.0, random_state=seed)
        sk_model.fit(X_tr, y_train)
        backend = "ridge_last_day"

    class _Wrapper:
        _default_batch_size = 65_536

        def _predict_chunk(self, X: np.ndarray) -> np.ndarray:
            feats = last_timestep_features(X)
            if task == "classification":
                probs = sk_model.predict_proba(feats)
                if probs.shape[1] == 2:
                    return probs[:, 1].astype(np.float32)
                return probs.astype(np.float32)
            return as_1d_predictions(sk_model.predict(feats))

        def predict(self, X: np.ndarray, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
            batch_size = batch_size or self._default_batch_size
            if len(X) <= batch_size:
                return self._predict_chunk(X)
            chunks = [self._predict_chunk(X[start : start + batch_size]) for start in range(0, len(X), batch_size)]
            if task == "classification" and getattr(sk_model, "classes_", np.array([])).shape[0] > 2:
                return np.concatenate(chunks, axis=0).astype(np.float32)
            return as_1d_predictions(np.concatenate(chunks, axis=0))

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


def _compile_classifier(model: tf.keras.Model, lr: float, n_classes: int = 2) -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    if n_classes > 2:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    return model


def _dense_classifier_head(x: tf.Tensor, n_classes: int = 2) -> tf.Tensor:
    if n_classes > 2:
        return tf.keras.layers.Dense(
            n_classes, activation="softmax", dtype="float32", name="target"
        )(x)
    return tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="target")(x)


def build_lstm_classifier(lookback: int, n_features: int, lr: float, n_classes: int = 2) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.LSTM(64, dropout=0.2)(x)
    outputs = _dense_classifier_head(x, n_classes)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr, n_classes)


def build_gru_classifier(lookback: int, n_features: int, lr: float, n_classes: int = 2) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.GRU(64, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.GRU(64, dropout=0.2)(x)
    outputs = _dense_classifier_head(x, n_classes)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr, n_classes)


def build_tcn_classifier(lookback: int, n_features: int, lr: float, n_classes: int = 2) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(32, 1, padding="same", use_bias=False, kernel_regularizer=l2)(inputs)
    for dilation in (1, 2, 4, 8):
        x = _tcn_residual_block(x, 32, 2, dilation, 0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = _dense_classifier_head(x, n_classes)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr, n_classes)


def build_cnn1d_classifier(lookback: int, n_features: int, lr: float, n_classes: int = 2) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="sequence")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = _dense_classifier_head(x, n_classes)
    return _compile_classifier(tf.keras.Model(inputs, outputs), lr, n_classes)


def build_classifier(
    architecture: str, lookback: int, n_features: int, lr: float, n_classes: int = 2
) -> tf.keras.Model:
    builders = {
        "gru": build_gru_classifier,
        "tcn": build_tcn_classifier,
        "lstm": build_lstm_classifier,
        "cnn1d": build_cnn1d_classifier,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture {architecture!r}; expected one of {sorted(builders)}")
    return builders[architecture](lookback, n_features, lr, n_classes)


def as_class_predictions(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 2 and pred.shape[1] > 1:  # softmax probabilities -> argmax
        return np.argmax(pred, axis=1).astype(np.float32)
    probs = pred.reshape(-1)
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
    n_classes: int = 2,
    balanced: bool = False,
):
    """Train a binary or multiclass tree classifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_tr, y_tr = _subsample_xy(X_train, y_train, max_train_samples, seed)
    X_va = flatten_sequences(X_val)
    X_tr = flatten_sequences(X_tr)
    y_tr_i = y_tr.astype(int)
    y_val_i = y_val.astype(int)
    sample_weight = None
    if balanced:
        class_weights = compute_class_weights(y_tr_i, n_classes)
        sample_weight = class_weights_to_sample_weights(y_tr_i, class_weights)

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
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            num_class=None if n_classes == 2 else n_classes,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=25,
            tree_method="hist",
        )
        model.fit(
            X_tr,
            y_tr_i,
            sample_weight=sample_weight,
            eval_set=[(X_va, y_val_i)],
            verbose=False,
        )
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
            class_weight="balanced" if balanced else None,
        )
        model.fit(X_tr, y_tr_i)
        backend = "sklearn_hist_gbc"

    class _Wrapper:
        def predict(self, X: np.ndarray, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
            flat = flatten_sequences(X)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(flat)
                if probs.shape[1] == 2:
                    return probs[:, 1].astype(np.float32)
                return probs.astype(np.float32)
            return model.predict(flat).astype(np.float32)

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
