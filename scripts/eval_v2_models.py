#!/usr/bin/env python3
"""Train and evaluate GRU v2 + TCN v2 (mirrors notebook logic)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"

V1 = {
    "bigru": {
        "val": {"accuracy": 0.495, "f1": 0.513, "auc": 0.493},
        "test": {"accuracy": 0.492, "f1": 0.515, "auc": 0.496},
        "notes": "train AUC ~0.89, val AUC ~0.49 (overfit)",
    },
    "tcn": {
        "val": {"accuracy": 0.549, "f1": 0.709, "auc": 0.499, "prob_std": 0.004},
        "test": {"accuracy": 0.549, "f1": 0.709, "auc": 0.629, "prob_std": 0.002},
        "notes": "collapsed to P(UP)≈0.52 for all samples",
    },
}


def load_data():
    data = np.load(PROCESSED / "aapl_sequences.npz")
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def tune_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.35, 0.65, 61):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def evaluate(model, X, y, threshold):
    prob = model.predict(X, verbose=0).flatten()
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)),
        "prob_std": float(prob.std()),
    }


def train_model(model, X_train, y_train, X_val, y_val, ckpt_name):
    cb = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=12, mode="max", restore_best_weights=True, verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0
        ),
    ]
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=cb,
        verbose=0,
        shuffle=False,
    )
    h = hist.history
    best = int(np.argmax(h["val_auc"]))
    gap = h["auc"][best] - h["val_auc"][best]
    return {
        "epochs": len(h["auc"]),
        "best_epoch": best + 1,
        "best_val_auc": float(h["val_auc"][best]),
        "train_val_auc_gap": float(gap),
        "checkpoint": ckpt_name,
    }


def build_gru(lookback, n_features, seed=42):
    tf.random.set_seed(seed)
    l2 = regularizers.l2(1e-3)
    inputs = keras.Input(shape=(lookback, n_features))
    x = layers.GRU(
        32,
        return_sequences=False,
        recurrent_dropout=0.2,
        kernel_regularizer=l2,
        recurrent_regularizer=l2,
    )(inputs)
    x = layers.Dropout(0.4, seed=seed)(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.3, seed=seed)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate, l2):
    residual = x
    x = layers.Conv1D(
        filters, kernel_size, dilation_rate=dilation_rate, padding="causal", use_bias=False, kernel_regularizer=l2
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    x = layers.Conv1D(
        filters, kernel_size, dilation_rate=dilation_rate, padding="causal", use_bias=False, kernel_regularizer=l2
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    if int(residual.shape[-1]) != filters:
        residual = layers.Conv1D(filters, 1, padding="same", use_bias=False, kernel_regularizer=l2)(residual)
    return layers.Add()([x, residual])


def build_tcn(lookback, n_features, train_up_rate, seed=42):
    tf.random.set_seed(seed)
    l2 = regularizers.l2(1e-4)
    prior_logit = float(np.log(train_up_rate / (1.0 - train_up_rate)))
    inputs = keras.Input(shape=(lookback, n_features))
    x = layers.Conv1D(24, 1, padding="same", use_bias=False, kernel_regularizer=l2)(inputs)
    for d in (1, 2, 4):
        x = tcn_block(x, 24, 2, d, 0.05, l2)
    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.2, seed=seed)(x)
    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=keras.initializers.Constant(prior_logit))(x)
    return keras.Model(inputs, outputs)


def delta(new, old, key):
    if key not in old:
        return "n/a"
    d = new[key] - old[key]
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    lookback, n_features = X_train.shape[1], X_train.shape[2]
    results = {}

    # GRU v2
    gru = build_gru(lookback, n_features)
    gru.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    train_info = train_model(gru, X_train, y_train, X_val, y_val, "gru_v2")
    val_prob = gru.predict(X_val, verbose=0).flatten()
    thr = tune_threshold(y_val, val_prob)
    results["gru_v2"] = {
        "train": train_info,
        "threshold": thr,
        "params": gru.count_params(),
        "val": evaluate(gru, X_val, y_val, thr),
        "test": evaluate(gru, X_test, y_test, thr),
    }

    # TCN v2
    tcn = build_tcn(lookback, n_features, float(y_train.mean()))
    tcn.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    train_info = train_model(tcn, X_train, y_train, X_val, y_val, "tcn_v2")
    val_prob = tcn.predict(X_val, verbose=0).flatten()
    thr = tune_threshold(y_val, val_prob)
    results["tcn_v2"] = {
        "train": train_info,
        "threshold": thr,
        "params": tcn.count_params(),
        "val": evaluate(tcn, X_val, y_val, thr),
        "test": evaluate(tcn, X_test, y_test, thr),
    }

    out = MODELS / "v2_eval_comparison.json"
    out.write_text(json.dumps({"v1_baseline": V1, "v2_results": results}, indent=2))

    print("=" * 72)
    print("GRU v2 vs BiGRU v1")
    print("=" * 72)
    g = results["gru_v2"]
    b = V1["bigru"]
    print(f"Params: {g['params']:,}  |  epochs={g['train']['epochs']} best={g['train']['best_epoch']} val_auc={g['train']['best_val_auc']:.4f} gap={g['train']['train_val_auc_gap']:.4f}")
    for split in ["val", "test"]:
        print(f"\n{split.upper()} (threshold={g['threshold']:.3f}):")
        for k in ["auc", "accuracy", "f1", "prob_std"]:
            nv = g[split].get(k, float("nan"))
            ov = b[split].get(k, float("nan"))
            if k == "prob_std" and k not in b[split]:
                print(f"  {k:10s}: {nv:.4f}")
            else:
                print(f"  {k:10s}: {ov:.4f} -> {nv:.4f}  ({delta(g[split], b[split], k)})")

    print("\n" + "=" * 72)
    print("TCN v2 vs TCN v1")
    print("=" * 72)
    t = results["tcn_v2"]
    tv = V1["tcn"]
    print(f"Params: {t['params']:,}  |  epochs={t['train']['epochs']} best={t['train']['best_epoch']} val_auc={t['train']['best_val_auc']:.4f} gap={t['train']['train_val_auc_gap']:.4f}")
    for split in ["val", "test"]:
        print(f"\n{split.upper()} (threshold={t['threshold']:.3f}):")
        for k in ["auc", "accuracy", "f1", "prob_std"]:
            print(f"  {k:10s}: {tv[split][k]:.4f} -> {t[split][k]:.4f}  ({delta(t[split], tv[split], k)})")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
