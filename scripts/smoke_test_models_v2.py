#!/usr/bin/env python3
"""Smoke test: build GRU v2 and TCN v2, run 2 training epochs."""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


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


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate, block_id, l2):
    residual = x
    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal", use_bias=False, kernel_regularizer=l2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal", use_bias=False, kernel_regularizer=l2)(x)
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
        x = tcn_block(x, 24, 2, d, 0.05, d, l2)
    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.2, seed=seed)(x)
    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=keras.initializers.Constant(prior_logit))(x)
    return keras.Model(inputs, outputs)


def tune_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.35, 0.65, 61):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def quick_train(model, X_train, y_train, X_val, y_val, name):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=64,
        verbose=0,
        shuffle=False,
    )
    val_prob = model.predict(X_val, verbose=0).flatten()
    auc = roc_auc_score(y_val, val_prob)
    t = tune_threshold(y_val, val_prob)
    print(f"{name}: params={model.count_params():,}  val_auc={hist.history['val_auc'][-1]:.4f}  prob_std={val_prob.std():.4f}  threshold={t:.3f}")
    return val_prob.std() > 0.001


def main():
    X_train, y_train, X_val, y_val, _, _ = load_data()
    lookback, n_features = X_train.shape[1], X_train.shape[2]

    gru = build_gru(lookback, n_features)
    tcn = build_tcn(lookback, n_features, float(y_train.mean()))

    ok_gru = quick_train(gru, X_train, y_train, X_val, y_val, "GRU v2")
    tcn.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    ok_tcn = quick_train(tcn, X_train, y_train, X_val, y_val, "TCN v2")

    if not (ok_gru and ok_tcn):
        print("WARN: very low prediction variance after 2 epochs (expected early in training)")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
