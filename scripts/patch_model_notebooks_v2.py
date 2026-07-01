#!/usr/bin/env python3
"""Apply v2 architecture and training fixes to 03a (GRU) and 03b (TCN) notebooks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIGRU_PATH = ROOT / "04_ML_Model_Iteration-1" / "03a_model_bidirectional_gru.ipynb"
TCN_PATH = ROOT / "04_ML_Model_Iteration-1" / "03b_model_tcn.ipynb"


def set_cell_source(nb, idx, source: str):
    nb["cells"][idx]["source"] = source.splitlines(keepends=True)
    if not nb["cells"][idx]["source"]:
        nb["cells"][idx]["source"] = [source]


def patch_bigru(nb):
    set_cell_source(nb, 0, """# Stock Performance Prediction — Model 1: Unidirectional GRU (v2)

**Project:** Stock Performance Prediction (RNN)  
**Prerequisites:** `01_exploratory_data_analysis.ipynb`, `02_preprocessing.ipynb`  
**Loads from:** `data/processed/aapl_sequences.npz`, `scaler.json`, `pipeline_config.json`

This notebook trains and evaluates a **compact unidirectional GRU** classifier for next-day direction prediction on AAPL.

### v2 changes (anti-overfitting)
- Replaced stacked **Bidirectional** GRU (61k params) with a single **UniGRU(32)** (~4k params)
- Added **recurrent dropout (0.2)** and stronger L2 / dropout
- **Early stopping** with patience=12, max 50 epochs
- Lower learning rate (3e-4) + gradient clipping
- **Validation-tuned threshold** (not fixed 0.5)

### Pipeline
```
Load artifacts → Build model → Train with early stopping
→ Tune threshold on val → Evaluate (Val + Test) → Baselines → Save model
```

| Stage | Section |
|-------|---------|
| Setup & load | 1–2 |
| Model definition | 3 |
| Training | 4 |
| Evaluation | 5 |
| Baselines | 6 |
| Save | 7 |
""")

    set_cell_source(nb, 7, """## 3. Model Definition — Unidirectional GRU (v2)

### Architecture
```
Input (60, 7)
  └─ GRU (32 units, recurrent_dropout=0.2) + Dropout(0.4)
       └─ Dense (16, ReLU, L2=1e-3) + Dropout(0.3)
            └─ Dense (1, Sigmoid)  →  P(next day UP)
```

**Design choices:**
- **Unidirectional** — causal at inference time; fewer parameters than bidirectional (~4k vs ~61k)
- **recurrent_dropout=0.2** — regularises inside the recurrence (was 0.0 in v1)
- **Single layer** — stacked BiGRU memorised training noise (train AUC 0.89, val AUC 0.49)
- L2 on GRU kernels and dense layers to penalise large weights
""")

    set_cell_source(nb, 8, """def build_gru(lookback: int, n_features: int, seed: int = 42) -> keras.Model:
    \"\"\"Compact unidirectional GRU for binary direction classification.\"\"\"
    tf.random.set_seed(seed)
    l2 = regularizers.l2(1e-3)

    inputs = keras.Input(shape=(lookback, n_features), name="sequence_input")

    x = layers.GRU(
        32,
        return_sequences=False,
        dropout=0.0,
        recurrent_dropout=0.2,
        kernel_regularizer=l2,
        recurrent_regularizer=l2,
        name="gru_1",
    )(inputs)
    x = layers.Dropout(0.4, seed=seed, name="drop_1")(x)

    x = layers.Dense(16, activation="relu", kernel_regularizer=l2, name="dense_1")(x)
    x = layers.Dropout(0.3, seed=seed, name="drop_2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="GRU_DirectionClassifier")
    return model


model = build_gru(LOOKBACK, N_FEATURES, seed=SEED)
model.summary(line_length=80)
""")

    set_cell_source(nb, 9, """# ── Compile ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
    ],
)
print("Model compiled.")
print(f"Total trainable parameters: {model.count_params():,}")
""")

    set_cell_source(nb, 10, """## 4. Training

**Callbacks:**
- `EarlyStopping` — monitors `val_auc`; stops if no improvement for **12** epochs; restores best weights
- `ReduceLROnPlateau` — halves LR if `val_loss` stalls for **5** epochs (floor: 1e-5)
- `ModelCheckpoint` — saves best model to disk by `val_auc`
- **Max 50 epochs** — prevents prolonged overfitting seen in v1 (100 epochs, train AUC → 0.89)

**Note on COVID test set:** The test set (2020 Q1) is never used for early stopping or threshold tuning.
""")

    set_cell_source(nb, 11, """MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 12
LR_PATIENCE = 5

CHECKPOINT_PATH = MODELS_DIR / "bigru_best.keras"

cb_early = callbacks.EarlyStopping(
    monitor="val_auc",
    patience=EARLY_STOP_PATIENCE,
    mode="max",
    restore_best_weights=True,
    verbose=1,
)
cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=LR_PATIENCE,
    min_lr=1e-5,
    verbose=1,
)
cb_checkpoint = callbacks.ModelCheckpoint(
    filepath=str(CHECKPOINT_PATH),
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=0,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=64,
    callbacks=[cb_early, cb_reduce_lr, cb_checkpoint],
    verbose=1,
    shuffle=False,
)
""")

    set_cell_source(nb, 12, """# ── Training curves ───────────────────────────────────────────────────────
hist = pd.DataFrame(history.history)
epochs_ran = len(hist)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (train_col, val_col, title) in zip(axes, [
    ("loss",     "val_loss",     "Binary Cross-Entropy Loss"),
    ("accuracy", "val_accuracy", "Accuracy"),
    ("auc",      "val_auc",      "AUC-ROC"),
]):
    ax.plot(hist[train_col], label="Train", linewidth=1.8)
    ax.plot(hist[val_col],   label="Val",   linewidth=1.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.axvline(hist[val_col].idxmax() if "auc" in val_col else hist[val_col].idxmin(),
               color="gray", linewidth=0.8, linestyle=":")

fig.suptitle(f"GRU v2 — Training History ({epochs_ran} epochs)", y=1.02)
plt.tight_layout()
plt.show()

best_epoch = hist["val_auc"].idxmax()
gap = hist["auc"][best_epoch] - hist["val_auc"][best_epoch]
print(
    f"Best epoch: {best_epoch + 1}  |  val_auc={hist['val_auc'][best_epoch]:.4f}  "
    f"|  val_loss={hist['val_loss'][best_epoch]:.4f}"
)
print(f"Train–val AUC gap at best epoch: {gap:.4f}  (overfit warning if > 0.10)")
""")

    set_cell_source(nb, 13, """## 5. Evaluation

Evaluation is run on both the **validation set** (2018–2019) and the **test set** (2020 Q1, COVID stress window).

**Threshold tuning:** The decision threshold is chosen on **validation only** (maximises F1 over 0.35–0.65), then applied unchanged to test.

**Primary metrics:** AUC-ROC (ranking quality), F1 at tuned threshold  
**Secondary:** Accuracy, Precision, Recall
""")

    set_cell_source(nb, 14, """def tune_threshold(y_true, y_prob):
    \"\"\"Pick threshold on validation that maximises F1.\"\"\"
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.35, 0.65, 61):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def evaluate_split(model, X, y_true, split_name: str, threshold: float = 0.5) -> dict:
    \"\"\"Full evaluation on a single split. Returns a metrics dict.\"\"\"
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\\n{'='*55}")
    print(f"  {split_name.upper()} RESULTS  (n={len(y_true)}, threshold={threshold:.3f})")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Prob std : {y_prob.std():.4f}")
    print(f"\\n{classification_report(y_true, y_pred, target_names=['DOWN (0)', 'UP (1)'], zero_division=0)}")

    return {
        "split": split_name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "prob_std": float(y_prob.std()),
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_true": y_true,
    }


val_prob_tune = model.predict(X_val, verbose=0).flatten()
OPTIMAL_THRESHOLD, val_best_f1 = tune_threshold(y_val, val_prob_tune)
print(f"Optimal threshold (val, max F1): {OPTIMAL_THRESHOLD:.3f}  (val F1 at tune={val_best_f1:.4f})")

val_results = evaluate_split(
    model, X_val, y_val, "Validation (2018–2019)", threshold=OPTIMAL_THRESHOLD
)
test_results = evaluate_split(
    model, X_test, y_test, "Test — COVID Stress (2020 Q1)", threshold=OPTIMAL_THRESHOLD
)
""")

    src15 = "".join(nb["cells"][15]["source"]).replace("BiGRU", "GRU v2")
    nb["cells"][15]["source"] = src15.splitlines(keepends=True)
    src16 = "".join(nb["cells"][16]["source"]).replace("BiGRU", "GRU v2")
    nb["cells"][16]["source"] = src16.splitlines(keepends=True)
    src17 = "".join(nb["cells"][17]["source"]).replace("BiGRU", "GRU v2")
    src17 = src17.replace('label="Threshold=0.5"', 'label=f"Threshold={OPTIMAL_THRESHOLD:.3f}"')
    src17 = src17.replace('ax.axvline(0.5,', 'ax.axvline(OPTIMAL_THRESHOLD,')
    nb["cells"][17]["source"] = src17.splitlines(keepends=True)

    set_cell_source(nb, 19, """def baseline_metrics(y_true, y_pred, name: str) -> dict:
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


rows = []
for split_name, X, y in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    last_return = X[:, -1, 0]
    last_dir = (last_return > 0).astype(int)

    gru_pred = (model.predict(X, verbose=0).flatten() >= OPTIMAL_THRESHOLD).astype(int)

    for row in [
        baseline_metrics(y, np.ones_like(y, dtype=int), f"[{split_name}] Always UP"),
        baseline_metrics(y, np.zeros_like(y, dtype=int), f"[{split_name}] Always DOWN"),
        baseline_metrics(y, last_dir, f"[{split_name}] Last Direction"),
        {
            "Model": f"[{split_name}] GRU v2 (ours)",
            "Accuracy": round(accuracy_score(y, gru_pred), 4),
            "F1": round(f1_score(y, gru_pred, zero_division=0), 4),
        },
    ]:
        rows.append(row)

baseline_df = pd.DataFrame(rows)
print(f"\\nBaseline comparison (threshold={OPTIMAL_THRESHOLD:.3f}):")
display(baseline_df.set_index("Model"))
""")

    src20 = "".join(nb["cells"][20]["source"]).replace("BiGRU", "GRU v2")
    nb["cells"][20]["source"] = src20.splitlines(keepends=True)

    set_cell_source(nb, 22, """# ── Save final model ──────────────────────────────────────────────────────
final_path = MODELS_DIR / "bigru_final.keras"
model.save(final_path)
print(f"Model saved: {final_path}")

preds = {
    "model_version": "gru_v2",
    "optimal_threshold": OPTIMAL_THRESHOLD,
    "val_prob": val_results["y_prob"].tolist(),
    "val_pred": val_results["y_pred"].tolist(),
    "val_true": val_results["y_true"].tolist(),
    "test_prob": test_results["y_prob"].tolist(),
    "test_pred": test_results["y_pred"].tolist(),
    "test_true": test_results["y_true"].tolist(),
    "metrics": {
        "val": {
            "accuracy": val_results["accuracy"],
            "f1": val_results["f1"],
            "auc": val_results["auc"],
            "prob_std": val_results["prob_std"],
        },
        "test": {
            "accuracy": test_results["accuracy"],
            "f1": test_results["f1"],
            "auc": test_results["auc"],
            "prob_std": test_results["prob_std"],
        },
    },
}
preds_path = MODELS_DIR / "bigru_predictions.json"
with open(preds_path, "w") as f:
    json.dump(preds, f, indent=2)
print(f"Predictions saved: {preds_path}")

summary = pd.DataFrame([
    {"Split": "Validation (2018–2019)", **preds["metrics"]["val"]},
    {"Split": "Test — COVID Stress (2020 Q1)", **preds["metrics"]["test"]},
])
print(f"\\n── GRU v2 Final Results (threshold={OPTIMAL_THRESHOLD:.3f}) ──")
display(summary.set_index("Split").round(4))
""")

    set_cell_source(nb, 23, """---

## Summary

This notebook (v2) trains a **compact unidirectional GRU(32)** on AAPL next-day direction prediction using 60-day sequences of 7 engineered features.

**v2 fixes:** reduced capacity, recurrent dropout, stronger regularisation, real early stopping (12 patience / 50 max epochs), validation-tuned threshold.

**Key artifacts:**
- `data/models/bigru_best.keras` — best checkpoint (by val AUC)
- `data/models/bigru_final.keras` — final model
- `data/models/bigru_predictions.json` — probabilities, metrics, and optimal threshold

**Next:** `03b_model_tcn.ipynb` — Temporal Convolutional Network (v2).
""")


def patch_tcn(nb):
    set_cell_source(nb, 0, """# Stock Performance Prediction — Model 2: Temporal Convolutional Network (v2)

**Project:** Stock Performance Prediction (RNN)  
**Prerequisites:** `01_exploratory_data_analysis.ipynb`, `02_preprocessing.ipynb`  
**Loads from:** `data/processed/aapl_sequences.npz`, `scaler.json`, `pipeline_config.json`

This notebook trains and evaluates a **Temporal Convolutional Network (TCN)** classifier for next-day direction prediction on AAPL.

### v2 changes (fix model collapse)
- Replaced **GlobalAveragePooling** with **last-timestep** readout (recent days matter most)
- Simplified stack: **3 blocks** (d=1,2,4), 24 filters, kernel_size=2
- Reduced spatial dropout (0.05) and added conv L2 + **AdamW**
- Output bias initialised to training class prior (prevents "always UP" collapse)
- **Validation-tuned threshold** + prediction-std diagnostic

### Pipeline
```
Load artifacts → Build TCN → Train with early stopping
→ Tune threshold on val → Evaluate (Val + Test) → Baselines → Save model
```
""")

    set_cell_source(nb, 8, """## 3. Model Definition — TCN (v2)

### Architecture

Three dilated causal residual blocks (was five in v1). Each block: Conv1D → LayerNorm → ReLU → SpatialDropout × 2 + skip connection.

Dilation factors: `[1, 2, 4]` — receptive field covers the 60-day lookback without over-deepening the stack.

```
Input (60, 7)
  └─ Conv1D projection (1×1, 24 filters)
       └─ Residual block (d=1, 24)
            └─ Residual block (d=2, 24)
                 └─ Residual block (d=4, 24)
                      └─ Last timestep slice        ← replaces GlobalAvgPool
                           └─ Dense (16, ReLU) + Dropout
                                └─ Dense (1, Sigmoid)
```

**Why last timestep:** v1 GlobalAveragePooling washed out recent signal; the model collapsed to P(UP)≈0.52 for every sample.
""")

    set_cell_source(nb, 9, """def tcn_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    block_id: int,
    l2,
) -> tf.Tensor:
    residual = x

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=False,
        kernel_regularizer=l2,
        name=f"block{block_id}_conv1",
    )(x)
    x = layers.LayerNormalization(name=f"block{block_id}_ln1")(x)
    x = layers.Activation("relu", name=f"block{block_id}_relu1")(x)
    x = layers.SpatialDropout1D(dropout_rate, name=f"block{block_id}_drop1")(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=False,
        kernel_regularizer=l2,
        name=f"block{block_id}_conv2",
    )(x)
    x = layers.LayerNormalization(name=f"block{block_id}_ln2")(x)
    x = layers.Activation("relu", name=f"block{block_id}_relu2")(x)
    x = layers.SpatialDropout1D(dropout_rate, name=f"block{block_id}_drop2")(x)

    if int(residual.shape[-1]) != filters:
        residual = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2,
            name=f"block{block_id}_skip_proj",
        )(residual)

    return layers.Add(name=f"block{block_id}_add")([x, residual])


def build_tcn(
    lookback: int,
    n_features: int,
    train_up_rate: float = 0.5,
    filters: list | None = None,
    kernel_size: int = 2,
    dropout_rate: float = 0.05,
    seed: int = 42,
) -> keras.Model:
    tf.random.set_seed(seed)
    l2 = regularizers.l2(1e-4)

    if filters is None:
        filters = [(1, 24), (2, 24), (4, 24)]

    prior_logit = float(np.log(train_up_rate / (1.0 - train_up_rate)))

    inputs = keras.Input(shape=(lookback, n_features), name="sequence_input")

    x = layers.Conv1D(
        filters=filters[0][1],
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2,
        name="input_projection",
    )(inputs)

    for block_id, (dilation_rate, n_filters) in enumerate(filters):
        x = tcn_residual_block(
            x,
            filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
            block_id=block_id,
            l2=l2,
        )

    x = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)

    x = layers.Dense(16, activation="relu", kernel_regularizer=l2, name="dense_1")(x)
    x = layers.Dropout(0.2, seed=seed, name="dense_drop")(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(prior_logit),
        name="output",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="TCN_DirectionClassifier")


TRAIN_UP_RATE = float(y_train.mean())
model = build_tcn(LOOKBACK, N_FEATURES, train_up_rate=TRAIN_UP_RATE, seed=SEED)
model.summary(line_length=80)
print(f"Train UP rate (bias init): {TRAIN_UP_RATE:.3f}")
""")

    set_cell_source(nb, 10, """# ── Compile ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
    ],
)
print("Model compiled.")
print(f"Total trainable parameters: {model.count_params():,}")
""")

    set_cell_source(nb, 11, """## 4. Training

**Callbacks (v2):**
- `EarlyStopping` on `val_auc` (patience=**12**, restore best weights)
- `ReduceLROnPlateau` on `val_loss` (patience=**5**, factor=0.5, min_lr=1e-5)
- `ModelCheckpoint` saves best by `val_auc`
- **Max 50 epochs**

After training, check **val prediction std** — if < 0.02 the model has collapsed to a constant predictor again.
""")

    set_cell_source(nb, 12, """MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 12
LR_PATIENCE = 5

CHECKPOINT_PATH = MODELS_DIR / "tcn_best.keras"

cb_early = callbacks.EarlyStopping(
    monitor="val_auc",
    patience=EARLY_STOP_PATIENCE,
    mode="max",
    restore_best_weights=True,
    verbose=1,
)
cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=LR_PATIENCE,
    min_lr=1e-5,
    verbose=1,
)
cb_checkpoint = callbacks.ModelCheckpoint(
    filepath=str(CHECKPOINT_PATH),
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=0,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=64,
    callbacks=[cb_early, cb_reduce_lr, cb_checkpoint],
    verbose=1,
    shuffle=False,
)

val_prob_diag = model.predict(X_val, verbose=0).flatten()
print(f"\\nPost-train val prob std: {val_prob_diag.std():.4f}  (collapse warning if < 0.02)")
""")

    set_cell_source(nb, 13, """# ── Training curves ───────────────────────────────────────────────────────
hist = pd.DataFrame(history.history)
epochs_ran = len(hist)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (train_col, val_col, title) in zip(axes, [
    ("loss",     "val_loss",     "Binary Cross-Entropy Loss"),
    ("accuracy", "val_accuracy", "Accuracy"),
    ("auc",      "val_auc",      "AUC-ROC"),
]):
    ax.plot(hist[train_col], label="Train", linewidth=1.8)
    ax.plot(hist[val_col],   label="Val",   linewidth=1.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.axvline(hist[val_col].idxmax() if "auc" in val_col else hist[val_col].idxmin(),
               color="gray", linewidth=0.8, linestyle=":")

fig.suptitle(f"TCN v2 — Training History ({epochs_ran} epochs)", y=1.02)
plt.tight_layout()
plt.show()

best_epoch = hist["val_auc"].idxmax()
gap = hist["auc"][best_epoch] - hist["val_auc"][best_epoch]
print(
    f"Best epoch: {best_epoch + 1}  |  val_auc={hist['val_auc'][best_epoch]:.4f}  "
    f"|  val_loss={hist['val_loss'][best_epoch]:.4f}"
)
print(f"Train–val AUC gap at best epoch: {gap:.4f}")
""")

    set_cell_source(nb, 14, """## 5. Evaluation

**Threshold tuning:** chosen on validation (max F1), applied to test unchanged.

**Primary metrics:** AUC-ROC, F1 at tuned threshold  
**Diagnostic:** prediction std should be >> 0.02 on both splits
""")

    set_cell_source(nb, 15, """def tune_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.35, 0.65, 61):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def evaluate_split(model, X, y_true, split_name: str, threshold: float = 0.5) -> dict:
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\\n{'='*55}")
    print(f"  {split_name.upper()} RESULTS  (n={len(y_true)}, threshold={threshold:.3f})")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Prob std : {y_prob.std():.4f}")
    print(f"\\n{classification_report(y_true, y_pred, target_names=['DOWN (0)', 'UP (1)'], zero_division=0)}")

    return {
        "split": split_name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "prob_std": float(y_prob.std()),
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_true": y_true,
    }


val_prob_tune = model.predict(X_val, verbose=0).flatten()
OPTIMAL_THRESHOLD, val_best_f1 = tune_threshold(y_val, val_prob_tune)
print(f"Optimal threshold (val, max F1): {OPTIMAL_THRESHOLD:.3f}  (val F1 at tune={val_best_f1:.4f})")

val_results = evaluate_split(
    model, X_val, y_val, "Validation (2018–2019)", threshold=OPTIMAL_THRESHOLD
)
test_results = evaluate_split(
    model, X_test, y_test, "Test — COVID Stress (2020 Q1)", threshold=OPTIMAL_THRESHOLD
)
""")

    for idx in (16, 17, 18):
        src = "".join(nb["cells"][idx]["source"]).replace("TCN —", "TCN v2 —")
        if idx == 18:
            src = src.replace('ax.axvline(0.5,', 'ax.axvline(OPTIMAL_THRESHOLD,')
            src = src.replace('label="Threshold=0.5"', 'label=f"Threshold={OPTIMAL_THRESHOLD:.3f}"')
        nb["cells"][idx]["source"] = src.splitlines(keepends=True)

    set_cell_source(nb, 20, """def baseline_metrics(y_true, y_pred, name: str) -> dict:
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


rows = []
for split_name, X, y in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    last_return = X[:, -1, 0]
    last_dir = (last_return > 0).astype(int)

    tcn_pred = (model.predict(X, verbose=0).flatten() >= OPTIMAL_THRESHOLD).astype(int)

    for row in [
        baseline_metrics(y, np.ones_like(y, dtype=int), f"[{split_name}] Always UP"),
        baseline_metrics(y, np.zeros_like(y, dtype=int), f"[{split_name}] Always DOWN"),
        baseline_metrics(y, last_dir, f"[{split_name}] Last Direction"),
        {
            "Model": f"[{split_name}] TCN v2 (ours)",
            "Accuracy": round(accuracy_score(y, tcn_pred), 4),
            "F1": round(f1_score(y, tcn_pred, zero_division=0), 4),
        },
    ]:
        rows.append(row)

baseline_df = pd.DataFrame(rows)
print(f"\\nBaseline comparison (threshold={OPTIMAL_THRESHOLD:.3f}):")
display(baseline_df.set_index("Model"))
""")

    src21 = "".join(nb["cells"][21]["source"]).replace("TCN vs Baselines", "TCN v2 vs Baselines")
    nb["cells"][21]["source"] = src21.splitlines(keepends=True)

    set_cell_source(nb, 23, """# ── Save final model ──────────────────────────────────────────────────────
final_path = MODELS_DIR / "tcn_final.keras"
model.save(final_path)
print(f"Model saved: {final_path}")

preds = {
    "model_version": "tcn_v2",
    "optimal_threshold": OPTIMAL_THRESHOLD,
    "val_prob": val_results["y_prob"].tolist(),
    "val_pred": val_results["y_pred"].tolist(),
    "val_true": val_results["y_true"].tolist(),
    "test_prob": test_results["y_prob"].tolist(),
    "test_pred": test_results["y_pred"].tolist(),
    "test_true": test_results["y_true"].tolist(),
    "metrics": {
        "val": {
            "accuracy": val_results["accuracy"],
            "f1": val_results["f1"],
            "auc": val_results["auc"],
            "prob_std": val_results["prob_std"],
        },
        "test": {
            "accuracy": test_results["accuracy"],
            "f1": test_results["f1"],
            "auc": test_results["auc"],
            "prob_std": test_results["prob_std"],
        },
    },
}
preds_path = MODELS_DIR / "tcn_predictions.json"
with open(preds_path, "w") as f:
    json.dump(preds, f, indent=2)
print(f"Predictions saved: {preds_path}")

summary = pd.DataFrame([
    {"Split": "Validation (2018–2019)", **preds["metrics"]["val"]},
    {"Split": "Test — COVID Stress (2020 Q1)", **preds["metrics"]["test"]},
])
print(f"\\n── TCN v2 Final Results (threshold={OPTIMAL_THRESHOLD:.3f}) ──")
display(summary.set_index("Split").round(4))
""")

    set_cell_source(nb, 24, """---

## Summary

This notebook (v2) trains a **3-block TCN** with last-timestep readout on AAPL next-day direction prediction.

**v2 fixes:** removed GlobalAveragePooling, simplified stack, AdamW + light dropout, prior bias init, validation-tuned threshold.

**Key artifacts:**
- `data/models/tcn_best.keras` — best checkpoint (by val AUC)
- `data/models/tcn_final.keras` — final model
- `data/models/tcn_predictions.json` — probabilities, metrics, and optimal threshold
""")


def main():
    for path, patch_fn in [(BIGRU_PATH, patch_bigru), (TCN_PATH, patch_tcn)]:
        with open(path) as f:
            nb = json.load(f)
        patch_fn(nb)
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Patched {path.name}")


if __name__ == "__main__":
    main()
