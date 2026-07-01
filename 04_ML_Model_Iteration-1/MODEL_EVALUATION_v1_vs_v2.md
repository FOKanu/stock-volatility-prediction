# Model Evaluation Report: Version 1 vs Version 2

**Project:** AAPL Next-Day Direction Prediction  
**Date:** June 2025  
**Notebooks:** `03a_model_bidirectional_gru.ipynb` (GRU), `03b_model_tcn.ipynb` (TCN)  
**Artifacts:** `data/models/v2_eval_comparison.json`, `data/models/tcn_predictions.json`

---

## Executive Summary

Two classifiers were trained to predict whether AAPL's adjusted close will move **UP (1)** or **DOWN (0)** on the next trading day, using 60-day sequences of 7 engineered features.

| Model | v1 Diagnosis | v2 Outcome | Val AUC (v1 → v2) |
|-------|--------------|------------|-------------------|
| **BiGRU → UniGRU** | Severe overfitting | Overfitting fixed; modest AUC gain | 0.493 → **0.516** |
| **TCN** | Model collapse (constant predictions) | Collapse fixed; clear AUC gain on validation | 0.499 → **0.559** |

**Bottom line:** Version 2 fixes the *training pathologies* that made version 1 unreliable. Neither model is a strong standalone predictor yet, but **TCN v2 is the stronger model on validation** (AUC 0.559, above the 0.52 target). GRU v2 is the bigger *training-stability* improvement (train–val AUC gap collapsed from ~0.40 to ~0.005).

---

## Task & Evaluation Protocol

| Setting | Value |
|---------|-------|
| Target | Next-day direction (binary) |
| Lookback | 60 trading days |
| Features | 7 (`log_return`, `log_volume`, `intraday_range`, `volatility_21`, `return_ma_21`, `rsi_14`, `spy_log_return`) |
| Train | ≤ 2017-12-31 |
| Validation | 2018-01-01 – 2019-12-31 (n ≈ 503) |
| Test | 2020 Q1 COVID stress window (n = 63) |
| Primary metric | **AUC-ROC** (ranking quality, threshold-independent) |
| Secondary metrics | Accuracy, F1 at validation-tuned threshold |
| v2 threshold | Chosen on validation (max F1 over 0.35–0.65), applied unchanged to test |

**Baselines for context (validation):**

| Baseline | Accuracy | F1 |
|----------|----------|-----|
| Always UP | 0.549 | 0.709 |
| Always DOWN | 0.451 | 0.000 |
| Last Direction | ~0.50 | ~0.50 |

Because ~55% of validation days are UP, a model that always predicts UP achieves high accuracy and F1 while providing zero ranking signal (AUC ≈ 0.50).

---

## Version 1 Performance

### Model 1: Bidirectional GRU (BiGRU)

**Architecture**

```
Input (60, 7)
  └─ Bidirectional GRU (64) → Bidirectional GRU (32)
       └─ Dense (32, ReLU) + Dropout(0.3)
            └─ Dense (16, ReLU) + Dropout(0.2)
                 └─ Dense (1, Sigmoid)
```

| Property | Value |
|----------|-------|
| Parameters | ~61,000 |
| Optimizer | Adam, lr = 1e-3 |
| Recurrent dropout | 0.0 |
| Max epochs | 100 |
| Early stopping patience | 15 |

**Training behaviour**

| Metric | Train | Validation |
|--------|-------|------------|
| AUC at best epoch (~89) | **~0.89** | **~0.49** |
| Train–val AUC gap | ~**0.40** | — |

The model memorised training patterns with no generalisation—a classic overfitting failure.

**Held-out results (threshold = 0.5)**

| Split | Accuracy | F1 | AUC-ROC |
|-------|----------|-----|---------|
| Validation (2018–2019) | 0.495 | 0.513 | **0.493** |
| Test (2020 Q1) | 0.492 | 0.515 | **0.496** |

BiGRU v1 performed **worse than the Always UP baseline** on validation accuracy and F1, and AUC was indistinguishable from random.

---

### Model 2: Temporal Convolutional Network (TCN)

**Architecture**

```
Input (60, 7)
  └─ Conv1D projection
       └─ 5 residual blocks (dilations 1, 2, 4, 8, 16)
            └─ GlobalAveragePooling1D
                 └─ Dense (32, ReLU) + Dropout
                      └─ Dense (1, Sigmoid)
```

| Property | Value |
|----------|-------|
| Optimizer | Adam, lr = 1e-3 |
| Spatial dropout | 0.2 |
| Readout | Global average over all timesteps |
| Max epochs | 100 |

**Training behaviour**

Training loss decreased smoothly, but the model **collapsed** to a near-constant output:

| Diagnostic | Validation | Test |
|------------|------------|------|
| Prediction mean P(UP) | ~0.52 | ~0.52 |
| Prediction std | **0.004** | **0.002** |
| % predicted UP | 100% | 100% |

Every sample received essentially the same score (~0.52), so the model was equivalent to predicting UP for everything.

**Held-out results (threshold = 0.5)**

| Split | Accuracy | F1 | AUC-ROC | Prob std |
|-------|----------|-----|---------|----------|
| Validation (2018–2019) | 0.549 | 0.709 | **0.499** | 0.004 |
| Test (2020 Q1) | 0.549 | 0.709 | **0.629** | 0.002 |

Validation accuracy and F1 **exactly match Always UP**—the model adds no discriminative value. The elevated test AUC (0.629) is a **misleading artifact** of ranking near-identical scores on a small, imbalanced test set (63 samples), not genuine predictive skill.

---

## Version 2 Changes

### GRU (formerly BiGRU) — anti-overfitting

| Area | v1 | v2 |
|------|----|----|
| Architecture | Stacked Bidirectional GRU (64→32) | Single **unidirectional** GRU(32) |
| Parameters | ~61,000 | **~4,500** |
| Recurrent dropout | 0.0 | **0.2** |
| Dropout / L2 | Light | Dropout 0.4/0.3, L2 = 1e-3 |
| Optimizer | Adam 1e-3 | Adam **3e-4**, clipnorm = 1.0 |
| Early stopping | patience 15, 100 epochs | patience **12**, max **50** epochs |
| Threshold | Fixed 0.5 | **Tuned on validation** (max F1) |

### TCN — anti-collapse

| Area | v1 | v2 |
|------|----|----|
| Blocks | 5 (d = 1, 2, 4, 8, 16) | **3** (d = 1, 2, 4) |
| Filters | 32 per block | **24** per block |
| Readout | GlobalAveragePooling1D | **Last timestep** (`t[:, -1, :]`) |
| Spatial dropout | 0.2 | **0.05** |
| Regularisation | None on conv | Conv L2 = 1e-4 |
| Optimizer | Adam 1e-3 | **AdamW** 3e-4, weight_decay = 1e-4, clipnorm = 1.0 |
| Output bias | Default (0) | **Initialised to training UP-rate logit (~0.51)** |
| Early stopping | patience 15, 100 epochs | patience **12**, max **50** epochs |
| Threshold | Fixed 0.5 | **Tuned on validation** |
| Diagnostics | None | Post-train **prediction std** check (collapse if < 0.02) |

**Rationale for TCN readout change:** Global average pooling diluted the most recent (and most predictive) timesteps into a 60-day mean, encouraging the network to learn a single scalar bias rather than temporal patterns. The last-timestep head preserves recent signal for a next-day prediction task.

---

## Version 2 Performance

Results below are from a controlled re-evaluation (`scripts/eval_v2_models.py`) using the v2 notebook architectures, training callbacks, and threshold-tuning logic. Random seed = 42; training on CPU (GPU unavailable in eval environment).

### GRU v2

**Training**

| Metric | v1 BiGRU | v2 GRU |
|--------|----------|--------|
| Parameters | ~61,000 | 4,481 |
| Epochs run | 100 | 26 (early stop @ epoch 14) |
| Best val AUC | ~0.49 | **0.521** |
| Train–val AUC gap | ~0.40 | **0.005** |

**Held-out results (threshold = 0.475)**

| Split | Accuracy | F1 | AUC-ROC | Prob std |
|-------|----------|-----|---------|----------|
| Validation | 0.555 | 0.711 | **0.516** | 0.012 |
| Test | 0.492 | 0.619 | **0.605** | 0.032 |

| vs v1 | Δ Val AUC | Δ Test AUC | Δ Val F1 | Δ Test F1 |
|-------|-----------|------------|----------|-----------|
| Change | **+0.023** | **+0.109** | +0.198 | +0.104 |

**Interpretation:** Overfitting is resolved. AUC improved on both splits. Val AUC remains barely above random (0.516). Test AUC (0.605) is encouraging on the stress window but should be treated cautiously given n = 63.

---

### TCN v2

**Training**

| Metric | v1 TCN | v2 TCN |
|--------|--------|--------|
| Parameters | ~15,000+ | 7,785 |
| Epochs run | ~20 (early stop) | 50 (best @ epoch 47) |
| Best val AUC | ~0.50 | **0.560** |
| Train–val AUC gap | N/A (collapsed) | **0.024** |
| Val prob std | **0.004** | **0.051** |

**Held-out results (threshold = 0.350)**

| Split | Accuracy | F1 | AUC-ROC | Prob std |
|-------|----------|-----|---------|----------|
| Validation | 0.549 | 0.709 | **0.559** | 0.051 |
| Test | 0.429 | 0.600 | **0.595** | 0.076 |

| vs v1 | Δ Val AUC | Δ Test AUC | Δ Val Acc | Δ Test Acc |
|-------|-----------|------------|-----------|------------|
| Change | **+0.060** | −0.034 | ~0 | −0.120 |

**Interpretation:**

- **Validation AUC +0.060** — genuine improvement; exceeds the 0.52 target. This is the primary success metric.
- **Prediction std 0.004 → 0.051** — collapse is fixed; the model now produces differentiated scores.
- **Validation accuracy/F1 unchanged** — still aligned with Always UP at the tuned threshold, but AUC shows the model now ranks samples meaningfully.
- **Test AUC 0.629 → 0.595** — a *decrease* vs v1, but v1's 0.629 was inflated by collapse. v2's 0.595 reflects honest ranking on 63 samples with real score variation.
- **Test accuracy 0.549 → 0.429** — worse than v1 because v1 effectively *was* the Always UP strategy; v2 makes actual UP/DOWN calls and misses on this volatile window.

---

## TCN v1 vs v2 — Detailed Comparison

### What v1 looked like in practice

```
Sample 1: P(UP) = 0.521
Sample 2: P(UP) = 0.519
Sample 3: P(UP) = 0.523
...
Sample N: P(UP) = 0.520
```

Standard deviation ≈ 0.004. No sample-level discrimination → AUC ≈ 0.50 on validation.

### What v2 looks like in practice

After v2 training, validation predictions span a meaningful range (std ≈ 0.05–0.08 depending on run), with the model assigning higher scores to some UP days and lower scores to some DOWN days. Example statistics from the controlled eval run:

| Statistic | v1 | v2 |
|-----------|----|----|
| Val prob mean | ~0.52 | ~0.51 |
| Val prob std | 0.004 | **0.051** |
| Val prob min / max | ~0.51 / ~0.53 | ~0.35 / ~0.65 (approx.) |
| Test prob std | 0.002 | **0.076** |

### Side-by-side summary

| Criterion | TCN v1 | TCN v2 | Winner |
|-----------|--------|--------|--------|
| Val AUC (primary) | 0.499 | **0.559** | v2 |
| Val ranking signal | None | Present | v2 |
| Model collapse | Yes | No | v2 |
| Val accuracy | 0.549 | 0.549 | Tie (baseline-level) |
| Test AUC (honest) | Misleading (0.629) | **0.595** | v2 (more trustworthy) |
| Test accuracy | 0.549 | 0.429 | v1 (but trivial strategy) |
| Training stability | Appeared stable | Stable with healthy gap | v2 |
| Architecture complexity | 5 blocks + GAP | 3 blocks + last step | v2 (simpler) |

---

## Head-to-Head: Best Model After v2

On **validation AUC** (most reliable split, n ≈ 503):

| Model | v1 AUC | v2 AUC |
|-------|--------|--------|
| GRU | 0.493 | 0.516 |
| TCN | 0.499 | **0.559** |

**TCN v2 is the recommended single model** for this iteration. GRU v2 is worth retaining for ensemble diversity (different inductive bias: recurrent vs convolutional).

On **test (2020 Q1, n = 63)** — interpret with caution:

| Model | v1 AUC | v2 AUC |
|-------|--------|--------|
| GRU | 0.496 | **0.605** |
| TCN | 0.629* | 0.595 |

\* v1 TCN inflated by collapse.

---

## Conclusions

1. **Version 1 failures were architectural/training issues, not data issues alone.**
   - BiGRU overfit aggressively (61k params, no recurrent dropout, 100 epochs).
   - TCN collapsed to a constant predictor due to GlobalAveragePooling and excessive dropout depth.

2. **Version 2 fixes worked as designed.**
   - GRU: train–val gap 0.40 → 0.005; val AUC +0.023.
   - TCN: prediction std 0.004 → 0.051; val AUC +0.060 (passes 0.52 target).

3. **Neither model is production-ready for directional trading.**
   - Val AUC of 0.56 (TCN) implies limited edge over random ranking.
   - High F1 scores largely reflect class imbalance and threshold tuning, not precision in calling direction.

4. **Recommended next steps**
   - Ensemble GRU v2 + TCN v2 predictions (decorrelated errors).
   - Feature ablation and alternative targets (e.g. magnitude buckets, multi-day horizon).
   - Walk-forward retraining to reduce regime sensitivity (2018–2019 val vs 2020 Q1 test).
   - Re-run notebooks on GPU for reproducible artifact refresh (`bigru_predictions.json`, `tcn_predictions.json`).

---

## Appendix: Data Sources

| File | Description |
|------|-------------|
| `data/processed/aapl_sequences.npz` | Train / val / test arrays |
| `data/processed/pipeline_config.json` | Split dates, lookback, features |
| `data/models/v2_eval_comparison.json` | Controlled v1 vs v2 comparison (primary v2 metrics) |
| `data/models/tcn_predictions.json` | Notebook-exported TCN v2 predictions |
| `scripts/eval_v2_models.py` | Reproducible v2 training and evaluation script |

**Note on run-to-run variance:** Neural network training is stochastic. Individual notebook runs may differ slightly from the controlled eval (e.g. notebook TCN v2: val AUC 0.527, test AUC 0.574 with threshold 0.48). Directional conclusions (collapse fixed, val AUC improved) hold across runs.
