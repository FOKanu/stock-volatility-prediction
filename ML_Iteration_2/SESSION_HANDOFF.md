# Session Handoff — GARCH-GRU / QLIKE / Conformal work (2026-07-09 → 2026-07-10)

Point a new chat session at this file to resume without prior conversation history.
Everything below is backed by artifacts in `data/models/` and code in `scripts/`.

## Update 2026-07-11 — mandatory disk-backed full-universe policy

A purported J01 stocks full run was audited and found to use a stale bundle
containing only AAL/AAME (`n_symbols=2`). A forced rebuild identified 1,838
eligible stocks, then the old in-memory pipeline reached ~45.8 GiB RSS and the
Linux OOM killer terminated Python, disrupting the Remote WSL session.

Decision (persisted in `.cursor/rules/full-universe-disk-backed.mdc`):

- Full-universe preprocessing/training MUST be disk-backed and chunked.
- Full runs use `memmap_v1`: atomic staged directory, `.npy` split memmaps,
  streaming feature moments, lazy loading, and chunked TensorFlow batches.
- Dev200 remains compressed NPZ for compatibility.
- Never concatenate all per-symbol sequences, copy complete memmaps with
  `.astype()`, or feed memmaps to `from_tensor_slices`.
- Verify manifest/config symbol counts before training.

The J01 stocks GARCH-GRU notebook is configured for `RUN_MODE="full"` and
`PATIENCE=8`. The invalid two-symbol full-tag model must not be promoted.

## Update 2026-07-12 — J01 stocks full-universe bake-off (baseline vs GARCH-GRU)

Both models trained on the verified 1,838-symbol `memmap_v1` bundle
(6,400,094 / 920,149 / 113,705 train/val/test sequences).

| Model | Val MAE | Crisis MAE | Val AUC@90 | Crisis AUC@90 |
|---|---|---|---|---|
| Ridge baseline (full) | 0.01427 | **0.02591** | 0.775 | **0.841** |
| GARCH-GRU (full) | **0.01353** | 0.02611 | **0.807** | 0.790 |
| Persistence (untrained) | 0.01780 | 0.03320 | — | — |

**Verdict:** GARCH-GRU beats full baseline on val MAE (−5.2%) but **loses on
crisis MAE** (+0.8%). Per rubric gate (`beats_baseline_val_mae` AND
`beats_baseline_crisis_mae`), **no full-universe champion promotion**. Dev200
GARCH-GRU remains recorded champion in `champions.json`.

Artifacts: `baseline_vol_risk_stocks_tf_predictions.json`,
`garchgru_vol_risk_stocks_tf_predictions.json`. Comparison canvas:
`canvases/j01-full-universe-bakeoff-comparison.canvas.tsx`.

**Next:** QLIKE GARCH-GRU full-universe run from
`ML_Iteration_2/J01_vol_risk/stocks/01f_garchgru_vol_risk_stocks.ipynb` with
`LOSS="qlike"` in section 0 (saves `garchgru_qlike_vol_risk_stocks_tf_*`,
does not overwrite Huber artifacts). Targets lower crisis under-prediction rate
(Huber full run: 49.9% vs baseline 44.5%). Dev200 QLIKE GARCH-GRU cut crisis
under-rate 53%→36% at ~30% MAE cost.

## Update 2026-07-10 — J02 bake-offs complete

- Binary champions: stocks=lstm (val 0.8022/crisis 0.6596), etfs=gru
  (0.7367/0.5878), market=baseline (0.6597/0.3091).
- Three-class balanced-weight champions in `J02v2_vol_regime3/champions.json`:
  stocks=lstm (0.4605/0.3897), etfs=gru (0.4939/0.3957), market=lstm
  (0.4971/0.2122; all market neural candidates predicted zero COVID crisis days).

## What was implemented (all merged, all tested)

| Component | Location | Status |
|---|---|---|
| GARCH(1,1)-GRU hybrid cell (`GarchGruCell`, constrained ω/α/β via softplus, neural param modulation, vol-scale calibration head) | `scripts/tf_notebook_utils.py` | trained on both universes |
| QLIKE loss (`loss="qlike"`) + `asymmetric_error_metrics()` (under/over-prediction MAE) | `scripts/tf_notebook_utils.py` | in all regression notebooks + payloads |
| `garchgru` registered as architecture letter **f** (regression/vol_risk only) | `scripts/ml_iteration2_jobs.py`, scaffold, replay regex | 01f notebooks generated + trained |
| Scaffold `--architecture` flag (regenerate one candidate without touching trained notebooks) | `scripts/scaffold_ml_iteration2_notebooks.py` | working |
| SPCI-style conformal intervals (block bootstrap, asymmetric quantiles) | `scripts/conformal.py` + `ML_Iteration_2/00_orchestration/00b_conformal_risk_bands.ipynb` | executed; 6 `*_conformal.json` saved |
| MMD drift monitor (RBF kernel, permutation-calibrated threshold) | `scripts/drift_monitor.py` | validated on dev200 splits |
| 3-class regime target `vol_regime3_21` (0=bull, 1=bear, 2=crisis @ train q90 fwd vol) | `scripts/run_preprocessing.py` | SPY-lane bundle built + sanity-trained |
| Multiclass classifier support (`n_classes=3`, softmax + sparse CCE) | `scripts/tf_notebook_utils.py` | binary path unchanged |
| QLIKE side-experiment trainer (saves `{arch}_qlike_*` artifacts) with GPU `LD_LIBRARY_PATH` re-exec fix | `scripts/train_qlike_variants.py` | 4 runs complete |

## Key results (dev200)

### J01 stocks — NEW CHAMPION: garchgru (recorded in `J01_vol_risk/champions.json`)

| Model | Val MAE | Crisis MAE | Notes |
|---|---|---|---|
| baseline (ridge) | 0.015471 | 0.026955 | |
| xgb (prev champion) | **0.014261** | 0.028041 | loses to baseline in crisis |
| **garchgru** | 0.014570 | **0.026794** | only model beating baseline on BOTH; rubric priority 2 |

### J01 etfs — champion unchanged: xgb

garchgru etfs: val 0.004783 / crisis 0.013101 (beats baseline both; LSTM best crisis 0.012976, xgb best val 0.004254).

### QLIKE experiment (`*_qlike_*_predictions.json`)

- Plain GRU **destabilizes** under QLIKE (stocks val MAE 0.051; etfs diverged, 0.248).
- GARCH-GRU absorbs it: crisis under-prediction rate drops 53%→36% (stocks), 61%→41% (etfs) at ~10–20% MAE cost. Candidate "conservative variant" for the product risk engine; not champion on MAE.

### Conformal coverage (90% target, calibrated 2018-19, tested Q1-2020)

All models ≈ **74% empirical coverage** on COVID window, 15–17% undercoverage → the coverage gap is a model-agnostic regime-shift detector. Product rule of thumb documented in `00b_conformal_risk_bands.ipynb` (gap >0.15 → reduce-risk mode).

### MMD drift (stocks dev200, last-day features, threshold = 99th pct null = 0.00228)

val 2018-19: MMD²=0.0071 (mild) | COVID Q1-20: MMD²=0.0414 (**18× threshold**).

### 3-class regime sanity (SPY lane, `spy_vol_regime3_21_sequences_plus5.npz`)

Train class rates 61/29/10 (bull/bear/crisis); COVID test = 55% crisis days. Unweighted GRU beats majority class on crisis macro-F1 (0.188 vs 0.158) but collapses to bull-heavy predictions → **needs class weights for the real bake-off**.

## Operational gotchas

- **Plain-script training falls back to CPU**: TF needs `LD_LIBRARY_PATH` before process start. Jupyter kernel has it; scripts must re-exec (see `_reexec_with_cuda_libs` in `train_qlike_variants.py`). Symptom: "Cannot dlopen some GPU libraries".
- **GarchGruCell is not cuDNN-fused** → ~3.5 min/epoch on stocks dev200 (vs seconds for plain GRU). Stocks 01f best epoch was 2; use `PATIENCE=8` (not 15) for garchgru re-runs.
- `nbconvert --inplace` rewrites notebooks on disk; reload IDE tabs to see outputs.
- QLIKE operates on variance: targets/preds squared inside the loss (`qlike_loss`), predictions clamped via relu + eps.

## Next steps (agreed order)

1. **J02 binary bake-off** (user's active job) — run `00a` then `02a–02e` for market/stocks/etfs, pick champions (`macro_f1` rubric). Untouched by this session.
2. **J02-v2 3-class bake-off** — build pooled stocks/etfs `vol_regime3_21` bundles (`run_preprocessing.py --target-type vol_regime3_21`), add class weights to `_compile_classifier`/fit, scaffold + train.
3. Optional: full-universe re-run of stocks champion (`01f`, `RUN_MODE="full"`, drop patience to 8).
4. Orchestration (`00a_ensemble_product_scores.ipynb` still scaffold): merge champion scores + conformal bands + MMD health check into product outputs.

## Research backlog (not scheduled)

- **PCA (not used anywhere in the pipeline today; deliberate — 15 interpretable features, and
  `GarchGruCell` needs the raw `log_return` column at a known index):
  1. Cross-sectional PCA over the universe return panel to extract market/sector factors as
     features; PC1 explained-variance ratio spikes in crises (extra regime signal alongside MMD).
  2. Diagnostic PCA on the 15 engineered features to flag near-redundant columns
     (`abs_return` / `squared_return` / `lag_abs_return_1` are correlated by construction).
- Foundation-model (TimesFM/Kronos-style) scoring for the low-history universe slice excluded
  by the 756-day filter.

## Artifact inventory added this session (`data/models/`)

- `garchgru_vol_risk_{stocks,etfs}_dev200_tf_{best,final}.keras` + `_predictions.json`
- `{gru,garchgru}_qlike_vol_risk_{stocks,etfs}_dev200_tf_predictions.json` (+ `_best.keras`)
- `{garchgru,xgb,baseline}_vol_risk_{stocks,etfs}_dev200_tf_conformal.json`
- `data/processed/spy_vol_regime3_21_*` (3-class SPY bundle)
