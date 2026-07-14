# Rebuild Implementation Plan — ML Iteration 2

**Last updated:** 2026-07-12  
**Project root:** `~/ai-workspace/experiments/stock-prediction`  
**Companion docs:** `SESSION_HANDOFF.md`, `README.md`, `docs/DATASET.md`, `.cursor/rules/full-universe-disk-backed.mdc`

This document records work completed to date, trained models and bake-offs, artifact inventory, operational procedures, and the recommended path forward. **§2** states the problems this rebuild must resolve; **§12** maps them to phased experiments and preprocessing changes.

---

## 1. Executive summary

| Area | Status |
|------|--------|
| **J01 vol_risk dev200** | Complete. Champion: **GARCH-GRU** (stocks), **XGB** (etfs). |
| **J01 vol_risk full (stocks)** | Partial. **Huber GARCH-GRU** + **Ridge baseline** trained on 1,838-symbol memmap bundle. GARCH-GRU **does not beat full baseline on crisis MAE** → no promotion. |
| **J01 QLIKE full (stocks)** | In progress / interrupted. Notebook configured (`LOSS="qlike"`); partial epoch checkpoints exist; no final `*_predictions.json` yet. |
| **J02 vol_regime dev200** | Complete. Champions: **LSTM** (stocks), **GRU** (etfs), **baseline** (market). |
| **J02v2 vol_regime3 dev200** | Trained (all lanes). No `champions.json` committed; best models per handoff below. |
| **Product layers** | SPCI conformal (dev200 J01), MMD drift monitor, orchestration scaffold only. |
| **Full-universe pipeline** | Disk-backed `memmap_v1` implemented, tested, bundle rebuilt (~25 GB). |

**Primary blocker for full-universe J01 promotion:** crisis-window performance — full GARCH-GRU loses to full Ridge baseline on crisis MAE and crisis top-10% vol AUC, despite better val MAE.

---

## 2. Problems to resolve

This rebuild is not only about scaling to 1,838 symbols. The full-universe bake-off exposed **model and target limitations** that a like-for-like data rebuild (same `abs_return` target, same features) will **not** fix on its own. The sections below define what we are trying to solve; Phases 1–4 in §12 map each problem to experiments, preprocessing changes, and product mitigations.

### 2.1 Problem inventory

| # | Problem | Evidence (full stocks, Huber GARCH-GRU) | Why it matters |
|---|---------|------------------------------------------|----------------|
| P1 | **Tail level error** — MAE blows up on highest-vol days | Crisis p95 tail MAE 0.1617 vs baseline 0.1459; p99 0.297 vs 0.278 | Point forecasts underestimate magnitude on stress days |
| P2 | **Tail detection / ranking** — poor separation of extreme vol days | Crisis top-10% AUC 0.790 vs baseline 0.841 (dev200 GARCH-GRU was 0.836) | Product cannot reliably flag “tomorrow is a top-decile vol day” |
| P3 | **Crisis under-prediction bias** | Crisis under-prediction rate 49.9% vs baseline 44.5% | Systematic low bias when risk is highest |
| P4 | **High-volume + high-vol joint misses** | Volume is a feature only; no target or loss emphasizes large-volume stress days | User-observed poor behavior on heavy-volume turbulent sessions |
| P5 | **Promotion gate failure vs full baseline** | Beats baseline val MAE but loses crisis MAE (0.02611 vs 0.02591) | Cannot promote dev200 champion to full-universe production |
| P6 | **Regime-shift uncertainty** | Conformal COVID coverage ~74% vs 90% target; MMD² 18× threshold in crisis | Bands and point scores untrustworthy without gating in COVID-like windows |
| P7 | **Decile calibration in the tail** | Mid-decile calibration is good; errors and ranking degrade in top realized-vol deciles | Model is “average-accurate” but not tail-accurate |

### 2.2 What each mitigation layer actually fixes

| Layer | Addresses | Does **not** replace |
|-------|-----------|----------------------|
| **QLIKE loss** (Phase 0/1e) | P3 under-prediction bias | P1 tail MAE, P2 tail AUC |
| **SPCI conformal** (Phase 0/3) | P6 product uncertainty; conservative upper bound | P2 ranking, P1 point tail MAE |
| **MMD drift gate** (Phase 3) | P6 regime-shift alarm | Model retraining |
| **Rebuild same `abs_return` bundle** | Scale, data quality, symbol coverage | P1–P4 without target/loss change |
| **Granular target + preprocessing rebuild** (Phase 2) | P1, P2, P4 at source | Nothing — this is the structural fix |
| **Classifier + regressor ensemble** (Phase 1/3) | P2 detection + P1 level jointly | Still needs good components |

**Important:** Dropping high-vol **return days** or relying on `iqr_upper` symbol filtering does **not** address P1–P4 (see §11.3 and §12 Phase 2).

### 2.3 Granularity strategy (three axes)

The scalar next-day `abs_return` target optimizes **mean error**, not tail ranking or tail level. Granularity means enriching the problem along three axes:

| Axis | Current | Proposed options | Primary problems helped |
|------|---------|------------------|-------------------------|
| **Temporal** | 1-day `\|log return\|` | 21d forward realized vol (`forward_vol`) | P1 noise; smoother vol signal |
| **Categorical** | Continuous regression only | `vol_regime_21`, `vol_regime3_21` (bull/bear/crisis) | P2 tail detection; P5 crisis behavior |
| **Distributional** | Single point forecast | Quantile / pinball heads (τ=0.5, 0.9, 0.95); log-variance target | P1 tail level; aligns with GARCH + QLIKE |

Supporting preprocessing (Phase 2, no outlier purge on returns):

- De-redundant features (`abs_return` / `squared_return` / `lag_abs_return_1` cluster)
- Cross-sectional vol rank and volume surge percentile (P4)
- Volume–volatility interaction terms (P4)

### 2.4 Success criteria (when we consider the rebuild “done”)

Aligned with §12 Phase 4 promotion gate — a new full-universe J01 stocks champion must:

- [ ] Beat full baseline on **val MAE** and **crisis MAE**
- [ ] Match or beat full baseline **crisis top-10% AUC** (≥ 0.841 today)
- [ ] Match or beat full baseline **crisis p95 tail MAE** (≤ 0.1459 today)
- [ ] Reduce crisis **under-prediction rate** vs Huber full (below 49.9%) without unacceptable val MAE cost
- [ ] Document **conformal coverage gap** on full predictions
- [ ] Demonstrate dev200 lift **before** full memmap rebuild for any new target type

---

## 3. Timeline of major work

| Date | Milestone |
|------|-----------|
| 2026-06-30 | J01 dev200 baselines + candidates trained (stocks, etfs). |
| 2026-07-01 | J01 stocks dev200 champion: XGB → later superseded. J02 market baseline. |
| 2026-07-09 – 07-10 | GARCH-GRU cell, QLIKE loss, conformal (SPCI), MMD drift, `01f` notebooks, dev200 QLIKE side-runs. |
| 2026-07-10 | J01 stocks dev200 champion: **GARCH-GRU** (only model beating baseline on val + crisis MAE). J02 binary bake-off complete. |
| 2026-07-11 | Full-universe audit: stale 2-symbol bundle invalidated. Disk-backed memmap pipeline implemented. 1,838-symbol bundle rebuilt. Full Huber GARCH-GRU trained (epoch 12, graceful interrupt). |
| 2026-07-11 – 07-12 | Full Ridge baseline trained. Full bake-off vs GARCH-GRU documented. Notebook prepared for manual QLIKE full run. |

---

## 4. Infrastructure implemented

### 4.1 Disk-backed full-universe pipeline

**Policy:** `.cursor/rules/full-universe-disk-backed.mdc` (always applied)

| Component | Path | Role |
|-----------|------|------|
| Streaming preprocessing | `scripts/run_preprocessing.py` | Two-pass memmap build, atomic staging/publish |
| Lazy loading | `scripts/tf_data.py` | `memmap_v1` manifest; prefers directory over stale NPZ |
| Chunked TF batches | `scripts/tf_notebook_utils.py` | `make_tf_datasets`, `predict_in_batches` |
| Baseline batch predict | `scripts/tf_notebook_utils.py` | Ridge wrapper predicts in 65k chunks |
| Graceful stop | `scripts/graceful_stop_notebook_training.py` | SIGINT + `stop_training` for long notebook runs |
| Tests | `tests/test_disk_backed_pipeline.py` | Parity, batching, compatibility (4 tests pass) |
| Docs | `docs/DATASET.md` | Storage contract, rebuild commands |

**Never do on full universe:** `np.concatenate` all symbols, `.astype()` on full memmap tensors, `from_tensor_slices` on complete arrays.

### 4.2 Model / product utilities

| Component | Path |
|-----------|------|
| GARCH(1,1)-GRU hybrid cell | `scripts/tf_notebook_utils.py` (`GarchGruCell`) |
| QLIKE loss | `scripts/tf_notebook_utils.py` (`qlike_loss`) |
| Asymmetric error metrics | `scripts/tf_notebook_utils.py` |
| QLIKE parallel trainer | `scripts/train_qlike_variants.py` |
| SPCI conformal bands | `scripts/conformal.py` |
| Conformal notebook | `ML_Iteration_2/00_orchestration/00b_conformal_risk_bands.ipynb` |
| MMD drift monitor | `scripts/drift_monitor.py` |
| Compare bake-offs | `scripts/compare_job_models.py` |
| Replay notebook outputs | `scripts/replay_notebook_outputs.py` |
| Notebook scaffold | `scripts/scaffold_ml_iteration2_notebooks.py` |

### 4.3 Canvases (audit artifacts)

| Canvas | Purpose |
|--------|---------|
| `canvases/j01-garchgru-full-performance-audit.canvas.tsx` | Huber full-run diagnostics |
| `canvases/j01-full-universe-bakeoff-comparison.canvas.tsx` | Full baseline vs GARCH-GRU scorecard |

---

## 5. Data bundles

### 5.1 Full-universe stocks (J01 primary)

| Property | Value |
|----------|-------|
| Path | `data/processed/stocks_universe_sequences_plus5/` |
| Format | `memmap_v1` |
| Symbols | **1,838** |
| Train / val / test sequences | **6,400,094 / 920,149 / 113,705** |
| Lookback | 60 days |
| Features | 15 (plus5 profile) |
| Target | Next-day `abs_return` (`\|log return\|`) |
| Train end | ≤ 2017-12-31 |
| Val | 2018-01-01 – 2019-12-31 |
| Crisis test | 2020-01-01 – 2020-04-01 (Q1 COVID) |
| Config | `data/processed/pipeline_config_stocks_universe_plus5.json` |
| Scaler | `data/processed/scaler_stocks_universe_plus5.json` |
| Approx size | ~25 GB |

### 5.2 Dev200 stocks / etfs (fast iteration)

| Bundle | Path pattern |
|--------|----------------|
| Stocks dev200 NPZ | `data/processed/stocks_universe_dev200_sequences_plus5.npz` |
| ETFs dev200 NPZ | `data/processed/etfs_universe_dev200_sequences_plus5.npz` |

Dev200 effective symbol count in config is often **126** (quality-filtered subset), not 200.

### 5.3 Regime bundles (J02 / J02v2)

| Target type | Dev200 bundles exist for |
|-------------|--------------------------|
| `vol_regime_21` | stocks, etfs (+ SPY market lane) |
| `vol_regime3_21` | stocks, etfs, SPY market (sanity / bake-off) |

### 5.4 Invalid / stale artifacts (do not use)

| Artifact | Issue |
|----------|-------|
| `data/models/gru_vol_risk_stocks_tf_predictions.json` | `n_symbols=2` (AAL/AAME only) — pre-rebuild stale run |
| Any full-tag model trained before 2026-07-11 memmap rebuild | Unverified universe coverage |

Backups of pre-recovery Huber full run: `data/models/.backup_garchgru_vol_risk_stocks_tf_20260711T*`

---

## 6. Job structure and bake-off rubrics

### 6.1 Notebook naming

```text
00a_baseline_{job_slug}_{universe}.ipynb     ← train first
{01|02}{a-f}_{architecture}_{job_slug}_{universe}.ipynb
```

| Letter | Architecture | Jobs |
|--------|--------------|------|
| 00a | Ridge / Logistic baseline | All |
| a | GRU | All |
| b | TCN | All |
| c | LSTM | All |
| d | CNN1D | All |
| e | XGB | All |
| f | GARCH-GRU | J01 vol_risk regression only |

### 6.2 Jobs

| Job | Slug | Task | Target | Universes | Phase |
|-----|------|------|--------|-----------|-------|
| J01 | `vol_risk` | Regression | `abs_return` (next-day \|log ret\|) | stocks, etfs | P1 |
| J02 | `vol_regime` | Binary classification | `vol_regime_21` | stocks, etfs, market | P3 |
| J02v2 | `vol_regime3` | 3-class classification | `vol_regime3_21` (bull/bear/crisis) | stocks, etfs, market | P3+ |
| J03 | `rel_strength` | Regression (scaffold) | 21d excess vs SPY | stocks | P4 |
| J04 | `etf_opportunity` | Regression (scaffold) | 21d excess vs SPY | etfs | P5 |

### 6.3 Selection rubrics

**J01 vol_risk (regression):**
1. `beats_baseline_val_mae`
2. `val_mae` (lower better)
3. `crisis_mae` (lower better)
4. `top10_vol_auc` (higher better)

**J02 / J02v2 (classification):**
1. `beats_baseline_val_macro_f1`
2. `val_macro_f1`
3. `crisis_macro_f1`

---

## 7. Trained models and performance

All metrics from `data/models/*_predictions.json` unless noted. Crisis = COVID Q1 2020 test window.

### 7.1 J01 vol_risk — stocks dev200 bake-off

| Model | Val MAE | Crisis MAE | Best epoch | Notes |
|-------|---------|------------|------------|-------|
| **xgb** | **0.014261** | 0.028041 | 1 | Best val; loses crisis vs baseline |
| lstm | 0.014395 | 0.027217 | 2 | |
| gru | 0.014461 | 0.027735 | 5 | |
| **garchgru** ★ | 0.014570 | **0.026794** | 2 | **Dev200 champion** — only model beating baseline on BOTH |
| cnn1d | 0.014785 | 0.026955 | 6 | |
| tcn | 0.015230 | 0.030331 | 16 | |
| baseline | 0.015471 | 0.026955 | 1 | Ridge last-day |
| garchgru_qlike | 0.018919 | 0.029056 | 6 | Crisis under-rate 35.8% (vs 52.8% Huber) |
| gru_qlike | 0.050582 | 0.055271 | 30 | Unstable under QLIKE |

★ Recorded in `J01_vol_risk/champions.json`

### 7.2 J01 vol_risk — stocks full-universe bake-off

| Model | Val MAE | Crisis MAE | Val AUC@90 | Crisis AUC@90 | Under-rate (crisis) | Best epoch |
|-------|---------|------------|------------|---------------|---------------------|------------|
| **garchgru** | **0.013527** | 0.026107 | **0.807** | 0.790 | 49.9% | 12 |
| **baseline** | 0.014270 | **0.025908** | 0.775 | **0.841** | 44.5% | 1 |
| persistence (untrained) | 0.017800 | 0.033200 | — | — | — | — |

**Verdict:** GARCH-GRU wins val MAE (−5.2% vs baseline) but **loses crisis MAE** (+0.8%). **`promotion_eligible: false`**. Dev200 GARCH-GRU champion retained.

**Tail MAE (crisis test, full GARCH-GRU):**

| Quantile threshold | Tail MAE | n |
|--------------------|----------|---|
| p90 | 0.1161 | 11,371 |
| p95 | 0.1617 | 5,686 |
| p99 | 0.2970 | 1,139 |

**Tail MAE (crisis test, full baseline):** p90 0.1026, p95 0.1459, p99 0.2781 — baseline wins at all reported tails.

### 7.3 J01 vol_risk — etfs dev200 bake-off

| Model | Val MAE | Crisis MAE | Best epoch |
|-------|---------|------------|------------|
| **xgb** ★ | **0.004254** | **0.013500** | 1 |
| lstm | 0.004493 | 0.012976 | 17 |
| gru | 0.004686 | 0.013030 | 28 |
| garchgru | 0.004783 | 0.013101 | 6 |
| cnn1d | 0.005091 | 0.013618 | 5 |
| baseline | 0.005475 | 0.014376 | 1 |
| garchgru_qlike | 0.005737 | 0.014424 | 27 |

★ ETF champion: **xgb** (`champions.json`)

### 7.4 J02 vol_regime — dev200 champions

| Universe | Champion | Val macro-F1 | Crisis macro-F1 |
|----------|----------|--------------|-----------------|
| stocks | **lstm** | 0.8022 | 0.6596 |
| etfs | **gru** | 0.7367 | 0.5878 |
| market | **baseline** | 0.6597 | 0.3091 |

Full scorecards in `J02_vol_regime/champions.json`. Market lane: no NN beats baseline on val; SPY-only with n≈503 val / n≈42 crisis.

### 7.5 J02v2 vol_regime3 — dev200 (trained, champions not in JSON)

| Universe | Best val macro-F1 | Best crisis macro-F1 | Notes |
|----------|-----------------|----------------------|-------|
| stocks | gru 0.4654 | lstm 0.3897 | Balanced-weight training |
| etfs | gru 0.4939 | gru 0.3957 | |
| market | lstm 0.4971 | baseline 0.2638 | Neural models predict few crisis days |

---

## 8. Product-layer artifacts (post-hoc)

### 8.1 SPCI conformal (dev200 J01 — 90% target)

| Artifact | COVID coverage | Coverage gap | Undercoverage rate |
|----------|----------------|--------------|-------------------|
| garchgru_vol_risk_stocks_dev200 | 0.740 | 0.160 | 0.155 |
| baseline_vol_risk_stocks_dev200 | 0.733 | 0.167 | 0.145 |
| xgb_vol_risk_stocks_dev200 | 0.737 | 0.163 | 0.170 |
| garchgru_vol_risk_etfs_dev200 | 0.675 | 0.225 | 0.205 |

**Product rule** (`00b_conformal_risk_bands.ipynb`): coverage gap > 0.15 → reduce-risk mode; 0.05–0.15 → widen bands.

**Not yet run on full-universe predictions.**

### 8.2 MMD drift (stocks dev200, last-day features)

| Split | MMD² | vs threshold (0.00228) |
|-------|------|------------------------|
| Val 2018–19 | 0.0071 | Mild |
| COVID Q1 20 | 0.0414 | **~18× threshold** |

---

## 9. Artifact inventory

### 9.1 Predictions JSON (`data/models/`)

**63** `*_predictions.json` files (excluding `.backup_*`). Key full-universe stocks:

| File | Status |
|------|--------|
| `garchgru_vol_risk_stocks_tf_predictions.json` | ✅ Authoritative Huber full run |
| `baseline_vol_risk_stocks_tf_predictions.json` | ✅ Full baseline |
| `garchgru_qlike_vol_risk_stocks_tf_predictions.json` | ❌ Not yet saved (training in progress) |
| `gru_vol_risk_stocks_tf_predictions.json` | ❌ Invalid (2 symbols) |

### 9.2 Model checkpoints

| Type | Count | Notes |
|------|-------|-------|
| `.keras` | 95 | Includes `garchgru_vol_risk_stocks_tf_best.keras`, epoch checkpoints |
| `.joblib` | 36 | Baselines + XGB |

**Full Huber GARCH-GRU checkpoints:** `garchgru_vol_risk_stocks_tf_best.keras`, `garchgru_vol_risk_stocks_tf_final.keras`, `garchgru_vol_risk_stocks_tf_epoch*.keras`

**Partial QLIKE full checkpoints:** `garchgru_qlike_vol_risk_stocks_tf_epoch*.keras` (interrupted runs; best epoch TBD)

### 9.3 Conformal JSON

6 files: `{garchgru,xgb,baseline}_vol_risk_{stocks,etfs}_dev200_tf_conformal.json`

### 9.4 Champion records

| File | Status |
|------|--------|
| `ML_Iteration_2/J01_vol_risk/champions.json` | Updated with full bake-off + baseline full |
| `ML_Iteration_2/J02_vol_regime/champions.json` | Complete |
| `ML_Iteration_2/J02v2_vol_regime3/champions.json` | **Missing** — promote from bake-off tables |

### 9.5 Naming convention

```text
{architecture}_{job_slug}_{universe}_{dataset_tag}_tf_predictions.json
{architecture}_{job_slug}_{universe}_{dataset_tag}_tf_best.keras

# Full universe: dataset_tag omitted in filename
garchgru_vol_risk_stocks_tf_predictions.json          # full
garchgru_vol_risk_stocks_dev200_tf_predictions.json   # dev200

# QLIKE parallel artifacts
garchgru_qlike_vol_risk_stocks_tf_predictions.json
```

---

## 10. Notebook run configuration (current)

### J01 GARCH-GRU stocks (`01f_garchgru_vol_risk_stocks.ipynb`)

| Setting | Value |
|---------|-------|
| `RUN_MODE` | `"full"` |
| `LOSS` | `"qlike"` (manual run; Huber artifacts separate) |
| `ARTIFACT_STEM` | `garchgru_qlike_vol_risk_stocks_tf` when `LOSS="qlike"` |
| `PATIENCE` | 8 |
| Recovery cell 4b | Uses `ARTIFACT_STEM`; skip on fresh QLIKE run |

**Replay Huber without retrain:** set `LOSS="huber"`, run section 4b only.

### J01 baseline stocks (`00a_baseline_vol_risk_stocks.ipynb`)

| Setting | Value |
|---------|-------|
| `RUN_MODE` | `"full"` |
| Artifacts | `baseline_vol_risk_stocks_tf_*` |

---

## 11. Diagnosis: why full universe “feels worse”

*Expands on §2 problem inventory with measured comparisons and anti-patterns.*

### 11.1 Absolute vs relative performance

| Comparison | Result |
|------------|--------|
| Full GARCH-GRU vs dev200 GARCH-GRU (MAE) | **Better** on val and crisis |
| Full GARCH-GRU vs full baseline (MAE) | Wins val, **loses crisis** |
| Full GARCH-GRU vs dev200 GARCH-GRU (crisis AUC@90) | **Worse** (0.790 vs 0.836) |
| Full baseline vs dev200 baseline (crisis) | **Much better** (0.02591 vs 0.02696) |

Scaling to 1,838 symbols **raised the baseline bar** more than GARCH-GRU gained on crisis ranking.

### 11.2 Failure modes (distinct problems)

| Symptom | Metric | Root cause (likely) |
|---------|--------|---------------------|
| Systematic low bias in crisis | Under-prediction rate 49.9% | Mean-regression objective; Huber symmetric |
| Poor extreme-day ranking | Crisis AUC@90 0.790 | Single scalar `abs_return` target optimizes MAE not ranking |
| Poor extreme-day level error | Tail MAE p99 0.297 | Same; + heteroscedastic tails underrepresented in loss |
| Product uncertainty | Conformal gap ~0.16 | Regime shift COVID ≠ calibration period |

### 11.3 What does **not** fix this

| Approach | Why insufficient alone |
|----------|------------------------|
| **QLIKE** | Reduces under-prediction bias; dev200 showed ~30% MAE cost; unlikely to fix tail AUC |
| **SPCI conformal** | Post-hoc bands around point forecast; does not retrain ranking/level |
| **`iqr_upper` symbol filter** | Removes long-history tickers only — not high-vol **days** |
| **Dropping return outliers** | Would remove the tail events you need to learn |

---

## 12. Recommended rebuild strategy

*Maps §2 problems (P1–P7) to phased work.*

### Phase 0 — Complete in-flight work (no rebuild)

| Step | Action | Effort |
|------|--------|--------|
| 0.1 | Finish **QLIKE GARCH-GRU full** from `01f` notebook (`LOSS="qlike"`, sections 0→6, skip 4b) | Hours (GPU) |
| 0.2 | Compare QLIKE vs Huber vs baseline on crisis under-rate + crisis MAE | Minutes |
| 0.3 | Run **conformal** on full Huber + baseline predictions (`00b` notebook pattern) | Hours |
| 0.4 | Promote J02v2 champions to `champions.json` | Trivial |

### Phase 1 — Model-side tail fixes (dev200 first)

**Principle:** Split *detection* (is tomorrow high-vol?) from *level* (how high?).

| ID | Experiment | Target / loss | Fixes | Notebook / script |
|----|------------|---------------|-------|-------------------|
| 1a | High-vol **classifier** on same universe | `vol_regime_21` or top-decile label | Tail AUC | J02 notebooks on stocks dev200 (already trained — **ensemble candidate**) |
| 1b | **Quantile / pinball** head | τ=0.90, 0.95 vol | Tail level MAE | New loss in `tf_notebook_utils.py` + `01f` variant |
| 1c | **Multi-task** GARCH-GRU | Huber + BCE top-decile | Both | Extend `01f` training cell |
| 1d | **Train sample weights** by vol decile | Upweight top 10–20% | Tail without rebuild | `make_tf_datasets(..., train_sample_weight)` — already supported |
| 1e | **QLIKE full** (in flight) | QLIKE | Under-prediction bias | `01f` with `LOSS="qlike"` |

**Success criteria (dev200):** beat baseline on crisis MAE **and** crisis top-10% AUC **and** p95 tail MAE.

### Phase 2 — Target & preprocessing rebuild (dev200 validate → full promote)

**Do not drop high-vol return days.** Improve representation:

| ID | Change | Rationale |
|----|--------|-----------|
| 2a | Target → **21d forward realized vol** (`forward_vol` already computed for J02) | Smoother vol signal; less 1-day noise |
| 2b | Target → **`vol_regime3_21`** (bull/bear/crisis) for a parallel J01b job | Crisis class = tail; direct AUC optimization |
| 2c | **De-redundant features** | Drop one of `abs_return` / `squared_return` / `lag_abs_return_1` cluster |
| 2d | **Cross-sectional features** | Symbol vol rank vs universe; volume surge percentile — helps “large volume + high vol” days |
| 2e | **Volume–vol interactions** | e.g. `volume_surge × volatility_21` |
| 2f | Target → **log-variance** `log(r²)` | Aligns with GARCH cell + QLIKE; stabilizes scale |

**`history-outlier-filter`:** keep `none` or `iqr_upper` for symbol hygiene only — **not** a tail fix.

**Rebuild command template (new target):**

```bash
python scripts/run_preprocessing.py \
  --asset-type stock \
  --profile plus5 \
  --symbol-scope all \
  --target-type <NEW_TARGET> \
  --storage-format memmap \
  --history-outlier-filter none
```

Validate on dev200 NPZ first; full memmap rebuild only after dev200 lift confirmed.

### Phase 3 — Product architecture (parallel track)

```
Stage A: high-vol classifier (J02 / J02v2 champion)
Stage B: magnitude model (J01 GARCH-GRU Huber or QLIKE)
Stage C: SPCI upper conformal band on Stage B
Stage D: MMD drift gate → widen bands / reduce-risk mode

Product risk_score = f(upper_conformal, classifier_prob, coverage_gap)
```

This addresses product safety **before** tail MAE is fully solved at the point-forecast layer.

### Phase 4 — Full-universe promotion gate

Promote a new J01 stocks full-universe champion only when:

- [ ] `beats_baseline_val_mae` on full bundle
- [ ] `beats_baseline_crisis_mae` on full bundle
- [ ] Crisis top-10% AUC ≥ full baseline (0.841) or justified trade-off documented
- [ ] Crisis p95 tail MAE ≤ full baseline (0.1459 on current crisis test)
- [ ] Conformal coverage gap documented on full predictions
- [ ] `champions.json` + `SESSION_HANDOFF.md` updated

---

## 13. Proposed new job slots (future scaffold)

| Job | Slug | Target | Purpose |
|-----|------|--------|---------|
| J01b | `vol_risk_quantile` | Pinball τ∈{0.5,0.9} | Tail level forecasts |
| J01c | `vol_risk_fwd21` | 21d forward realized vol | Smoother regression target |
| J05 | `vol_ensemble` | Orchestration | Classifier + regressor + conformal fusion |

---

## 14. Operational procedures

### Full-universe training checklist

1. Verify `manifest.json`: `n_symbols=1838`, split shapes match config.
2. Kernel: **`Python (stock-prediction)`**.
3. `RUN_MODE="full"`, `FORCE_REBUILD_DATA=False` if bundle exists.
4. Never `.astype()` full memmap tensors in notebooks.
5. Use `predict_in_batches` for evaluation.
6. `PATIENCE=8` for GARCH-GRU; expect ~12,501 steps/epoch stocks full.
7. First GPU epoch may take 10–30+ min (CUDA JIT on RTX 5090).
8. Graceful stop: `scripts/graceful_stop_notebook_training.py` if needed.
9. Save section 6 before closing kernel.

### Compare / replay

```bash
# Dev200 bake-off
python scripts/compare_job_models.py --job-slug vol_risk --universe stocks --dataset-tag dev200

# Replay dev200 notebook outputs
python scripts/replay_notebook_outputs.py --notebook ML_Iteration_2/J01_vol_risk/stocks/01f_garchgru_vol_risk_stocks.ipynb --dataset-tag dev200

# Full-universe replay: use notebook section 4b with LOSS="huber" (replay script expects _full_ suffix — known gap)
```

---

## 15. Open gaps and tech debt

| Gap | Priority |
|-----|----------|
| `replay_notebook_outputs.py` full-tag path mismatch (`_full_` vs empty tag) | Medium |
| `compare_job_models.py` same glob issue for full universe | Medium |
| Conformal not run on full-universe predictions | High (product) |
| J02v2 `champions.json` missing | Low |
| `gru_vol_risk_stocks_tf_predictions.json` stale 2-symbol file should be archived | Low |
| J03/J04 scaffold only | Future |
| `00a_ensemble_product_scores.ipynb` orchestration scaffold | High (product) |
| Full QLIKE run incomplete | In progress |
| Research backlog: PCA factors, foundation models for low-history symbols | Future |

---

## 16. Decision log

| Decision | Date | Rationale |
|----------|------|-----------|
| Mandatory disk-backed full-universe policy | 2026-07-11 | OOM at ~45.8 GiB RSS on old pipeline |
| Dev200 GARCH-GRU remains champion after full bake-off | 2026-07-12 | Full GARCH-GRU fails crisis MAE gate |
| QLIKE as parallel artifact stem (`garchgru_qlike_*`) | 2026-07-12 | Preserve Huber full run; avoid overwrite |
| Single notebook `LOSS` switch for Huber vs QLIKE | 2026-07-12 | Manual training from `01f` |
| No full-universe champion promotion until crisis metrics fixed | 2026-07-12 | Rubric + product risk |

---

## 17. Immediate next actions (ordered)

1. **Complete QLIKE full run** — `01f`, `LOSS="qlike"`, sections 0→6.
2. **Score QLIKE** vs Huber + baseline (crisis MAE, under-rate, tail MAE, AUC@90).
3. **Conformal on full predictions** — extend `00b` to full artifacts.
4. **Dev200 experiment:** multi-task or quantile head on GARCH-GRU (Phase 1b/1c).
5. **Prototype J01+J02 ensemble** on dev200 (classifier gates magnitude).
6. **If dev200 lifts tails:** rebuild preprocessing with `forward_vol` or `vol_regime3_21` target (Phase 2).
7. **Full memmap rebuild** only after dev200 validation.
8. Update `champions.json` when §12 Phase 4 promotion gate clears.

---

*This plan should be updated after QLIKE full completion, conformal full run, and any preprocessing/target experiments.*
