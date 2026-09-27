# Stock Volatility Prediction — Project Report & Rebuild Guide

**Date:** 2026-07-17
**Audited root:** `~/ai-workspace/experiments/stock-prediction`
**Method:** every metric below was re-read from the artifacts on disk (`data/models/*_predictions.json`, `champions.json` files) — not copied from prior docs. Where prior docs disagree with artifacts, the artifact wins and the discrepancy is noted.
**Companion docs:** `ML_Iteration_2/REBUILD_IMPLEMENTATION_PLAN.md` (strategy, 2026-07-12), `ML_Iteration_2/SESSION_HANDOFF.md` (running log through 2026-07-14), `docs/DATASET.md` (data contract).

---

# PART A — PROJECT REPORT

## A1. Executive summary

The project set out to predict next-day stock *direction* (Iteration 1), discovered that direction is essentially unlearnable at this data scale (val AUC ≈ 0.50–0.56), and pivoted to a **volatility/risk product** (Iteration 2): forecast *how risky* tomorrow is, *which regime* the next month is in, and eventually *which names look relatively attractive* — packaged for novice investors as a risk engine, not a stock picker.

Where it stands today:

- **Four bake-off jobs** designed; **three fully trained on dev200** (J01 vol_risk regression; J02 binary regime; J02v2 three-class regime). J03/J04 (relative strength, ETF opportunity) are scaffolds.
- **63 prediction artifacts** on disk, champions recorded for all trained lanes.
- A **full-universe (1,838 stocks, 7.4M sequences) pipeline** exists and is disk-backed after an OOM incident; baseline + GARCH-GRU (Huber and QLIKE) trained on it.
- **The central open problem:** on the full universe, the neural champion beats the Ridge baseline on average error but **loses in the crisis window** — worse crisis MAE, much worse tail ranking (AUC@90 0.79 vs 0.84), and stronger under-prediction bias exactly when risk peaks. Per the promotion rubric, **no full-universe champion has been promoted**.
- **Product risk layers** (conformal bands, MMD drift gate) are built and quantified; the orchestration layer that fuses them into a product `risk_score` is still a scaffold.

The rebuild, therefore, is not "redo the same thing tidily." It is: fix the **tail problem** (detection, level, bias) via better targets/losses/ensembles, prove it on dev200, then re-earn full-universe promotion through an explicit gate — on top of a cleaned, reproducible repo.

## A2. Product & ML framing (unchanged, still sound)

| Layer | Purpose | Backing job |
|---|---|---|
| Risk engine | "High turbulence expected tomorrow" | J01 vol_risk |
| Regime gate | Market weather; reduce-risk mode in crisis-like regimes | J02 / J02v2 |
| Opportunity ranker | Names vs SPY over ~21d (watchlist) | J03 (scaffold) |
| ETF lane | Core/satellite allocation hints | J04 (scaffold) |
| Coach rules + uncertainty | Position limits, conformal bands, drift alarm | 00_orchestration |

Evaluation protocol (all jobs): train ≤ 2017-12-31 · validation 2018–2019 · stress test 2020 Q1 COVID. 60-day lookback, 15 features ("plus5" profile), pooled cross-symbol training. Champion rubric: must beat the trainable baseline on validation, then crisis metrics break ties.

## A3. Iteration history

| Iteration | Focus | Outcome |
|---|---|---|
| EDA / preprocessing (`Analytics notebooks/`) | Dataset cleaning, alignment, feature design | Reference; produced the cleaning report |
| **Iteration 1** (`04_ML_Model_Iteration-1/`) | AAPL next-day direction, BiGRU + TCN | v1 pathologies (BiGRU overfit train-val AUC gap ~0.40; TCN collapsed to constant output). v2 fixed the pathologies (TCN val AUC 0.559) but direction remained ~coin-flip. **Deprioritized** — the key strategic learning: predict learnable targets (volatility), not direction. |
| **Iteration 2** (`ML_Iteration_2/`) | Universe-scale vol/risk job portfolio with baseline-gated bake-offs | Active; detailed below |
| Full-universe scale-up | 1,838 stocks, memmap pipeline | Built after OOM failure; promotion blocked on crisis metrics |

## A4. Data layer

**Raw:** "Stock Market Dataset" (~2.6 GB OHLCV CSVs, stocks + ETFs + `symbols_valid_meta.csv`), not in git; resolved via `STOCK_DATASET_DIR` → project-root symlink → fallbacks in `scripts/paths.py`.

**Processed bundles:**

| Bundle | Format | Scale |
|---|---|---|
| dev200 stocks / etfs (per target: `abs_return`, `vol_regime_21`, `vol_regime3_21`) | compressed NPZ | 126 stocks / 127 ETFs effective (quality-filtered from 200); ~63.4k val rows pooled |
| SPY market lane (per target) | NPZ | 1 symbol; 503 val / 42 crisis rows |
| **Full stocks `abs_return`** | **`memmap_v1` directory** (~25 GB) | **1,838 symbols; 6,400,094 / 920,149 / 113,705 train/val/test** |

The `memmap_v1` policy (atomic staged publish, `.npy` split memmaps, streaming scaler moments, chunked TF datasets, manifest verification) is mandatory for full runs — codified in `.cursor/rules/full-universe-disk-backed.mdc` and `docs/DATASET.md` after the 2026-07-11 incident (below).

**Features (15, plus5):** log_return, abs_return, squared_return, intraday_range, range_ratio, volatility_5, volatility_21, log_volume, volume_surge, spy_volatility_21, return_ma_21, rsi_14, volatility_ratio_5_21, lag_abs_return_1, spy_log_return. Known redundancy cluster: `abs_return` / `squared_return` / `lag_abs_return_1`.

## A5. Model performance — verified bake-off results

### A5.1 J01 vol_risk (next-day |log return|, MAE, lower better)

**Stocks dev200** (126 symbols; champion ★):

| Model | Val MAE | Crisis MAE | Beats baseline (val / crisis) |
|---|---|---|---|
| xgb | **0.014261** | 0.028041 | ✅ / ❌ |
| lstm | 0.014395 | 0.027217 | ✅ / ❌ |
| gru | 0.014461 | 0.027735 | ✅ / ❌ |
| **garchgru ★** | 0.014570 | **0.026794** | ✅ / ✅ — only model beating baseline on both |
| cnn1d | 0.014785 | 0.026955 | ✅ / ≈tie |
| tcn | 0.015230 | 0.030331 | ✅ / ❌ |
| baseline (ridge) | 0.015471 | 0.026955 | — |
| garchgru_qlike | 0.018919 | 0.029056 | side-experiment: crisis under-rate 53%→36% at ~30% MAE cost |
| gru_qlike | 0.050582 | 0.055271 | plain GRU destabilizes under QLIKE |
| persistence (untrained) | 0.019687 | 0.035009 | reference floor — all trained models clear it |

**ETFs dev200** (127 symbols; champion ★ = xgb):

| Model | Val MAE | Crisis MAE |
|---|---|---|
| **xgb ★** | **0.004254** | 0.013500 |
| lstm | 0.004493 | **0.012976** |
| gru | 0.004686 | 0.013030 |
| garchgru | 0.004783 | 0.013101 |
| cnn1d | 0.005091 | 0.013618 |
| baseline | 0.005475 | 0.014376 |
| tcn | 0.005528 | 0.016818 |
| garchgru_qlike | 0.005737 | 0.014424 (under-rate 61%→41%) |

ETF volatility is ~3× easier than single stocks (lower base error, every serious model beats baseline).

**Stocks FULL universe** (1,838 symbols — the decisive experiment):

| Model | Val MAE | Crisis MAE | Val AUC@90 | Crisis AUC@90 | Crisis under-rate | Conformal COVID coverage (90% target) |
|---|---|---|---|---|---|---|
| Ridge baseline | 0.014270 | **0.025908** | 0.775 | **0.841** | **44.5%** | 0.736 |
| GARCH-GRU Huber | **0.013564** | 0.026027 | 0.807 | 0.793 | 48.6% | 0.736 |
| GARCH-GRU QLIKE | 0.013635 | 0.026026 | **0.808** | 0.795 | 47.5% | 0.731 |
| Persistence | 0.017800 | 0.033200 | — | — | — | — |

Tail MAE on crisis days (full run): GARCH-GRU p95 0.1617 / p99 0.297 vs baseline p95 0.1459 / p99 0.278 — **baseline wins at every reported tail**.

**Verdict (recorded):** `promotion_eligible: false`. Scaling raised the baseline's crisis bar more than the neural model gained. Dev200 GARCH-GRU stays champion on paper, but the product cannot ship on it.

### A5.2 J02 vol_regime binary (21-day high-vol regime, macro-F1, higher better)

| Lane | baseline | gru | tcn | lstm | cnn1d | xgb | Champion |
|---|---|---|---|---|---|---|---|
| **stocks** val | 0.7587 | 0.8008 | 0.7974 | **0.8022** | 0.8010 | 0.7971 | **lstm ★** |
| stocks crisis | 0.6396 | 0.6293 | 0.6271 | **0.6596** | 0.6513 | 0.6399 | (beats baseline both) |
| **etfs** val | 0.7324 | **0.7367** | 0.7262 | 0.7335 | 0.7144 | 0.7243 | **gru ★** (only one beating baseline on both; margins thin) |
| etfs crisis | 0.5864 | **0.5878** | 0.5204 | 0.5110 | 0.4667 | 0.4713 | |
| **market (SPY)** val | **0.6597** | 0.6425 | 0.5415 | 0.6350 | 0.5645 | 0.5747 | **baseline ★** (no NN beats it on val) |
| market crisis | 0.3091 | 0.3091 | 0.1547 | **0.3538** | 0.2615 | 0.3091 | lstm noted as crisis-specialist candidate |

Market-lane pathology: n=503 val / n=42 crisis rows for a single symbol → NNs overfit, and *all* models severely under-predict the high-vol class during COVID (true rate 0.857, predicted 0.12–0.21).

### A5.3 J02v2 vol_regime3 three-class (bull/bear/crisis @ train-q90 forward vol, macro-F1)

| Lane | baseline | Champion | Champion val | Champion crisis | Beats baseline (val/crisis) |
|---|---|---|---|---|---|
| stocks | 0.4292 / 0.3586 | **lstm ★** | 0.4605 | 0.3897 | ✅ / ✅ |
| etfs | 0.4236 / 0.3895 | **gru ★** | 0.4939 | 0.3957 | ✅ / ✅ (only candidate on both) |
| market | 0.3607 / 0.2638 | **lstm ★** | 0.4971 | 0.2122 | ✅ / ❌ — neural models under-predict the crisis class |

Three-class is much harder than binary (macro-F1 ~0.46–0.49 vs ~0.74–0.80) but gives the product a direct "crisis" class. Class weighting was required — unweighted models collapse to bull-heavy predictions.

### A5.4 Product risk layers (quantified)

- **SPCI conformal bands** (block bootstrap, 90% target, calibrated 2018–19): COVID empirical coverage ≈ **73–74% for every model** (dev200 and full). The ~0.16 coverage gap is model-agnostic → repurposed as a **regime-shift detector**. Product rule recorded: gap > 0.15 → reduce-risk mode; 0.05–0.15 → widen bands.
- **MMD drift monitor** (RBF kernel, permutation-calibrated threshold 0.00228): val 2018–19 MMD² = 0.0071 (mild); COVID Q1 = 0.0414 (**~18× threshold**). Works as an independent alarm.
- **Orchestration** (`00a_ensemble_product_scores.ipynb`): scaffold only — the single biggest unbuilt product piece.

## A6. Infrastructure inventory (working)

| Component | Location | State |
|---|---|---|
| Streaming preprocessing (NPZ + memmap, atomic publish) | `scripts/run_preprocessing.py` | ✅ tested (`tests/test_disk_backed_pipeline.py`, 4 pass) |
| Lazy bundle loading | `scripts/tf_data.py` | ✅ |
| Model builders, GarchGruCell, QLIKE loss, asymmetric metrics, chunked TF datasets, batch predict | `scripts/tf_notebook_utils.py` | ✅ |
| Job registry / naming | `scripts/ml_iteration2_jobs.py` | ✅ (garchgru = letter f, vol_risk only) |
| Scaffold generator (`--architecture` scoped regen) | `scripts/scaffold_ml_iteration2_notebooks.py` | ✅ |
| Bake-off comparison CLI | `scripts/compare_job_models.py` | ⚠️ dev200 only (full-tag glob mismatch) |
| Replay outputs into notebooks | `scripts/replay_notebook_outputs.py` | ⚠️ same full-tag gap; regex misses garchgru/qlike/vol_regime3 stems |
| Conformal engine + notebook | `scripts/conformal.py`, `00b_conformal_risk_bands.ipynb` | ✅ dev200 + full |
| Drift monitor | `scripts/drift_monitor.py` | ✅ |
| QLIKE parallel trainer (with CUDA re-exec fix) | `scripts/train_qlike_variants.py` | ✅ |
| Graceful-stop for long notebook runs | `scripts/graceful_stop_notebook_training.py` | ✅ |
| Kernel management | `install-jupyter-kernel.sh`, `fix_notebook_kernels.py` etc. | ✅ |
| Presentation dashboard (Streamlit) | `dashboard/` | ⚠️ **stale**: discovery regex only knows baseline/gru/tcn/lstm/cnn1d/xgb + dev200 tag → blind to garchgru, qlike, vol_regime3, and full-universe artifacts |

## A7. Challenges & bottlenecks (root-cause view)

**Modeling (the real blockers):**

| # | Problem | Evidence |
|---|---|---|
| P1 | Tail level error — MAE blows up on the wildest days | crisis p95/p99 tail MAE worse than baseline |
| P2 | Tail detection — extreme-day ranking degrades at scale | crisis AUC@90 0.79 vs baseline 0.84 (dev200 GARCH-GRU had 0.836) |
| P3 | Crisis under-prediction bias | under-rate 48.6% Huber vs 44.5% baseline; QLIKE helps (−1.1 pp) but doesn't close it |
| P4 | High-volume × high-vol joint misses | volume is a plain feature; no interaction/cross-sectional signal |
| P5 | Promotion gate failure | wins val, loses crisis → recorded `promotion_eligible: false` |
| P6 | Regime-shift uncertainty | conformal coverage 74% vs 90% target in COVID; MMD² 18× threshold |
| P7 | Good average calibration, bad tail calibration | decile calibration clean mid-range, degrades in top deciles |

Root cause running through P1–P3+P7: a **single scalar next-day `abs_return` target trained on symmetric mean-error loss** optimizes average accuracy, not tail ranking or tail level. The plan's granularity strategy (temporal → 21d forward vol; categorical → regime classes; distributional → quantile/log-variance heads) is the structural answer, and it is the right one.

**Data-scale asymmetry:** the market (SPY) lane is unlearnable for NNs at n=503/42 — baseline champions by default. Anything single-symbol must either pool context or be served by the baseline/classifier stack.

**Engineering incidents & gotchas (all now mitigated, must survive the rebuild):**

1. **OOM kill at ~45.8 GiB** concatenating 1,838 symbols in RAM → mandatory `memmap_v1` disk-backed policy; never `np.concatenate` full universes, never `.astype()` a full memmap, never `from_tensor_slices` on complete arrays.
2. **Stale-bundle near-miss:** a "full" run silently trained on a 2-symbol leftover bundle (AAL/AAME). Fix codified: verify `manifest.json` symbol count + split shapes before any full training. The invalid artifact (`gru_vol_risk_stocks_tf_predictions.json`, n_symbols=2) is **still in `data/models/`** and should be archived.
3. **CPU-fallback trap:** plain scripts miss `LD_LIBRARY_PATH` → TF silently trains on CPU. Fixed via re-exec pattern in `train_qlike_variants.py`; Jupyter kernel unaffected.
4. **GarchGruCell is not cuDNN-fused** → ~3.5 min/epoch on stocks dev200 (vs seconds for plain GRU); use PATIENCE=8, expect ~12.5k steps/epoch on full.
5. RTX 5090 CUDA JIT: first GPU epoch can take 10–30+ min.
6. QLIKE destabilizes plain GRU (val MAE 0.05 stocks; diverged on ETFs) — only use it with the GARCH cell.

**Repo hygiene findings (this audit):**

| Finding | Detail |
|---|---|
| Uncommitted state | Modified: `J01_vol_risk/champions.json`, `SESSION_HANDOFF.md`, `00b_conformal_risk_bands.ipynb`. Untracked: `J02v2_vol_regime3/champions.json`, 3 full-universe `*_conformal.json` |
| Stray/legacy files | `ML_Iteration_2/J00a_baseline_vol_regime_etfs/` (orphan folder), `J02_vol_regime/market/Untitled.ipynb`, legacy `09a_gru_15_fts_all_stock.ipynb`, `untitled.chat` at root |
| Stale artifacts | 2-symbol `gru_vol_risk_stocks_tf_*`; duplicate legacy `vol_gru_stocks_universe_plus5_dev200_*`; `xgb_vol_regime_market_tf_predictions.json` tagged `full` but byte-identical metrics to the dev200 SPY run (mislabeled tag) |
| Doc drift | `ML_Iteration_2/README.md` says J01 champion "garchgru (stocks), xgb (etfs)" ✅ but the top-level `README.md` still describes J02 as "active/champion pending" and doesn't mention garchgru/QLIKE/conformal/full universe |
| Tooling gaps | `compare_job_models.py` + `replay_notebook_outputs.py` can't address full-tag artifacts; dashboard blind to everything post-2026-07-02 |

---

# PART B — REBUILD GUIDE

## B0. Rebuild philosophy

Three principles, in priority order:

1. **Don't lose what's proven.** The bake-off discipline (trainable baseline gate → candidates → champion JSON → crisis stress test) is the project's best asset. The rebuild keeps it and extends it — it does not restart it.
2. **Dev200 first, always.** Every new target, loss, head, or feature must show lift on dev200 (minutes–hours) before any full memmap rebuild (~25 GB, hours). This rule is already written down; the rebuild makes it a hard gate.
3. **The tail is the product.** A volatility product that is right on calm days and wrong in crises is worse than useless. Every phase below is judged against the tail criteria (crisis MAE, AUC@90, p95 tail MAE, under-rate), not average MAE.

**Definition of done** (unchanged from the plan, restated as the single finish line): a full-universe J01 stocks champion that (a) beats the full baseline on val MAE **and** crisis MAE, (b) matches/beats crisis AUC@90 ≥ 0.841, (c) matches/beats crisis p95 tail MAE ≤ 0.1459, (d) under-rate < 49.9% without unacceptable val cost, (e) has a documented conformal coverage gap — plus a working orchestration layer producing a product `risk_score`.

## B1. Phase R0 — Freeze, commit, and clean (half a day)

Goal: a tagged, reproducible starting point. Nothing scientific happens until the repo is clean.

1. **Commit the in-flight state** (it contains champion records — losing it would be expensive):
   ```bash
   git add ML_Iteration_2/J01_vol_risk/champions.json ML_Iteration_2/SESSION_HANDOFF.md \
           ML_Iteration_2/00_orchestration/00b_conformal_risk_bands.ipynb \
           ML_Iteration_2/J02v2_vol_regime3/champions.json \
           data/models/baseline_vol_risk_stocks_tf_conformal.json \
           data/models/garchgru_vol_risk_stocks_tf_conformal.json \
           data/models/garchgru_qlike_vol_risk_stocks_tf_conformal.json
   git commit -m "chore: commit full-universe conformal results and champion records"
   git tag pre-rebuild-2026-07
   ```
2. **Archive invalid/stale artifacts** (move, don't delete) into `data/models/_archive/`:
   - `gru_vol_risk_stocks_tf_*` (2-symbol invalid full run)
   - `vol_gru_stocks_universe_plus5_dev200_*` (pre-migration duplicate of gru dev200)
   - `xgb_vol_regime_market_tf_*` (mislabeled `full` duplicate of the dev200 SPY run)
   - `.backup_garchgru_*` files
3. **Remove strays:** `ML_Iteration_2/J00a_baseline_vol_regime_etfs/`, `J02_vol_regime/market/Untitled.ipynb`, root `untitled.chat`; move `09a_gru_15_fts_all_stock.ipynb` to `_archive/` (it's already migrated to 01a).
4. **Sync docs:** update top-level `README.md` job table (J02/J02v2 complete with champions; garchgru architecture; full-universe status + promotion block). One commit.
5. **Decide repo vs fresh clone.** Recommendation: **rebuild in-place on a branch** (`rebuild/iter3`), not a fresh repo — the artifacts and champion history are the project's lab notebook. A fresh clone onto a new machine follows `README.md` quick start + `docs/DATASET.md` and inherits everything.

## B2. Phase R1 — Environment verification (half a day, or on any new machine)

```bash
bash scripts/setup-wsl-venv.sh
source .venv/bin/activate
bash scripts/install-jupyter-kernel.sh
python scripts/fix_notebook_kernels.py --check
pytest tests/test_disk_backed_pipeline.py   # 4 tests must pass
python - <<'EOF'
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
EOF
```

Acceptance: GPU visible from the *kernel*; the 4 pipeline tests pass; kernel check clean. Record TF/CUDA versions in `SESSION_HANDOFF.md`. Remember the two traps: never set `PIP_USER=true` with the venv active; plain scripts need the `LD_LIBRARY_PATH` re-exec pattern from `train_qlike_variants.py` or they fall back to CPU silently.

## B3. Phase R2 — Data layer rebuild (1 day, mostly compute)

Only needed if bundles are missing/suspect or when a **new target** is introduced (Phase R4). Verification is the point; the commands are established:

```bash
# dev200 (fast iteration; NPZ)
python scripts/run_preprocessing.py --asset-type stock --profile plus5 \
  --symbol-scope all --max-symbols 200 --target-type <TARGET>

# full universe (disk-backed; NEVER in-memory)
python scripts/run_preprocessing.py --asset-type stock --profile plus5 \
  --symbol-scope all --target-type <TARGET> --storage-format memmap \
  --history-outlier-filter none
```

**Mandatory post-build checklist (the 2-symbol incident must never recur):**
- [ ] `manifest.json` `n_symbols` matches expectation (1,838 stocks full; ~126/127 dev200)
- [ ] Split shapes match `pipeline_config*.json`
- [ ] Spot-check one symbol's last training date ≤ 2017-12-31
- [ ] Scaler moments finite; no NaN rows
- [ ] For memmap: bundle was atomically published (no `_staging` remnant)

Keep `--history-outlier-filter` at `none`/`iqr_upper` for symbol hygiene only — **do not drop high-vol days**; they are the learning signal for the tail.

## B4. Phase R3 — Model-side tail experiments on dev200 (1–2 weeks, the scientific core)

Split the problem the way the plan prescribes: **detection** (will tomorrow be extreme?) vs **level** (how large?). Run all experiments on the stocks dev200 bundle; success = beat dev200 baseline on **crisis MAE + crisis AUC@90 + p95 tail MAE** simultaneously (the triple that full-universe GARCH-GRU failed).

Ordered by expected information-per-hour:

| # | Experiment | How | Status |
|---|---|---|---|
| R3.1 | **Classifier × regressor ensemble** | Gate/blend J01 GARCH-GRU magnitude with J02 LSTM P(high-vol): e.g. `risk = pred_mae × (1 + k·p_highvol)` or quantile switch. Components already trained — this is a pure analysis notebook on saved predictions. | Components ✅, fusion ✗ |
| R3.2 | **Sample weights by vol decile** | `make_tf_datasets(..., train_sample_weight)` already supported; upweight top 10–20% target deciles; retrain 01f. Cheapest retrain-side lever. | Supported, untested |
| R3.3 | **Multi-task GARCH-GRU** | Huber head + BCE top-decile head on shared trunk; extend 01f training cell. Directly optimizes detection + level jointly. | Planned (1c) |
| R3.4 | **Quantile/pinball heads** (τ = 0.5, 0.9, 0.95) | New loss in `tf_notebook_utils.py`; product consumes τ=0.9 as conservative level. | Planned (1b) |
| R3.5 | QLIKE refinements | Keep as the bias-reducer it proved to be (under-rate 53→36% dev200); combine with R3.2/R3.3 rather than alone. | ✅ characterized |

Log every run in `SESSION_HANDOFF.md` with the four tail metrics + val MAE. Kill experiments that don't move crisis AUC@90 — that is the metric with the biggest full-universe deficit.

## B5. Phase R4 — Target & feature rebuild (1 week, only for winners)

For whichever R3 direction wins, rebuild the *data* to support it structurally (dev200 first, per B3 commands with new `--target-type`):

| Change | Rationale | Job slot |
|---|---|---|
| Target → **21d forward realized vol** (`forward_vol`) | Smoother than 1-day |r|; less label noise; already computed for J02 | J01c `vol_risk_fwd21` |
| Target → **log-variance** `log(r²+ε)` | Scale-stabilized; aligns with GARCH cell + QLIKE | J01 variant |
| Parallel classifier lane on `vol_regime3_21` | Crisis class = direct tail objective | J01b / reuse J02v2 |
| Drop one of `abs_return`/`squared_return`/`lag_abs_return_1` | Redundancy cluster confirmed | feature profile v2 |
| **Cross-sectional features**: symbol vol rank vs universe, volume-surge percentile | Attacks P4 (heavy-volume turbulent days); gives single-symbol lanes universe context | feature profile v2 |
| Volume×vol interaction (`volume_surge × volatility_21`) | Same | feature profile v2 |

Scaffold new job slots with the existing generator (do not hand-copy notebooks):
```bash
python scripts/scaffold_ml_iteration2_notebooks.py --job-slug vol_risk_fwd21
```
which requires registering the job + target in `ml_iteration2_jobs.py` and `run_preprocessing.py` first — follow the `vol_regime3_21` addition as the worked example (it touched exactly: preprocessing target fn, job registry, scaffold, multiclass support).

## B6. Phase R5 — Full-universe promotion (per winner; days of compute)

Only after a candidate clears the dev200 triple gate:

1. Rebuild the full memmap bundle for the new target (B3 checklist).
2. Train full baseline first, then the candidate (`RUN_MODE="full"`, PATIENCE=8 for GARCH variants, graceful-stop script armed).
3. Score against the **Phase 4 promotion gate** (all boxes, no exceptions):
   - beats full baseline val MAE ✅ and crisis MAE ✅
   - crisis AUC@90 ≥ 0.841 (or documented trade-off)
   - crisis p95 tail MAE ≤ 0.1459
   - under-rate < 49.9%
   - conformal run on full predictions; coverage gap recorded
4. Update `champions.json` (keep the dev200/full nesting pattern J01 already uses), `SESSION_HANDOFF.md`, and the decision log.

## B7. Phase R6 — Product orchestration (parallel track, 1 week)

Build `00a_ensemble_product_scores.ipynb` from scaffold to working. This does **not** wait for R5 — it can ship on current champions because the risk layers are honest about their limits:

```
Stage A  regime gate      = J02 champion P(high-vol) (+ J02v2 crisis class)
Stage B  magnitude        = J01 champion point forecast
Stage C  uncertainty      = SPCI upper conformal band on B
Stage D  drift gate       = MMD health check (18× threshold in COVID = proven alarm)

risk_score = f(upper_band, P(high-vol), coverage_gap, drift_flag)
rules: coverage gap >0.15 OR drift alarm → reduce-risk mode (documented in 00b)
```

Deliverable: one JSON per universe per day-equivalent with `risk_score`, `regime`, `band`, `health` — the contract the future app consumes. Add a replay-based backtest over val + COVID to show the reduce-risk rule actually fires in Feb–Mar 2020.

## B8. Phase R7 — Tooling & dashboard catch-up (2–3 days)

1. **Fix full-tag handling** in `compare_job_models.py` + `replay_notebook_outputs.py` (empty dataset-tag in filenames; the known glob/regex gap).
2. **Extend artifact regexes** (both scripts + `dashboard/data_loader.py`) to cover: `garchgru`, `garchgru_qlike`, `gru_qlike`, `vol_regime3`, full-tag stems, and future `vol_risk_fwd21`. Single source of truth: export one artifact-name regex from `ml_iteration2_jobs.py` and import it everywhere instead of three private copies.
3. **Dashboard update:** add J02v2 lanes, GARCH-GRU/QLIKE rows, the full-universe bake-off panel (val-vs-crisis + tail table + promotion-gate scorecard), and conformal coverage tiles. The promotion-gate scorecard doubles as the rebuild's progress tracker.
4. Wire `tests/` into a habit: `pytest` + `fix_notebook_kernels.py --check` before any full run.

## B9. Working practices for the rebuild (carry-forward checklist)

- One experiment = one notebook (or one trainer-script run) = one `*_predictions.json` artifact = one handoff-log line. No untracked results.
- Baselines first, always — a lane without a trained baseline produces no interpretable candidate results.
- Full-universe runs: verify manifest → train baseline → train candidate → gate → record. Never trust a `full` filename without a manifest check.
- Champions are append-only decisions with notes; never overwrite the reasoning in `champions.json` `notes` fields.
- Dates in docs are absolute; every training artifact carries `trained_at` (already enforced by payload writers).
- After venv or scaffold changes: `cleanup-jupyter-kernels.sh` → `install-jupyter-kernel.sh` → `fix_notebook_kernels.py`.

## B10. Suggested sequencing (realistic solo cadence)

| Week | Work | Exit criterion |
|---|---|---|
| 1 | R0 hygiene + R1 env + R3.1 ensemble analysis | tagged repo; ensemble result vs triple gate known |
| 2 | R3.2 sample weights + R3.3 multi-task on dev200 | best tail recipe identified, logged |
| 3 | R3.4 quantile heads; pick winner; start R4 target rebuild (dev200) | dev200 triple gate cleared by ≥1 candidate |
| 4 | R6 orchestration MVP in parallel with R4 validation | `risk_score` JSON + COVID backtest exists |
| 5 | R5 full rebuild + promotion attempt | gate scorecard filled, promoted or documented why not |
| 6 | R7 tooling/dashboard; write-up | dashboard reflects reality; report updated |

If a week slips, slip everything — do not skip dev200 validation to "save time" on the full universe; that trade already failed once (2-symbol incident + blocked promotion).

---

## Appendix: authoritative file map

| What | Where |
|---|---|
| Champion records | `ML_Iteration_2/J01_vol_risk/champions.json`, `J02_vol_regime/champions.json`, `J02v2_vol_regime3/champions.json` |
| Running lab log | `ML_Iteration_2/SESSION_HANDOFF.md` |
| Strategy (problems P1–P7, phases) | `ML_Iteration_2/REBUILD_IMPLEMENTATION_PLAN.md` |
| Data contract + memmap policy | `docs/DATASET.md`, `.cursor/rules/full-universe-disk-backed.mdc` |
| All metrics ground truth | `data/models/*_predictions.json` (+ `*_conformal.json`) |
| Iteration-1 post-mortem | `04_ML_Model_Iteration-1/MODEL_EVALUATION_v1_vs_v2.md` |
| Dashboard | `dashboard/` (needs R7 catch-up) |
