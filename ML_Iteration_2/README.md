# ML Iteration 2 — Model Portfolio

Product-oriented model bake-offs: **one trainable baseline + 5 architectures per job**, pick a champion per universe on validation + COVID crisis test.

## Folder layout

```text
ML_Iteration_2/
├── README.md
├── 00_orchestration/
│   └── 00a_ensemble_product_scores.ipynb
├── J01_vol_risk/
│   ├── stocks/          # 00a baseline + 01a–01e  (ACTIVE)
│   ├── etfs/            # 00a baseline + 01a–01e  (ACTIVE)
│   └── champions.json
```

## Naming convention

```text
00a_baseline_{job_slug}_{universe}.ipynb    ← train first
{job_num}{letter}_{architecture}_{job_slug}_{universe}.ipynb
```

| ID | Architecture | Role |
|----|--------------|------|
| **00a** | `baseline` | **Ridge on last-day features** (trainable reference) |
| a | `gru` | Candidate model |
| b | `tcn` | Candidate model |
| c | `lstm` | Candidate model |
| d | `cnn1d` | Candidate model |
| e | `xgb` | Candidate model |

## Kernel (important)

All notebooks use **`Python (stock-prediction)`** (`stock-prediction` kernelspec).

One kernel only — do not use `python3`, `.venv`, or `stock-prediction-wsl` aliases.

Repair after venv changes or scaffold updates (scoped — does not wipe other jobs):

```bash
cd ~/ai-workspace/experiments/stock-prediction
# Regenerate J02 only (leaves J01 notebook outputs untouched):
python scripts/scaffold_ml_iteration2_notebooks.py --job-slug vol_regime

bash scripts/cleanup-jupyter-kernels.sh
bash scripts/install-jupyter-kernel.sh
python scripts/fix_notebook_kernels.py
```

Restore train/eval plots from saved `data/models/*_predictions.json` **without retraining**:

```bash
python scripts/replay_notebook_outputs.py --job-slug vol_risk --dataset-tag dev200
# or one notebook:
python scripts/replay_notebook_outputs.py --notebook ML_Iteration_2/J01_vol_risk/stocks/01e_xgb_vol_risk_stocks.ipynb
```

Then reopen notebooks in Jupyter — train and evaluate cells show replayed metrics/plots.

## Baseline workflow

1. Run **`00a_baseline_*`** in each job/universe folder (fast, CPU-friendly).
2. Run **`01a`–`01e`** — each notebook prints and saves `baseline_comparison` (beats/loses vs baseline).
3. Compare all artifacts:

```bash
python scripts/compare_job_models.py --job-slug vol_risk --universe stocks --dataset-tag dev200
```

**J01 vol/risk** also reports untrained **persistence** (today \|r\| → tomorrow) inside the baseline notebook for extra context.

## Jobs

| Job | Slug | Target | Universe | Phase | Status |
|-----|------|--------|----------|-------|--------|
| J01 | `vol_risk` | Next-day `\|log return\|` (`abs_return`) | stocks, etfs | P1 | **complete** |
| J02 | `vol_regime` | 21d high-vol regime (`vol_regime_21`) | stocks, etfs, market | P3 | **active** |
| J03 | `rel_strength` | 21d excess return vs SPY | stocks | P4 | scaffold |
| J04 | `etf_opportunity` | 21d excess return vs SPY | etfs | P5 | scaffold |

## Artifact naming

Saved under `data/models/`:

```text
{architecture}_{job_slug}_{universe}_{dataset_tag}_tf_predictions.json
{architecture}_{job_slug}_{universe}_{dataset_tag}_tf_best.keras   # or .joblib for xgb
```

Example: `gru_vol_risk_stocks_dev200_tf_predictions.json`

## Champion selection (J01)

| Priority | Rule |
|----------|------|
| 1 | Must beat trained baseline on val MAE (or note if none do) |
| 2 | Lowest crisis MAE |
| 3 | Highest top-10% vol AUC |

## Run order

1. ~~**J01 stocks + etfs**~~ — champions: **xgb** (`champions.json`)
2. **J02 market** — run `00a` baseline, then `02a`–`02e` (SPY regime lane first)
3. **J02 stocks / etfs** — same bake-off pattern with `vol_regime_21` artifacts
4. `python scripts/compare_job_models.py --job-slug vol_regime --universe market --dataset-tag dev200`
5. Pick champions → optional `RUN_MODE = "full"` re-run

## Product mapping

| App feature | Job | Champion output |
|-------------|-----|-----------------|
| Turbulence alert | J01 | `risk_score` |
| Market weather | J02 market | `regime` |
| Stock watchlist | J03 | `rel_rank` |
| ETF allocation hint | J04 | `etf_rank` |
| Position limits | Coach rules | `00_orchestration` |

## Migration note

Legacy `09a_gru_15_fts_all_stock.ipynb` was migrated to `J01_vol_risk/stocks/01a_gru_vol_risk_stocks.ipynb`.
