# Stock Volatility Prediction

Machine-learning bake-offs for **next-day volatility risk** and **21-day volatility regimes** across a pooled stock & ETF universe — validated on 2018–19 and stress-tested on the COVID-19 crash of 2020 Q1.

Instead of predicting price direction (mostly noise), this project predicts *risk*: how turbulent tomorrow is likely to be, and which volatility regime a symbol is in. Every candidate model is compared against a trainable baseline and naive persistence, and a champion is only selected if it beats the baseline on **both** validation and the COVID stress window.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red) ![License](https://img.shields.io/badge/License-MIT-green)

## Results (dev200 universe, recorded in `ML_Iteration_2/J*/champions.json`)

| Job | Target | Universe | Champion | Val | Baseline val | COVID crisis |
|---|---|---|---|---|---|---|
| J01 · vol_risk | Next-day \|log return\| (regression, MAE ↓) | Stocks | GARCH-GRU hybrid | **0.01457** | 0.01547 | 0.02679 |
| J01 · vol_risk | Next-day \|log return\| (regression, MAE ↓) | ETFs | XGBoost | **0.00425** | 0.00548 | 0.01350 |
| J02 · vol_regime | 21-day high-vol regime (binary, macro-F1 ↑) | Stocks | LSTM | **0.8022** | 0.7587 | 0.6596 |
| J02 · vol_regime | 21-day high-vol regime (binary, macro-F1 ↑) | ETFs | GRU | **0.7367** | — | 0.5878 |
| J02v2 · vol_regime3 | 21-day regime, 3-class bull/bear/crisis (macro-F1 ↑) | Stocks | LSTM | **0.4605** | 0.4292 | 0.3897 |
| J02v2 · vol_regime3 | 21-day regime, 3-class bull/bear/crisis (macro-F1 ↑) | ETFs | GRU | **0.4939** | 0.4236 | 0.3957 |

Full per-model results, training timestamps, and selection notes live in the three `champions.json` files — they are the single source of truth for these numbers.

## What this demonstrates

- **Evaluation discipline:** baselines and persistence are first-class citizens; champions must win on validation *and* an out-of-sample crisis window. On the J02 ETF lane the baseline was highly competitive and GRU's margin was thin (+0.0042 val) — recorded as such, not rounded into a victory.
- **Honest negative results:** a full-universe GARCH-GRU trained with a QLIKE loss beat the full baseline on validation MAE but *lost* on crisis MAE (0.02603 vs 0.02591) — so promotion was withheld. That decision is documented in `J01_vol_risk/champions.json`.
- **Custom modelling:** a GARCH(1,1)-GRU hybrid cell (`GarchGruCell`) that embeds a conditional-variance recursion inside the recurrent cell, plus QLIKE loss variants for volatility regression.
- **Uncertainty & monitoring:** SPCI conformal risk bands (`scripts/conformal.py`) and RBF-MMD drift monitoring (`scripts/drift_monitor.py`) around the champions.
- **Data engineering:** disk-backed, streaming preprocessing (Welford running moments, sequence bundles written to `.npz`) so the full universe never needs to fit in RAM (`scripts/run_preprocessing.py`, `tests/test_disk_backed_pipeline.py`).
- **Product framing:** separate models per job (risk engine, regime gate, opportunity ranker) rather than one mega-network — designed for a novice-investor product that favours risk control over stock-picking.

## Approach

```text
Shared preprocessing (scripts/run_preprocessing.py)
    │
    ├── J01 vol_risk        → next-day |log return|     [complete]
    ├── J02 vol_regime      → 21d vol regime (binary)   [complete]
    ├── J02v2 vol_regime3   → 21d regime (3-class)      [complete]
    ├── J03 rel_strength    → 21d excess vs SPY         [scaffold]
    ├── J04 etf_opportunity → ETF excess vs SPY         [scaffold]
    └── 00 orchestration    → conformal bands, drift, champion merge
```

Each job runs one trainable baseline plus candidate architectures (GRU, GARCH-GRU, TCN, LSTM, CNN-1D, XGBoost) per universe (stocks / ETFs / market), exports prediction artifacts, and records its champion in `champions.json`.

## Quick start

Training was developed on WSL2/Linux with a TensorFlow GPU environment (Python 3.12).

```bash
git clone https://github.com/FOKanu/stock-volatility-prediction.git
cd stock-volatility-prediction

bash scripts/setup-wsl-venv.sh
source .venv/bin/activate
bash scripts/install-jupyter-kernel.sh
```

**Generated artifacts are not committed.** Prediction JSONs, processed sequences (`.npz`), and checkpoints (`.keras` / `.joblib`) are git-ignored build outputs (~650 MB locally). Regenerate them with:

```bash
python scripts/run_preprocessing.py          # builds data/processed/*.npz
# then run the training notebooks under ML_Iteration_2/J*/ in order
# (00a baseline first, then the candidate architectures)
```

See [docs/DATASET.md](docs/DATASET.md) for dataset details and [docs/PROJECT_REPORT_AND_REBUILD_GUIDE.md](docs/PROJECT_REPORT_AND_REBUILD_GUIDE.md) for the full rebuild guide.

## Dashboard

A Streamlit dashboard (`dashboard/app.py`) turns the saved artifacts into a leaderboard, prediction explorer, and COVID stress-test view:

```bash
streamlit run dashboard/app.py
```

It reads only saved outputs (`data/models/*_tf_predictions.json`, `ML_Iteration_2/J*/champions.json`) and never retrains. If the prediction artifacts have not been generated yet, it says so and stops — the champion metrics above come from the committed `champions.json` files regardless.

## Limitations

- Research project — **not investment advice** and not connected to any live trading.
- Champions were selected on the dev200 universe; full-universe promotion is deliberately stricter and, for J01, currently withheld (see above).
- J03 and J04 are scaffolds; J01, J02, and J02v2 are the completed bake-offs.
- The dashboard's market-universe date axis is approximate (exchange holidays are not modelled).
- The test suite (`tests/`) requires the TensorFlow environment described above.

## Tech stack

Python · TensorFlow/Keras · XGBoost · pandas / NumPy · Streamlit · Jupyter

## License

MIT — see [LICENSE](LICENSE).
