# Stock Volatility Prediction

Machine-learning pipeline for **next-day volatility**, **risk regimes**, and **relative opportunity** across a pooled stock and ETF universe. Built for WSL2/Linux with TensorFlow GPU support (tested on RTX 5090).

Originally a BIT4.2C analytics project (RNN direction prediction); the active workstream is **ML Iteration 2**, oriented toward a novice-investor product that emphasizes risk control and benchmark-relative selection—not daily stock picking.

---

## Quick start (new machine)

```bash
git clone https://github.com/FOKanu/Stock-Volitility-Prediction.git
cd Stock-Volitility-Prediction

# Python 3.12 + TensorFlow GPU (WSL/Linux)
bash scripts/setup-wsl-venv.sh
source .venv/bin/activate
bash scripts/install-jupyter-kernel.sh

# Open a notebook — kernel: Python (stock-prediction)
jupyter lab ML_Iteration_2/J01_vol_risk/stocks/00a_baseline_vol_risk_stocks.ipynb
```

**No raw dataset required** for dev200 training: processed sequences and model checkpoints are included in the repo. See [docs/DATASET.md](docs/DATASET.md) if you need to rebuild preprocessing or run the full universe.

---

## Goals

### Product (long-term)

Help novice investors **avoid blowing up** and **make calmer allocation decisions**, not day-trade on noisy signals.

| Layer | Purpose | Example output |
|-------|---------|----------------|
| Risk engine | Flag turbulent periods | "High turbulence expected tomorrow" |
| Regime gate | Market/stock weather | Reduce risk mode during crisis-like regimes |
| Opportunity ranker | Names vs SPY over ~21 days | Watchlist, not buy/sell arrows |
| ETF lane | Core + satellite allocation hints | Favor ETFs vs benchmark |
| Coach (rules) | Hard limits | Max position %, diversification |

"Beat the market" is framed as **risk-adjusted, benchmark-relative performance** over weeks—not binary next-day direction calls.

### ML (current)

- Predict **learnable** targets (vol/risk first; direction last).
- Run **bake-offs**: one trainable baseline + five architectures per job; pick a champion on validation + COVID stress test.
- Compare every candidate against the **baseline** and naive **persistence** (today's \|return\| → tomorrow).
- Export reproducible artifacts under `data/models/` for orchestration and backtests.

---

## Current approach (ML Iteration 2)

### Architecture strategy

**Separate models per job**, not one mega-network:

```text
Shared preprocessing (run_preprocessing.py)
        │
        ├── J01 vol_risk      → next-day |log return|  [complete]
        ├── J02 vol_regime    → 21d vol regime class   [active]
        ├── J03 rel_strength  → 21d excess vs SPY        [scaffold]
        ├── J04 etf_opportunity → ETF excess vs SPY    [scaffold]
        └── 00 orchestration  → merge champion scores
```

Each job folder contains:

- `00a_baseline_*` — Ridge on last-day features (train **first**)
- `01a`–`01e` — GRU, TCN, LSTM, 1D CNN, XGBoost/HistGBR

See [ML_Iteration_2/README.md](ML_Iteration_2/README.md) for naming, metrics, and run order.

### Active job: J02 vol / regime

| Item | Value |
|------|-------|
| Target | 21-day high-volatility regime (`vol_regime_21`) |
| Features | 15 (plus5 profile): returns, vol windows, volume, SPY factors |
| Lookback | 60 trading days |
| Train | ≤ 2017-12-31 |
| Validation | 2018–2019 |
| Test (stress) | 2020 Q1 COVID window |
| `fast_dev` | 200 symbols, 30 epochs |
| `full` | Full universe, 100 epochs |

### Data flow

```text
Stock Market Dataset/          (not in repo, ~2.6 GB — see docs/DATASET.md)
        ↓
scripts/run_preprocessing.py
        ↓
data/processed/*.npz           sequences + config + scaler
        ↓
ML_Iteration_2 notebooks
        ↓
data/models/*                  checkpoints + predictions JSON
```

### What's in the repo

| Included | Excluded |
|----------|----------|
| **Trained** notebooks (J01 complete; J02 stocks + market partial) | Raw `Stock Market Dataset/` (~2.6 GB) |
| Training scripts + scaffold generator | Scaffold jobs (J03, J04, orchestration) |
| dev200 sequence bundles (`.npz`) | Untrained notebooks (J02 etfs, partial J02 market/stocks) |
| Pipeline configs, scalers | Full-universe model-frame CSVs (>100 MB) |
| Trained checkpoints (`.keras`, `.joblib`) | `.venv/` (rebuild locally) |
| Prediction JSON for bake-off comparison | Legacy iteration-1 checkpoints |

Regenerate scaffold notebooks on any machine:

```bash
python scripts/scaffold_ml_iteration2_notebooks.py --job-slug rel_strength
python scripts/scaffold_ml_iteration2_notebooks.py --job-slug vol_regime  # etfs + untrained candidates
```

### Iteration history

| Folder | Focus | Status |
|--------|-------|--------|
| `Analytics notebooks/` | EDA, preprocessing design | Reference |
| `04_ML_Model_Iteration-1/` | AAPL direction (BiGRU, TCN) | Superseded for product |
| `ML_Iteration_2/` | Universe vol/risk + product jobs | **Active** |

---

## Environment

### First-time setup

```bash
bash scripts/setup-wsl-venv.sh    # TensorFlow GPU + deps
source .venv/bin/activate         # or: source scripts/activate-venv.sh
bash scripts/install-jupyter-kernel.sh
```

### Daily activation

```bash
source .venv/bin/activate
```

**WSL pip gotcha:** Do not set `PIP_USER=true` while `.venv` is active (breaks installs inside the virtualenv).

### Hardware notes (RTX 5090)

- TensorFlow 2.21 may **JIT-compile** CUDA kernels for compute capability 12.0a on first GPU use (long first epoch, then normal).
- Notebooks enable **mixed float16** when a GPU is visible.
- If GPU libraries are missing from `LD_LIBRARY_PATH`, training falls back to CPU—check setup cell output for `GPU devices`.

---

## Jupyter kernel

| Field | Value |
|-------|-------|
| Kernelspec name | `stock-prediction` |
| Display name | **`Python (stock-prediction)`** |
| Python binary | `.venv/bin/python` |

Every notebook is pinned to that kernel. **Do not** pick generic `.venv`, `python3`, or legacy kernel entries.

### Register / repair

```bash
bash scripts/cleanup-jupyter-kernels.sh
bash scripts/install-jupyter-kernel.sh
python scripts/fix_notebook_kernels.py
python scripts/fix_notebook_kernels.py --check   # verify only
```

### Compare models after training

```bash
python scripts/compare_job_models.py --job-slug vol_risk --universe stocks --dataset-tag dev200
```

---

## Repository layout

```text
Stock-Volitility-Prediction/
├── README.md                    ← this file
├── docs/DATASET.md              ← raw data setup + dev200 artifact guide
├── ML_Iteration_2/              ← active notebooks + job README
├── Analytics notebooks/         ← EDA, preprocessing
├── 04_ML_Model_Iteration-1/     ← v1 direction models
├── scripts/                     ← preprocessing, training utils, kernel fix
├── data/
│   ├── processed/               ← dev200 sequences, configs, scalers
│   └── models/                  ← checkpoints + predictions JSON
├── Stock Market Dataset/        ← external OHLCV (not in git)
├── .venv/                       ← Python 3.12 + TensorFlow GPU (local)
└── requirements.txt
```

### Key scripts

| Script | Role |
|--------|------|
| `run_preprocessing.py` | Build universe sequences (`--profile plus5`, `--asset-type stock\|etf`) |
| `tf_notebook_utils.py` | GRU/TCN/LSTM/CNN builders, baseline, GPU bootstrap |
| `tf_data.py` | Load `*_sequences_*.npz` bundles |
| `scaffold_ml_iteration2_notebooks.py` | Regenerate notebook tree |
| `fix_notebook_kernels.py` | Unify kernelspec metadata |
| `compare_job_models.py` | Baseline comparison table |
| `replay_notebook_outputs.py` | Restore plots from saved predictions JSON |

---

## Training workflow (J01 stocks example)

1. Open `ML_Iteration_2/J01_vol_risk/stocks/00a_baseline_vol_risk_stocks.ipynb`
2. Kernel: **`Python (stock-prediction)`**
3. Run all cells → saves `data/models/baseline_vol_risk_stocks_dev200_tf_predictions.json`
4. Run `01a` through `01e`; each reports beats/loses vs baseline
5. `python scripts/compare_job_models.py --job-slug vol_risk --universe stocks`
6. Record champion in `ML_Iteration_2/J01_vol_risk/champions.json`
7. Re-run champion with `RUN_MODE = "full"` when raw dataset is available

---

## Further reading

- [ML_Iteration_2/README.md](ML_Iteration_2/README.md) — job portfolio, bake-off rules
- [docs/DATASET.md](docs/DATASET.md) — raw dataset setup, dev200 artifacts
- [04_ML_Model_Iteration-1/MODEL_EVALUATION_v1_vs_v2.md](04_ML_Model_Iteration-1/MODEL_EVALUATION_v1_vs_v2.md) — why v1 direction models were deprioritized
- [Dataset Cleaning and Alignment Report.md](Dataset%20Cleaning%20and%20Alignment%20Report.md)
- [remote-jupyter-rtx5090.md](remote-jupyter-rtx5090.md) — SSH Jupyter on a remote GPU workbench

---

## License

MIT — see [LICENSE](LICENSE).
