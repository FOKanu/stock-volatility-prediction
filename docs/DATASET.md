# Stock Market Dataset

The raw OHLCV dataset (~2.6 GB) is **not included in this repository** due to size. You need it only if you want to rebuild preprocessing from scratch or run the full (non-dev200) universe.

## Expected layout

Place the dataset at the project root or set `STOCK_DATASET_DIR`:

```text
Stock Market Dataset/
├── symbols_valid_meta.csv
├── stocks/
│   └── {SYMBOL}.csv
└── etfs/
    └── {SYMBOL}.csv
```

## Configure on a new machine

**Option A — project root (recommended)**

```bash
# Linux / macOS / WSL
ln -s /your/path/to/Stock\ Market\ Dataset "Stock Market Dataset"

# Windows (cmd, run as admin)
mklink /J "Stock Market Dataset" "C:\path\to\Stock Market Dataset"
```

**Option B — environment variable**

```bash
cp .env.example .env
# Edit STOCK_DATASET_DIR, then:
export STOCK_DATASET_DIR="/your/path/to/Stock Market Dataset"
```

The training scripts resolve the path in this order: `STOCK_DATASET_DIR` → `./Stock Market Dataset` → known fallback paths in `scripts/paths.py`.

## Continue training without the raw dataset

The repo ships **dev200 processed artifacts** under `data/processed/`:

| Artifact | Purpose |
|----------|---------|
| `*_dev200_*_sequences_plus5.npz` | Train/val/test tensors (200-symbol subset) |
| `pipeline_config_*_dev200_*.json` | Split dates, feature list, lookback |
| `scaler_*_dev200_*.json` | Fitted StandardScaler parameters |
| `eligible_symbols.json` | Symbol universe metadata |

Trained checkpoints live in `data/models/` (`.keras`, `.joblib`, `*_predictions.json`).

Clone the repo, set up the Python environment (see [README](../README.md)), and open a **trained** notebook (J01 vol/risk or partial J02 vol/regime) with kernel **`Python (stock-prediction)`** — no raw dataset required for dev200 workflows.

Scaffold-only jobs (J03, J04, orchestration) are not in the repo; regenerate with `scripts/scaffold_ml_iteration2_notebooks.py`.

## Rebuild preprocessing (requires raw dataset)

```bash
source .venv/bin/activate
python scripts/run_preprocessing.py --help
```

Example (stocks, plus5 profile, dev200):

```bash
python scripts/run_preprocessing.py \
  --asset-type stock \
  --profile plus5 \
  --dev-symbols 200
```
