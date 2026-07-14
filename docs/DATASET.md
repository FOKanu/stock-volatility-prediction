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
  --symbol-scope all \
  --max-symbols 200
```

## Full-universe storage policy

Full-universe preprocessing is always disk-backed. The previous monolithic NPZ
pipeline exceeded the 47 GiB WSL limit while concatenating 1,838 symbols and was
terminated by the Linux OOM killer.

```bash
python scripts/run_preprocessing.py \
  --asset-type stock \
  --profile plus5 \
  --symbol-scope all \
  --target-type abs_return \
  --storage-format memmap
```

`--storage-format auto` (the default) selects the same memory-safe format for
uncapped `--symbol-scope all` runs. It writes:

```text
data/processed/stocks_universe_sequences_plus5/
├── manifest.json
├── X_train.npy / y_train.npy
├── X_val.npy / y_val.npy
├── X_test.npy / y_test.npy
├── pipeline_config.json
└── scaler.json
```

The `.npy` splits are opened read-only with NumPy memory mapping and fed to
TensorFlow in shuffled blocks. Existing dev200 `.npz` bundles remain supported.
Artifacts are built in a staging directory and atomically published only after
all rows are written; interrupted runs cannot masquerade as complete bundles.

Before full training, verify `manifest.json` and the paired pipeline config have
the expected symbol count and split sizes. A `full` filename alone is not proof
that the entire universe was processed.
