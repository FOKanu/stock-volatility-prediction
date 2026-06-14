# Stock Performance Prediction

BIT4.2C Programming for Data Analytics — Final Project (RNN-based stock direction prediction).

## Setup (WSL — RTX 5090 workbench)

Project root:

```
~/ai-workspace/experiments/stock-prediction
```

```bash
cd ~/ai-workspace/experiments/stock-prediction
bash scripts/setup-wsl-venv.sh   # first time only (TensorFlow GPU)
source .venv/bin/activate
```

Or use the existing venv after migration:

```bash
source ~/ai-workspace/experiments/stock-prediction/.venv/bin/activate
pip install -r requirements.txt
```

## Setup (legacy Windows path)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The **Stock Market Dataset** (~2.6 GB, not in this repo) must be placed at:

```
Stock Market Dataset/
├── symbols_valid_meta.csv
├── stocks/
└── etfs/
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `Analytics notebooks/01_exploratory_data_analysis.ipynb` | EDA |
| `Analytics notebooks/02_preprocessing.ipynb` | Preprocessing pipeline → `data/processed/` |
| `04_ML_Model_Iteration-1/03a_model_bidirectional_gru.ipynb` | Bidirectional GRU model |
| `04_ML_Model_Iteration-1/03b_model_tcn.ipynb` | TCN model |

## Docs

- `Dataset Cleaning and Alignment Report.md`
- `remote-jupyter-rtx5090.md` — RTX 5090 Jupyter workflow
