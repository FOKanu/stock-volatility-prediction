# Stock Performance Prediction

BIT4.2C Programming for Data Analytics — Final Project (RNN-based stock direction prediction).

## Setup

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
| `notebooks/01_exploratory_data_analysis.ipynb` | EDA |
| `notebooks/02_preprocessing.ipynb` | Preprocessing pipeline → `data/processed/` |

## Docs

- `Dataset Cleaning and Alignment Report.md`
- `remote-jupyter-rtx5090.md` — RTX 5090 Jupyter workflow
