# Volatility Lab — presentation dashboard

Streamlit dashboard over the ML Iteration 2 artifacts. It reads only saved
outputs (`data/models/*_tf_predictions.json` and `ML_Iteration_2/J*/champions.json`)
— it never retrains models or touches notebooks, so it is always safe to run.

## Run

```bash
cd /home/friyekekanu/ai-workspace/experiments/stock-prediction
source .venv/bin/activate
pip install -r dashboard/requirements.txt   # first time only
streamlit run dashboard/app.py
```

Then open http://localhost:8501 (with WSL mirrored networking this works from
the Windows browser too).

## Tabs

| Tab | What it shows |
|---|---|
| **Overview** | KPI cards, job portfolio status (J01–J04), champions/leaders table |
| **Model Bake-off** | Per job × universe leaderboards vs baseline, full metrics, training curves |
| **Prediction Explorer** | Regression: pred-vs-actual, residuals, decile calibration. Classification: confusion matrix, ROC, probability split, SPY regime timeline |
| **COVID Stress Test** | Validation vs 2020 Q1 metric scatter — the generalization gap |
| **Pipeline** | Data-flow diagram, evaluation protocol, features, project rationale |

## Presentation tips

- New models appear automatically: retrain any notebook and refresh — the
  dashboard discovers `*_dev200_tf_predictions.json` artifacts by filename.
- The J02 "leader" is computed live from artifacts and flips to "champion"
  once `champions.json` is filled in.
- The SPY regime-timeline dates are approximate business days (exchange
  holidays are not modelled); everything else is exact from the artifacts.

## Files

- `app.py` — layout, tabs, styling
- `data_loader.py` — artifact discovery + cached JSON loading
- `charts.py` — Plotly figure builders
- `.streamlit/config.toml` — dark theme
