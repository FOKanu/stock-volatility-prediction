"""Volatility Lab — presentation dashboard for the stock-volatility ML pipeline.

Run from the project root:
    streamlit run dashboard/app.py

Reads saved artifacts only (data/models/*.json, champions.json); never retrains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import charts
from data_loader import (
    ARCHITECTURE_LABELS,
    JOB_TARGETS,
    JOB_TITLES,
    JOB_TASK_KIND,
    current_leader,
    load_champions,
    market_dates,
    metrics_table,
    predictions_for,
    primary_metric,
    summary_for,
)

st.set_page_config(
    page_title="Volatility Lab · Stock Volatility Prediction",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1250px; }
      h1 { letter-spacing: -0.02em; }
      [data-testid="stMetric"] {
        background: rgba(79, 139, 249, 0.07);
        border: 1px solid rgba(79, 139, 249, 0.18);
        border-radius: 12px;
        padding: 14px 18px;
      }
      [data-testid="stMetricLabel"] { color: #8B93A7; }
      .job-card {
        border: 1px solid rgba(139, 147, 167, 0.25);
        border-radius: 12px;
        padding: 16px 18px;
        height: 100%;
      }
      .job-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; }
      .job-card p { margin: 4px 0; color: #8B93A7; font-size: 0.82rem; }
      .pill {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
      }
      .pill-complete { background: rgba(46,204,143,0.15); color: #2ECC8F; }
      .pill-active { background: rgba(79,139,249,0.15); color: #4F8BF9; }
      .pill-scaffold { background: rgba(139,147,167,0.15); color: #8B93A7; }
      .subtitle { color: #8B93A7; margin-top: -0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

df = metrics_table()
champions = load_champions()

st.title("📈 Volatility Lab")
st.markdown(
    '<p class="subtitle">Machine-learning bake-offs for next-day volatility risk and '
    "21-day volatility regimes across a pooled stock &amp; ETF universe — validated on "
    "2018–19 and stress-tested on the COVID 2020 Q1 crash.</p>",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No prediction artifacts found under data/models/. Train a notebook first.")
    st.stop()

tab_overview, tab_bakeoff, tab_explorer, tab_stress, tab_pipeline = st.tabs(
    ["Overview", "Model Bake-off", "Prediction Explorer", "COVID Stress Test", "Pipeline"]
)

UNIVERSE_LABELS = {"stocks": "Stocks", "etfs": "ETFs", "market": "Market (SPY)"}


def fmt_arch(a: str) -> str:
    return "Baseline" if a == "baseline" else str(a).upper()


# ---------------------------------------------------------------- Overview
with tab_overview:
    n_models = len(df)
    n_candidates = int((~df["is_baseline"]).sum())
    j01 = champions.get("vol_risk", {})
    j01_stock_champ = (j01.get("champions") or {}).get("stocks") or {}
    j01_etf_champ = (j01.get("champions") or {}).get("etfs") or {}
    j02_leader = current_leader(df, "vol_regime", "stocks")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trained models", f"{n_models}", f"{n_candidates} candidates + baselines")
    if j01_stock_champ.get("val_mae") is not None:
        base_mae = (j01.get("baseline", {}).get("stocks") or {}).get("val_mae")
        delta = (
            f"{(j01_stock_champ['val_mae'] - base_mae) / base_mae * 100:+.1f}% vs baseline"
            if base_mae
            else None
        )
        c2.metric(
            "J01 champion · stocks val MAE",
            f"{j01_stock_champ['val_mae']:.4f}",
            delta,
            delta_color="inverse",
        )
    if j01_etf_champ.get("val_mae") is not None:
        base_mae = (j01.get("baseline", {}).get("etfs") or {}).get("val_mae")
        delta = (
            f"{(j01_etf_champ['val_mae'] - base_mae) / base_mae * 100:+.1f}% vs baseline"
            if base_mae
            else None
        )
        c3.metric(
            "J01 champion · ETFs val MAE",
            f"{j01_etf_champ['val_mae']:.4f}",
            delta,
            delta_color="inverse",
        )
    if j02_leader is not None:
        base = df[
            (df["job_slug"] == "vol_regime")
            & (df["universe"] == "stocks")
            & (df["is_baseline"])
        ]
        base_f1 = float(base["val_macro_f1"].iloc[0]) if not base.empty else None
        delta = (
            f"{(j02_leader['val_macro_f1'] - base_f1):+.3f} vs baseline" if base_f1 else None
        )
        c4.metric(
            f"J02 leader · stocks macro-F1 ({fmt_arch(j02_leader['architecture'])})",
            f"{j02_leader['val_macro_f1']:.3f}",
            delta,
        )

    st.markdown("### Job portfolio")
    st.caption(
        "Each job runs a bake-off: one trainable baseline plus five architectures "
        "(GRU, TCN, LSTM, 1D CNN, XGBoost). A champion must beat the baseline on "
        "validation — and prove itself on the COVID stress window."
    )

    job_status = {
        "vol_risk": ("complete", "Champion: XGB (stocks + ETFs)"),
        "vol_regime": (
            "active",
            f"Leader: {fmt_arch(j02_leader['architecture'])} on stocks"
            if j02_leader is not None
            else "Training in progress",
        ),
        "rel_strength": ("scaffold", "Notebooks scaffolded — training pending"),
        "etf_opportunity": ("scaffold", "Notebooks scaffolded — training pending"),
    }
    cols = st.columns(4)
    for col, (slug, (status, note)) in zip(cols, job_status.items()):
        trained = df[df["job_slug"] == slug]
        with col:
            st.markdown(
                f"""
                <div class="job-card">
                  <span class="pill pill-{status}">{status.upper()}</span>
                  <h4>{JOB_TITLES[slug]}</h4>
                  <p><b>Target:</b> {JOB_TARGETS[slug]}</p>
                  <p><b>Models trained:</b> {len(trained)}</p>
                  <p>{note}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Champions & leaders at a glance")
    rows = []
    for slug in ["vol_risk", "vol_regime"]:
        val_col, crisis_col, higher = primary_metric(slug)
        for universe in df[df["job_slug"] == slug]["universe"].unique():
            leader = current_leader(df, slug, universe)
            if leader is None:
                continue
            base = df[
                (df["job_slug"] == slug)
                & (df["universe"] == universe)
                & (df["is_baseline"])
            ]
            base_val = float(base[val_col].iloc[0]) if not base.empty else np.nan
            recorded = (
                (champions.get(slug, {}).get("champions") or {})
                .get(universe, {})
                .get("architecture")
            )
            rows.append(
                {
                    "Job": JOB_TITLES[slug],
                    "Universe": UNIVERSE_LABELS.get(universe, universe),
                    "Best model": fmt_arch(leader["architecture"]),
                    "Status": "Champion (recorded)" if recorded else "Leader (pending)",
                    "Metric": "macro-F1" if higher else "MAE",
                    "Validation": float(leader[val_col]),
                    "Baseline": base_val,
                    "COVID crisis": float(leader[crisis_col]),
                    "Beats baseline (val)": bool(leader["beats_baseline_val"])
                    if leader["beats_baseline_val"] is not None
                    else None,
                }
            )
    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Validation": st.column_config.NumberColumn(format="%.4f"),
            "Baseline": st.column_config.NumberColumn(format="%.4f"),
            "COVID crisis": st.column_config.NumberColumn(format="%.4f"),
            "Beats baseline (val)": st.column_config.CheckboxColumn(),
        },
    )
    st.caption(
        "J01 champions are recorded in champions.json; J02 leaders are computed live "
        "from saved artifacts and remain provisional until a champion is selected."
    )

# ------------------------------------------------------------- Model Bake-off
with tab_bakeoff:
    left, right = st.columns([1, 3])
    with left:
        job = st.radio(
            "Job",
            ["vol_risk", "vol_regime"],
            format_func=lambda s: JOB_TITLES[s],
            key="bakeoff_job",
        )
        universes = sorted(df[df["job_slug"] == job]["universe"].unique())
        universe = st.radio(
            "Universe",
            universes,
            format_func=lambda u: UNIVERSE_LABELS.get(u, u),
            key="bakeoff_universe",
        )
    sub = df[(df["job_slug"] == job) & (df["universe"] == universe)]
    val_col, crisis_col, higher = primary_metric(job)
    metric_name = "macro-F1" if higher else "MAE"
    recorded_champ = (
        (champions.get(job, {}).get("champions") or {}).get(universe, {}) or {}
    ).get("architecture")
    leader = current_leader(df, job, universe)
    crown = recorded_champ or (str(leader["architecture"]) if leader is not None else None)

    with right:
        st.markdown(f"#### {JOB_TITLES[job]} — {UNIVERSE_LABELS.get(universe, universe)}")
        st.caption(
            f"Target: {JOB_TARGETS[job]} · Primary metric: validation {metric_name} · "
            f"{'👑 recorded champion' if recorded_champ else '👑 current leader (champion pending)'}"
        )
        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(
                charts.leaderboard_bars(
                    sub, val_col, f"Validation {metric_name} (2018–19)",
                    higher_is_better=higher, champion=crown,
                ),
                width="stretch",
            )
        with b2:
            st.plotly_chart(
                charts.leaderboard_bars(
                    sub, crisis_col, f"COVID crisis {metric_name} (2020 Q1)",
                    higher_is_better=higher, champion=crown,
                ),
                width="stretch",
            )

    st.markdown("##### Full results")
    if higher:
        show_cols = {
            "architecture": "Model",
            "val_macro_f1": "Val macro-F1",
            "crisis_macro_f1": "Crisis macro-F1",
            "val_accuracy": "Val accuracy",
            "crisis_accuracy": "Crisis accuracy",
            "beats_baseline_val": "Beats baseline (val)",
            "beats_baseline_crisis": "Beats baseline (crisis)",
            "best_epoch": "Best epoch",
        }
    else:
        show_cols = {
            "architecture": "Model",
            "val_mae": "Val MAE",
            "crisis_mae": "Crisis MAE",
            "val_rmse": "Val RMSE",
            "beats_baseline_val": "Beats baseline (val)",
            "beats_baseline_crisis": "Beats baseline (crisis)",
            "best_epoch": "Best epoch",
        }
    table = sub[list(show_cols)].rename(columns=show_cols).copy()
    table["Model"] = table["Model"].map(fmt_arch)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in table.columns
            if table[c].dtype.kind == "f"
        },
    )

    if not higher:
        pers = sub["persistence_mae_val"].dropna()
        if not pers.empty:
            st.caption(
                f"Naive persistence reference (today's |return| → tomorrow): "
                f"val MAE {float(pers.iloc[0]):.4f}, "
                f"crisis MAE {float(sub['persistence_mae_test'].dropna().iloc[0]):.4f} — "
                "every trained model clears it."
            )

    st.markdown("##### Training convergence")
    summaries = [summary_for(p) for p in sub["path"]]
    curves = charts.training_curves(summaries, "loss")
    if curves.data:
        st.plotly_chart(curves, width="stretch")
        st.caption(
            "Validation loss per epoch (XGBoost and the ridge/logistic baseline fit in a "
            "single pass, so they have no curve)."
        )
    else:
        st.info("No multi-epoch training histories for this selection.")

# --------------------------------------------------------- Prediction Explorer
with tab_explorer:
    left, right = st.columns([1, 3])
    with left:
        e_job = st.radio(
            "Job",
            ["vol_risk", "vol_regime"],
            format_func=lambda s: JOB_TITLES[s],
            key="explorer_job",
        )
        e_universes = sorted(df[df["job_slug"] == e_job]["universe"].unique())
        e_universe = st.radio(
            "Universe",
            e_universes,
            format_func=lambda u: UNIVERSE_LABELS.get(u, u),
            key="explorer_universe",
        )
        arch_options = (
            df[(df["job_slug"] == e_job) & (df["universe"] == e_universe)]["architecture"]
            .astype(str)
            .tolist()
        )
        e_arch = st.selectbox(
            "Model", arch_options, format_func=fmt_arch, key="explorer_arch",
            index=len(arch_options) - 1,
        )
        split = st.radio(
            "Evaluation window",
            ["val", "test"],
            format_func=lambda s: "Validation (2018–19)" if s == "val" else "COVID stress (2020 Q1)",
            key="explorer_split",
        )

    row = df[
        (df["job_slug"] == e_job)
        & (df["universe"] == e_universe)
        & (df["architecture"].astype(str) == e_arch)
    ].iloc[0]
    preds = predictions_for(row["path"])[split]
    y_true, y_pred = preds["y_true"], preds["y_pred"]
    window = "Validation 2018–19" if split == "val" else "COVID 2020 Q1"
    label = f"{fmt_arch(e_arch)} · {UNIVERSE_LABELS.get(e_universe, e_universe)}"

    with right:
        st.markdown(f"#### {label}")
        st.caption(
            f"{ARCHITECTURE_LABELS.get(e_arch, '')} · {window} · "
            f"{len(y_true):,} pooled predictions across {row['n_symbols']} symbol(s)"
        )
        if JOB_TASK_KIND[e_job] == "regression":
            m1, m2 = st.columns(2)
            with m1:
                st.plotly_chart(
                    charts.pred_vs_actual(y_true, y_pred, "Predicted vs actual"),
                    width="stretch",
                )
            with m2:
                st.plotly_chart(
                    charts.residual_hist(y_true, y_pred, "Residual distribution"),
                    width="stretch",
                )
            st.plotly_chart(
                charts.calibration_by_decile(
                    y_true, y_pred, "Calibration across volatility deciles"
                ),
                width="stretch",
            )
            st.caption(
                "The decile view shows whether the model tracks rising risk in rank "
                "order — key for a risk engine, even when extreme days are underestimated."
            )
        else:
            y_prob = preds.get("y_prob")
            m1, m2 = st.columns(2)
            with m1:
                st.plotly_chart(
                    charts.confusion_heatmap(y_true, y_pred, "Confusion matrix"),
                    width="stretch",
                )
            with m2:
                if y_prob is not None and len(np.unique(y_true)) > 1:
                    st.plotly_chart(
                        charts.roc_chart(y_true, y_prob, "ROC curve"),
                        width="stretch",
                    )
            if y_prob is not None:
                st.plotly_chart(
                    charts.probability_by_class(
                        y_true, y_prob, "Predicted probability by actual class"
                    ),
                    width="stretch",
                )
            if e_universe == "market" and y_prob is not None:
                val_dates, test_dates = market_dates(
                    len(predictions_for(row["path"])["val"]["y_true"]),
                    len(predictions_for(row["path"])["test"]["y_true"]),
                )
                dates = val_dates if split == "val" else test_dates
                st.plotly_chart(
                    charts.regime_timeline(
                        dates, y_true, y_prob,
                        f"Market regime signal over time — {window}",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Shaded bands mark actual high-volatility periods (dates are "
                    "approximate trading days; exchange holidays not modelled). "
                    "During the COVID window the signal should ramp up as the crash unfolds."
                )

# ----------------------------------------------------------- COVID Stress Test
with tab_stress:
    st.markdown("#### Does validation skill survive a crisis?")
    st.caption(
        "Every model is stress-tested on 2020 Q1 — a regime none of them saw in "
        "training. Points on the dotted line keep their validation performance; "
        "distance from it is the generalization gap."
    )
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(
            charts.stress_scatter(
                df[df["job_slug"] == "vol_risk"],
                "val_mae", "crisis_mae",
                "J01 vol risk — MAE (lower is better)",
                higher_is_better=False,
            ),
            width="stretch",
        )
    with s2:
        st.plotly_chart(
            charts.stress_scatter(
                df[df["job_slug"] == "vol_regime"],
                "val_macro_f1", "crisis_macro_f1",
                "J02 vol regime — macro-F1 (higher is better)",
                higher_is_better=True,
                fmt=".3f",
            ),
            width="stretch",
        )
    st.markdown(
        """
        **How to read this**
        - **J01 (left):** every model's error roughly doubles in the COVID window —
          absolute-return levels were unprecedented — yet all stay well below the naive
          persistence reference. ETF models (diamonds) are consistently easier to fit
          than single stocks.
        - **J02 (right):** the stocks regime classifiers hold up well under stress
          (macro-F1 ≈ 0.65), while the market (SPY-only) models collapse — with a single
          symbol and only 42 crisis days, there is too little signal to generalize.
          That contrast motivates pooled-universe training.
        """
    )

# ------------------------------------------------------------------- Pipeline
with tab_pipeline:
    st.markdown("#### From raw OHLCV to champion models")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          bgcolor="transparent";
          node [shape=box, style="rounded,filled", fillcolor="#1a1f2e",
                color="#4F8BF9", fontcolor="#E4E7F0", fontname="Inter", fontsize=11];
          edge [color="#8B93A7", fontcolor="#8B93A7", fontname="Inter", fontsize=9];
          raw [label="Stock Market Dataset\\n(OHLCV, ~2.6 GB)"];
          prep [label="run_preprocessing.py\\n15 features · 60-day lookback"];
          seq [label="data/processed/*.npz\\nsequences + scaler + config"];
          nb [label="Bake-off notebooks\\nbaseline + 5 architectures"];
          art [label="data/models/\\ncheckpoints + predictions JSON"];
          champ [label="champions.json\\nper job × universe"];
          dash [label="This dashboard", fillcolor="#243252"];
          raw -> prep -> seq -> nb -> art -> champ;
          art -> dash; champ -> dash;
        }
        """
    )

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("##### Evaluation protocol")
        st.markdown(
            """
            | Split | Window | Role |
            |---|---|---|
            | Train | ≤ 2017-12-31 | Model fitting |
            | Validation | 2018 – 2019 | Champion selection |
            | Stress test | 2020 Q1 (COVID) | Crisis robustness |

            Selection rubric: **beat the trainable baseline on validation**, then
            compare crisis performance. Naive persistence (today's move → tomorrow)
            must always be cleared.
            """
        )
        st.markdown("##### Bake-off architectures")
        st.markdown(
            "\n".join(
                f"- **{fmt_arch(k)}** — {v}"
                for k, v in ARCHITECTURE_LABELS.items()
            )
        )
    with p2:
        st.markdown("##### Model inputs (plus5 profile)")
        sample = summary_for(df.iloc[0]["path"])
        feats = sample.get("feature_cols", [])
        st.markdown(
            f"**{len(feats)} features × 60-day lookback**, dev200 subset "
            f"(~200 symbols, {int(df['n_symbols'].max())} eligible stocks):"
        )
        st.code("\n".join(feats), language="text")
        st.markdown("##### Why volatility, not direction?")
        st.markdown(
            """
            Iteration 1 tried next-day *direction* — barely better than a coin flip.
            Volatility is **learnable**: it clusters in time, so risk can be forecast
            even when direction cannot. The product framing follows: help novice
            investors *avoid blowing up* (risk engine + regime gate) rather than
            day-trade noisy signals.
            """
        )

st.divider()
st.caption(
    "Volatility Lab · ML Iteration 2 · dev200 fast-dev runs · all figures replayed "
    "from saved artifacts under data/models/ — no retraining."
)
