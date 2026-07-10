#!/usr/bin/env python3
"""Render the presentation as a PDF (stock_prediction_project.pdf).

Mirrors the 16 slides of build_presentation.py using matplotlib pages.
Run: .venv/bin/python presentation/export_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = Path(__file__).resolve().parent
ASSETS = OUT_DIR / "assets"

BG = "#1b2631"
PANEL = "#24323f"
HEADER = "#11181f"
ACCENT = "#f39c12"
GOOD = "#2ecc71"
WARN = "#e74c3c"
TEXT = "#ecf0f1"
MUTED = "#95a5a6"

W, H = 13.333, 7.5


def new_page():
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    return fig, ax


def title(ax, text, subtitle=None):
    ax.text(0.55, 0.65, text, fontsize=26, color=ACCENT, weight="bold", va="top")
    if subtitle:
        ax.text(0.55, 1.15, subtitle, fontsize=13, color=MUTED, va="top")


def bullets(ax, x, y, items, line_h=0.42):
    """items: list of (text, size, color, bold, level)."""
    cur = y
    for text, size, color, bold, level in items:
        indent = x + level * 0.35
        prefix = "" if level == 0 and bold else ("• " if level else "")
        ax.text(indent, cur, prefix + text, fontsize=size, color=color,
                weight="bold" if bold else "normal", va="top", wrap=True)
        cur += line_h * (size / 13.0)
    return cur


def table(ax, x, y, col_widths, rows, font_size=10.5, row_h=0.42):
    for r, row in enumerate(rows):
        cx = x
        ry = y + r * row_h
        for c, cell in enumerate(row):
            w = col_widths[c]
            face = HEADER if r == 0 else PANEL
            ax.add_patch(plt.Rectangle((cx, ry), w, row_h, facecolor=face,
                                       edgecolor=BG, lw=1.2))
            ax.text(cx + 0.12, ry + row_h / 2, str(cell), fontsize=font_size,
                    color=ACCENT if r == 0 else TEXT,
                    weight="bold" if r == 0 else "normal", va="center")
            cx += w
    return y + len(rows) * row_h


def image(ax, path, x, y, width):
    img = mpimg.imread(str(path))
    ih, iw = img.shape[:2]
    height = width * ih / iw
    ax.imshow(img, extent=(x, x + width, y + height, y), aspect="auto", zorder=2)
    return y + height


def build() -> Path:
    out = OUT_DIR / "stock_prediction_project.pdf"
    with PdfPages(out) as pdf:
        # 1 — Title
        fig, ax = new_page()
        ax.text(W / 2, 2.7, "Stock Volatility Prediction: ML Model Portfolio",
                fontsize=30, color=ACCENT, weight="bold", ha="center")
        ax.text(W / 2, 3.5, "From next-day direction calls to product-oriented risk intelligence",
                fontsize=16, color=TEXT, ha="center")
        ax.text(W / 2, 4.6, "Francis Okorie Iyeke-Kanu", fontsize=20, color=TEXT, weight="bold", ha="center")
        ax.text(W / 2, 5.5, "BIT4.2C Analytics Project → ML Iteration 2", fontsize=12,
                color=MUTED, ha="center")
        ax.text(W / 2, 5.9, "July 2026", fontsize=12, color=MUTED, ha="center")
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 2 — Vision & goals
        fig, ax = new_page()
        title(ax, "Project Vision & Goals",
              "Help novice investors avoid blowing up — not day-trade on noisy signals")
        table(ax, 0.55, 1.75, [2.6, 4.6, 5.0], [
            ["Product layer", "Purpose", "Example output"],
            ["Risk engine", "Flag turbulent periods", '"High turbulence expected tomorrow"'],
            ["Regime gate", "Market / stock weather", "Reduce-risk mode in crisis-like regimes"],
            ["Opportunity ranker", "Names vs SPY over ~21 days", "Watchlist, not buy/sell arrows"],
            ["ETF lane", "Core + satellite allocation hints", "Favor ETFs vs benchmark"],
            ["Coach (rules)", "Hard limits", "Max position %, diversification"],
        ])
        bullets(ax, 0.55, 4.8, [
            ('"Beat the market" reframed as risk-adjusted, benchmark-relative performance over '
             'weeks — not binary next-day direction calls.', 13.5, ACCENT, True, 0),
            ("ML principle: predict learnable targets (volatility first, direction last), and "
             "compare every model against honest baselines.", 12.5, TEXT, False, 0),
        ], line_h=0.55)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 3 — Dataset
        fig, ax = new_page()
        title(ax, "The Dataset",
              "Raw OHLCV panel, ~2.6 GB — heavy cleaning required before modeling")
        table(ax, 0.55, 1.75, [2.6, 3.3], [
            ["Component", "Value"],
            ["Stock CSV files", "5,884 symbols"],
            ["ETF CSV files", "2,165 symbols"],
            ["Columns", "Date, OHLC, Adj Close, Volume"],
            ["Date range", "~1962 → April 2020"],
            ["Dev universe", "dev200 (200 symbols)"],
        ])
        bullets(ax, 7.0, 1.75, [
            ("Cleaning challenges found in EDA:", 13.5, ACCENT, True, 0),
            ("Uneven history per symbol: 1 to 12,676 trading days", 12, TEXT, False, 1),
            ("No shared calendar index across files", 12, TEXT, False, 1),
            ("~5% of files contain nulls; ~58% have zero-volume days", 12, TEXT, False, 1),
            ("Metadata ↔ file inventory mismatches", 12, TEXT, False, 1),
            ("Fixes applied:", 13.5, ACCENT, True, 0),
            ("Minimum-history threshold for the training universe", 12, TEXT, False, 1),
            ("Explicit date-merge alignment for multi-symbol features", 12, TEXT, False, 1),
            ("Time-based train/val/test splits (never random rows)", 12, TEXT, False, 1),
        ])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 4 — Hardest challenge: understanding the data
        fig, ax = new_page()
        title(ax, "Challenges: Understanding the Data",
              "The hardest part of the project — before any model could train")
        bullets(ax, 0.55, 1.75, [
            ("Why this was hardest:", 13.5, ACCENT, True, 0),
            ("7,000+ symbols with uneven histories (1 to 12,676 trading days each)", 11.5, TEXT, False, 1),
            ("No shared calendar — every CSV had its own date index", 11.5, TEXT, False, 1),
            ("Nulls, zero-volume days, and metadata/file mismatches", 11.5, TEXT, False, 1),
            ("Deciding what 'a training sample' even means across a pooled universe", 11.5, TEXT, False, 1),
            ("What I had to figure out:", 13.5, ACCENT, True, 0),
            ("Which symbols belong in the training universe", 11.5, TEXT, False, 1),
            ("How to align SPY market factors with each stock's calendar", 11.5, TEXT, False, 1),
            ("Whether missing data means 'skip' or 'zero'", 11.5, TEXT, False, 1),
        ], line_h=0.38)
        bullets(ax, 6.8, 1.75, [
            ("Research-driven decisions:", 13.5, ACCENT, True, 0),
            ("Extensive EDA before modeling — not optional", 11.5, TEXT, False, 1),
            ("Documented every cleaning rule in the Dataset Alignment Report", 11.5, TEXT, False, 1),
            ("Time-based splits only — random row splits would leak future info", 11.5, TEXT, False, 1),
            ("COVID Q1 2020 held out as a stress test, not part of training", 11.5, TEXT, False, 1),
            ("Outcome:", 13.5, GOOD, True, 0),
            ("A reproducible preprocessing pipeline that turns 2.6 GB of raw OHLCV "
             "into standardized 60-day sequence bundles per job.", 11.5, GOOD, False, 1),
        ], line_h=0.38)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 5 — Architecture research
        fig, ax = new_page()
        title(ax, "Challenges: Architecture Research & Target Design",
              "Matching ML architectures to learnable financial targets")
        bullets(ax, 0.55, 1.75, [
            ("Not every architecture fits every target — I researched and tested:", 13.5, ACCENT, True, 0),
            ("Sequence models (GRU, LSTM, TCN, 1D-CNN) for temporal patterns", 11.5, TEXT, False, 1),
            ("Tree ensembles (XGBoost) on flattened sequences for tabular signal", 11.5, TEXT, False, 1),
            ("Linear baselines (Ridge / logistic) as honest references", 11.5, TEXT, False, 1),
        ], line_h=0.42)
        table(ax, 0.55, 3.4, [3.5, 2.8, 6.0], [
            ["Target type", "Best fit", "Why"],
            ["Next-day direction (v1)", "BiGRU, TCN", "Sequence classification — but signal was too weak"],
            ["Next-day |return| (J01)", "XGBoost", "Volatility persistence; trees beat deep nets"],
            ["21d vol regime (J02)", "LSTM", "Longer-horizon regime memory; crisis stress-test critical"],
            ["21d excess vs SPY (J03–J04)", "TBD", "Ranking task — scaffolded for future work"],
        ], font_size=10, row_h=0.4)
        bullets(ax, 0.55, 5.7, [
            ("Key insight: the target matters more than the architecture. Pivoting from "
             "direction to volatility unlocked real predictive skill.", 12.5, GOOD, True, 0),
        ])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 6 — Preprocessing & training scale
        fig, ax = new_page()
        title(ax, "Challenges: Preprocessing Pipeline & Training 20+ Models",
              "Building infrastructure to compare architectures fairly at scale")
        table(ax, 0.55, 1.7, [2.5, 3.3], [
            ["Pipeline stage", "Output"],
            ["Universe selection", "eligible_symbols.json"],
            ["Feature engineering", "15 features (plus5 profile)"],
            ["Sequence builder", "60-day windows → .npz bundles"],
            ["Scaler + config", "JSON artifacts per job/universe"],
            ["Notebook bake-offs", "1 baseline + 5 candidates each"],
        ], font_size=10.5, row_h=0.38)
        bullets(ax, 6.7, 1.7, [
            ("Training scale:", 13.5, ACCENT, True, 0),
            ("20+ distinct model runs across jobs and universes", 11.5, TEXT, False, 1),
            ("J01: 12 models (6 stocks + 6 ETFs) — complete", 11.5, TEXT, False, 1),
            ("J02: 9+ models (stocks + market partial) — in progress", 11.5, TEXT, False, 1),
            ("Each run exports checkpoints + predictions JSON for fair comparison", 11.5, TEXT, False, 1),
            ("Engineering overhead:", 13.5, ACCENT, True, 0),
            ("Scaffold generator for consistent notebook structure", 11.5, TEXT, False, 1),
            ("GPU setup on WSL2 (RTX 5090), mixed float16, kernel management", 11.5, TEXT, False, 1),
            ("compare_job_models.py + replay for reproducibility", 11.5, TEXT, False, 1),
        ], line_h=0.38)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 7 — Iteration 1 plan + failure
        fig, ax = new_page()
        title(ax, "Initial Plan (Iteration 1): Predict AAPL Direction",
              "Next-day UP/DOWN · 60-day lookback · 7 features · BiGRU and TCN")
        bullets(ax, 0.55, 1.8, [
            ("v1 failed in two distinct ways:", 14, ACCENT, True, 0),
            ("BiGRU — severe overfitting: train AUC ~0.89 vs validation ~0.49 (gap of 0.40)",
             12.5, WARN, False, 1),
            ("TCN — model collapse: predicted P(UP) ≈ 0.52 for every sample (std 0.004), "
             "equivalent to an 'always UP' strategy", 12.5, WARN, False, 1),
        ])
        table(ax, 0.55, 3.6, [1.7, 5.4, 1.5, 3.6], [
            ["Model", "v1 diagnosis", "Val AUC", "Verdict"],
            ["BiGRU", "Memorized training data, zero generalization", "0.493",
             "Worse than 'always UP'"],
            ["TCN", "Constant output — no sample discrimination", "0.499", "Random ranking"],
        ])
        bullets(ax, 0.55, 5.4, [
            ("Key insight: high accuracy / F1 were illusions of class imbalance (~55% UP days) — "
             "not real predictive skill.", 13.5, ACCENT, True, 0),
        ])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 5 — v2
        fig, ax = new_page()
        title(ax, "Iteration 1 v2: Fixes Worked, But the Ceiling Was Low",
              "Training pathologies solved — yet direction remained barely learnable")
        table(ax, 0.55, 1.8, [2.5, 5.2, 4.5], [
            ["Fix", "Change", "Result"],
            ["GRU anti-overfit", "61k → 4.5k params, recurrent dropout, LR 3e-4",
             "Train–val AUC gap 0.40 → 0.005"],
            ["TCN anti-collapse", "Last-timestep readout (was global pooling), AdamW",
             "Val AUC 0.499 → 0.559; pred std 0.004 → 0.051"],
            ["Honest thresholds", "Tuned on validation, applied unchanged to test",
             "No more baseline-mimicking F1"],
        ])
        bullets(ax, 0.55, 4.2, [
            ("Verdict: even after fixes, best validation AUC was 0.559 — a marginal edge, not "
             "production-ready for directional trading.", 13.5, WARN, True, 0),
            ("Conclusion that drove the pivot: next-day direction is nearly random at daily "
             "horizon. Volatility, however, is persistent and learnable.", 13.5, GOOD, True, 0),
        ], line_h=0.62)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 6 — Pivot
        fig, ax = new_page()
        title(ax, "The Pivot: Predict What Is Learnable",
              "ML Iteration 2 — a portfolio of jobs, each mapped to a product feature")
        table(ax, 0.55, 1.8, [2.4, 4.1, 3.4, 2.3], [
            ["Job", "Prediction target", "Product feature", "Status"],
            ["J01 vol_risk", "Next-day |log return|", "Turbulence alert", "Complete"],
            ["J02 vol_regime", "21-day high-volatility regime", "Market weather", "Active"],
            ["J03 rel_strength", "21d excess return vs SPY", "Stock watchlist", "Planned"],
            ["J04 etf_opportunity", "21d excess return vs SPY (ETFs)", "ETF allocation hints",
             "Planned"],
        ])
        bullets(ax, 0.55, 4.6, [
            ("Separate models per job — not one mega-network. Shared preprocessing feeds each "
             "job; an orchestration layer merges champion scores.", 12.5, TEXT, False, 0),
            ("Every job serves the same product story: risk control and benchmark-relative "
             "selection for novice investors.", 12.5, TEXT, False, 0),
        ], line_h=0.6)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 7 — Methodology
        fig, ax = new_page()
        title(ax, "Methodology: Bake-Offs With Honest Baselines",
              "1 trainable baseline + 5 architectures per job/universe → champion")
        bullets(ax, 0.55, 1.8, [
            ("Candidates per bake-off:", 13.5, ACCENT, True, 0),
            ("Baseline: Ridge / logistic on last-day features", 12, TEXT, False, 1),
            ("GRU · TCN · LSTM · 1D-CNN · XGBoost", 12, TEXT, False, 1),
            ("Champion rubric:", 13.5, ACCENT, True, 0),
            ("1. Must beat trained baseline on validation", 12, TEXT, False, 1),
            ("2. Best COVID-crisis performance", 12, TEXT, False, 1),
            ("3. Tie-breakers per job (e.g. top-10% vol AUC)", 12, TEXT, False, 1),
        ])
        table(ax, 7.0, 1.8, [2.2, 3.2], [
            ["Split", "Period"],
            ["Train", "≤ 2017-12-31"],
            ["Validation", "2018 – 2019"],
            ["Stress test", "2020 Q1 COVID crash"],
        ])
        bullets(ax, 7.0, 3.9, [
            ("Every model exports reproducible artifacts (predictions JSON + checkpoints) for "
             "cross-model comparison and replay.", 12, TEXT, False, 0),
            ("Naive references also reported: persistence and train-mean.", 12, TEXT, False, 0),
        ], line_h=0.6)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 8 — Training setup
        fig, ax = new_page()
        title(ax, "Training Setup")
        table(ax, 0.55, 1.7, [2.4, 4.0], [
            ["Setting", "Value"],
            ["Features", "15 engineered (plus5 profile)"],
            ["Lookback", "60 trading days"],
            ["Universe", "dev200 (200 symbols, pooled)"],
            ["fast_dev", "30 epochs (bake-offs)"],
            ["full", "100 epochs (champions)"],
        ])
        bullets(ax, 7.3, 1.7, [
            ("Feature groups:", 13.5, ACCENT, True, 0),
            ("Returns (log, absolute, squared)", 12, TEXT, False, 1),
            ("Volatility windows (5d, 21d, ratios)", 12, TEXT, False, 1),
            ("Volume (log volume, surge)", 12, TEXT, False, 1),
            ("Technicals (RSI-14, intraday range)", 12, TEXT, False, 1),
            ("Market factors (SPY return, SPY vol)", 12, TEXT, False, 1),
            ("Hardware & stack:", 13.5, ACCENT, True, 0),
            ("RTX 5090 GPU on WSL2, TensorFlow 2.21, mixed float16", 12, TEXT, False, 1),
            ("XGBoost / HistGBR on CPU with flattened sequences", 12, TEXT, False, 1),
        ])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 9 — J01 results
        fig, ax = new_page()
        title(ax, "Results: J01 Vol/Risk — Complete", "Champion: XGBoost in both universes")
        img_bottom = image(ax, ASSETS / "j01_leaderboard.png", 1.3, 1.55, 10.7)
        table(ax, 0.55, img_bottom + 0.15, [2.2, 3.2, 3.2, 3.6], [
            ["Universe", "Champion val MAE", "Baseline val MAE", "Beats baseline (val/crisis)"],
            ["ETFs", "0.0043", "0.0055", "Yes / Yes"],
            ["Stocks", "0.0143", "0.0155", "Yes / slightly worse"],
        ], font_size=10, row_h=0.38)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 10 — J02 results
        fig, ax = new_page()
        title(ax, "Results: J02 Vol Regime — In Progress",
              "Classifying 21-day high-volatility regimes (macro-F1)")
        img_bottom = image(ax, ASSETS / "j02_val_vs_crisis.png", 1.3, 1.55, 10.7)
        bullets(ax, 0.55, img_bottom + 0.2, [
            ("Stocks (5/6 trained): LSTM leads — val F1 0.802 vs baseline 0.759; crisis F1 0.660.",
             11.5, GOOD, False, 0),
            ("Market/SPY (4/6 trained): baseline still leads validation (0.660); LSTM best on "
             "crisis (0.354).", 11.5, TEXT, False, 0),
            ("Cautionary tale: market TCN scored 0.54 on validation but collapsed to 0.15 on the "
             "COVID window — why every candidate is stress-tested.", 11.5, WARN, False, 0),
        ], line_h=0.5)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 14 — What I learned
        fig, ax = new_page()
        title(ax, "What I Learned",
              "Insights that changed how I think about financial ML")
        bullets(ax, 0.55, 1.75, [
            ("Next-day price direction is difficult and unreliable", 13.5, WARN, True, 0),
            ("Daily returns are close to random walk at 1-day horizon", 11.5, TEXT, False, 1),
            ("High accuracy/F1 can be an illusion of class imbalance (~55% UP days)", 11.5, TEXT, False, 1),
            ("Even after fixing overfitting, best val AUC was only 0.559 — not tradeable", 11.5, TEXT, False, 1),
            ("Without strong exogenous signals, direction models fail", 11.5, TEXT, False, 1),
            ("Volatility / risk prediction is more useful", 13.5, GOOD, True, 0),
            ("Volatility clusters and persists — models can beat naive baselines", 11.5, TEXT, False, 1),
            ("Actionable: 'expect turbulence tomorrow' vs 'buy/sell AAPL'", 11.5, TEXT, False, 1),
            ("Better business fit: position sizing, risk alerts, allocation discipline", 11.5, TEXT, False, 1),
        ], line_h=0.36)
        bullets(ax, 6.8, 1.75, [
            ("Markets behave differently in crisis periods", 13.5, ACCENT, True, 0),
            ("2020 Q1 COVID crash broke patterns learned on 2018–2019 validation", 11.5, TEXT, False, 1),
            ("Models strong in calm markets can collapse in crises "
             "(market TCN: val F1 0.54 → crisis F1 0.15)", 11.5, WARN, False, 1),
            ("Stress-testing is not optional — it must be part of model selection", 11.5, TEXT, False, 1),
            ("Regime detection (J02) directly addresses this structural break", 11.5, TEXT, False, 1),
            ("Broader lessons:", 13.5, ACCENT, True, 0),
            ("Simple baselines (Ridge, persistence) are genuinely hard to beat", 11.5, TEXT, False, 1),
            ("Define success before training: baseline-relative, crisis-tested, product-mapped",
             11.5, GOOD, False, 1),
        ], line_h=0.36)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 15 — Applications
        fig, ax = new_page()
        title(ax, "Applications & Potential Use",
              "From research project to novice-investor risk intelligence")
        table(ax, 0.55, 1.7, [3.0, 3.2, 6.0], [
            ["Application", "Model layer", "User value"],
            ["Turbulence alerts", "J01 vol/risk", "Warn before high-volatility days"],
            ["Market weather dashboard", "J02 vol/regime", "Defensive posture in crisis regimes"],
            ["Stock watchlist", "J03 rel strength", "Names outperforming SPY over ~21 days"],
            ["ETF allocation hints", "J04 ETF opportunity", "Core vs satellite ETF suggestions"],
            ["Portfolio coach", "Rule engine + ML scores", "Position limits, diversification"],
        ], font_size=10, row_h=0.38)
        bullets(ax, 0.55, 4.4, [
            ("Who benefits:", 13.5, ACCENT, True, 0),
            ("Novice investors — calmer decisions, not day-trading on noisy signals", 11.5, TEXT, False, 1),
            ("Portfolio apps / fintech — embed risk scores as a product feature", 11.5, TEXT, False, 1),
            ("Researchers — reproducible bake-off framework for financial time-series ML", 11.5, TEXT, False, 1),
            ("Deployment path:", 13.5, ACCENT, True, 0),
            ("Champion models → orchestration layer → dashboard / API → alerts & watchlists",
             11.5, TEXT, False, 1),
            ("GitHub repo ships dev200 artifacts so training can resume on any machine",
             11.5, TEXT, False, 1),
        ], line_h=0.38)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # 16 — Roadmap
        fig, ax = new_page()
        title(ax, "Future Roadmap & Conclusion")
        bullets(ax, 0.55, 1.8, [
            ("Near-term", 14.5, ACCENT, True, 0),
            ("Finish J02 (ETF universe + missing architectures), select regime champions",
             12, TEXT, False, 1),
            ("Build J03 relative-strength and J04 ETF-opportunity rankers", 12, TEXT, False, 1),
            ("Scale-up", 14.5, ACCENT, True, 0),
            ("Orchestration layer: merge champion scores into unified product signals",
             12, TEXT, False, 1),
            ("Full-universe champion retraining (beyond dev200) + walk-forward validation",
             12, TEXT, False, 1),
            ("Product delivery", 14.5, ACCENT, True, 0),
            ("Dashboard demo → turbulence alerts, market weather, watchlist, ETF hints",
             12, TEXT, False, 1),
        ])
        bullets(ax, 0.55, 5.9, [
            ('Takeaway: pivoting from "predict direction" to "predict risk" turned a failing '
             "model into a working model portfolio.", 15, GOOD, True, 0),
        ])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    return out


if __name__ == "__main__":
    path = build()
    print(f"Saved: {path}")
