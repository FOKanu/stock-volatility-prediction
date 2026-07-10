#!/usr/bin/env python3
"""Build the project presentation (stock_prediction_project.pptx).

Charts are generated from real artifacts in data/models/*_predictions.json.
Re-run after retraining to refresh numbers: .venv/bin/python presentation/build_presentation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"
OUT_DIR = Path(__file__).resolve().parent
ASSETS = OUT_DIR / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- palette
DARK = RGBColor(0x1B, 0x26, 0x31)      # slide background
PANEL = RGBColor(0x24, 0x32, 0x40)     # card background
ACCENT = RGBColor(0xF3, 0x9C, 0x12)    # orange accent
GOOD = RGBColor(0x2E, 0xCC, 0x71)
WARN = RGBColor(0xE7, 0x4C, 0x3C)
TEXT = RGBColor(0xEC, 0xF0, 0xF1)
MUTED = RGBColor(0x95, 0xA5, 0xA6)
BASELINE_GREY = "#95a5a6"
CHAMPION_ORANGE = "#f39c12"
BAR_BLUE = "#3498db"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ---------------------------------------------------------------- data
def load_metrics(job_slug: str) -> dict[str, list[dict]]:
    """{universe: [row, ...]} for all dev200 artifacts of a job."""
    out: dict[str, list[dict]] = {}
    for p in sorted(MODELS.glob(f"*_{job_slug}_*_dev200_tf_predictions.json")):
        d = json.loads(p.read_text())
        row = {
            "architecture": d.get("architecture"),
            "is_baseline": bool(d.get("is_baseline")),
            "val_mae": d.get("val_mae"),
            "crisis_mae": d.get("crisis_mae"),
            "val_macro_f1": d.get("val_macro_f1"),
            "crisis_macro_f1": d.get("crisis_macro_f1"),
        }
        out.setdefault(d.get("universe", "?"), []).append(row)
    return out


def make_j01_chart() -> Path:
    data = load_metrics("vol_risk")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#1b2631")
    for ax, universe in zip(axes, ["stocks", "etfs"]):
        rows = sorted(data[universe], key=lambda r: r["val_mae"])
        names = [r["architecture"] for r in rows]
        vals = [r["val_mae"] for r in rows]
        colors = []
        for i, r in enumerate(rows):
            if r["is_baseline"]:
                colors.append(BASELINE_GREY)
            elif i == 0:
                colors.append(CHAMPION_ORANGE)
            else:
                colors.append(BAR_BLUE)
        ax.bar(names, vals, color=colors)
        ax.set_title(f"J01 {universe} — validation MAE (lower is better)",
                     color="white", fontsize=11)
        ax.set_facecolor("#1b2631")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#555")
        ax.yaxis.label.set_color("white")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom",
                    color="white", fontsize=7.5)
    fig.suptitle("Champion = orange · Baseline = grey", color="#f39c12", fontsize=10, y=1.0)
    fig.tight_layout()
    path = ASSETS / "j01_leaderboard.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def make_j02_chart() -> Path:
    data = load_metrics("vol_regime")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#1b2631")
    for ax, universe in zip(axes, ["stocks", "market"]):
        rows = sorted(data[universe], key=lambda r: -(r["val_macro_f1"] or 0))
        names = [r["architecture"] for r in rows]
        val = [r["val_macro_f1"] for r in rows]
        crisis = [r["crisis_macro_f1"] for r in rows]
        x = range(len(names))
        w = 0.38
        ax.bar([i - w / 2 for i in x], val, width=w, color=BAR_BLUE, label="Validation F1")
        ax.bar([i + w / 2 for i in x], crisis, width=w, color="#e67e22", label="COVID crisis F1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"J02 {universe} — macro-F1 (higher is better)", color="white", fontsize=11)
        ax.set_facecolor("#1b2631")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#555")
        ax.legend(fontsize=8, facecolor="#24323f", labelcolor="white", edgecolor="#555")
    fig.suptitle("Regime classification: validation vs COVID stress test", color="#f39c12",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    path = ASSETS / "j02_val_vs_crisis.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- pptx helpers
def add_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK
    return slide


def add_text(slide, left, top, width, height, lines, *, align=PP_ALIGN.LEFT):
    """lines: list of (text, size_pt, color, bold, bullet_level)."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, (text, size, color, bold, level) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.level = level
        run = p.add_run()
        run.text = text
        f = run.font
        f.size = Pt(size)
        f.color.rgb = color
        f.bold = bold
        p.space_after = Pt(int(size * 0.45))
    return box


def add_title(slide, title, subtitle=None):
    lines = [(title, 30, ACCENT, True, 0)]
    if subtitle:
        lines.append((subtitle, 15, MUTED, False, 0))
    add_text(slide, Inches(0.55), Inches(0.3), Inches(12.2), Inches(1.2), lines)


def add_table(slide, left, top, width, rows_data, col_widths=None, font_size=12):
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    height = Inches(0.4 * n_rows)
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for c, wfrac in enumerate(col_widths):
            table.columns[c].width = Emu(int(width * wfrac / total))
    for r, row in enumerate(rows_data):
        for c, cell_text in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if r else RGBColor(0x11, 0x18, 0x1F)
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(cell_text)
            f = run.font
            f.size = Pt(font_size)
            f.bold = r == 0
            f.color.rgb = ACCENT if r == 0 else TEXT
    return shape


def bullet(text, size=16, color=TEXT, bold=False, level=0):
    return (text, size, color, bold, level)


# ---------------------------------------------------------------- deck
def build() -> Path:
    j01_chart = make_j01_chart()
    j02_chart = make_j02_chart()

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # 1 — Title
    s = add_slide(prs)
    add_text(s, Inches(1.0), Inches(1.9), Inches(11.3), Inches(2.4), [
        ("Stock Volatility Prediction: ML Model Portfolio", 40, ACCENT, True, 0),
        ("From next-day direction calls to product-oriented risk intelligence", 20, TEXT, False, 0),
    ])
    add_text(s, Inches(1.0), Inches(4.5), Inches(11.3), Inches(0.8), [
        ("Francis Okorie Iyeke-Kanu", 24, TEXT, True, 0),
    ])
    add_text(s, Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.2), [
        ("BIT4.2C Analytics Project → ML Iteration 2", 15, MUTED, False, 0),
        ("July 2026", 15, MUTED, False, 0),
    ])

    # 2 — Vision & goals
    s = add_slide(prs)
    add_title(s, "Project Vision & Goals",
              "Help novice investors avoid blowing up — not day-trade on noisy signals")
    add_table(s, Inches(0.55), Inches(1.7), Inches(12.2), [
        ["Product layer", "Purpose", "Example output"],
        ["Risk engine", "Flag turbulent periods", '"High turbulence expected tomorrow"'],
        ["Regime gate", "Market / stock weather", "Reduce-risk mode in crisis-like regimes"],
        ["Opportunity ranker", "Names vs SPY over ~21 days", "Watchlist, not buy/sell arrows"],
        ["ETF lane", "Core + satellite allocation hints", "Favor ETFs vs benchmark"],
        ["Coach (rules)", "Hard limits", "Max position %, diversification"],
    ], col_widths=[0.22, 0.38, 0.40], font_size=13)
    add_text(s, Inches(0.55), Inches(4.6), Inches(12.2), Inches(1.6), [
        bullet('"Beat the market" reframed as risk-adjusted, benchmark-relative performance '
               "over weeks — not binary next-day direction calls.", 17, ACCENT, True),
        bullet("ML principle: predict learnable targets (volatility first, direction last), and "
               "compare every model against honest baselines.", 16),
    ])

    # 3 — Dataset
    s = add_slide(prs)
    add_title(s, "The Dataset", "Raw OHLCV panel, ~2.6 GB — heavy cleaning required before modeling")
    add_table(s, Inches(0.55), Inches(1.6), Inches(5.9), [
        ["Component", "Value"],
        ["Stock CSV files", "5,884 symbols"],
        ["ETF CSV files", "2,165 symbols"],
        ["Columns", "Date, OHLC, Adj Close, Volume"],
        ["Date range", "~1962 → April 2020"],
        ["Dev universe", "dev200 (200 symbols)"],
    ], col_widths=[0.45, 0.55], font_size=13)
    add_text(s, Inches(6.9), Inches(1.6), Inches(6.0), Inches(4.6), [
        bullet("Cleaning challenges found in EDA:", 17, ACCENT, True),
        bullet("Uneven history per symbol: 1 to 12,676 trading days", 15, TEXT, False, 1),
        bullet("No shared calendar index across files", 15, TEXT, False, 1),
        bullet("~5% of files contain nulls; ~58% have zero-volume days", 15, TEXT, False, 1),
        bullet("Metadata ↔ file inventory mismatches", 15, TEXT, False, 1),
        bullet("Fixes applied:", 17, ACCENT, True),
        bullet("Minimum-history threshold for the training universe", 15, TEXT, False, 1),
        bullet("Explicit date-merge alignment for multi-symbol features", 15, TEXT, False, 1),
        bullet("Time-based train/val/test splits (never random rows)", 15, TEXT, False, 1),
    ])

    # 4 — Hardest challenge: understanding the data
    s = add_slide(prs)
    add_title(s, "Challenges: Understanding the Data",
              "The hardest part of the project — before any model could train")
    add_text(s, Inches(0.55), Inches(1.6), Inches(6.0), Inches(5.4), [
        bullet("Why this was hardest:", 17, ACCENT, True),
        bullet("7,000+ symbols with uneven histories (1 to 12,676 trading days each)", 15, TEXT, False, 1),
        bullet("No shared calendar — every CSV had its own date index", 15, TEXT, False, 1),
        bullet("Nulls, zero-volume days, and metadata/file mismatches", 15, TEXT, False, 1),
        bullet("Deciding what 'a training sample' even means across a pooled universe", 15, TEXT, False, 1),
        bullet("What I had to figure out:", 17, ACCENT, True),
        bullet("Which symbols belong in the training universe (eligible_symbols.json)", 15, TEXT, False, 1),
        bullet("How to align SPY market factors with each stock's calendar", 15, TEXT, False, 1),
        bullet("Whether missing data means 'skip' or 'zero' — and when each applies", 15, TEXT, False, 1),
    ])
    add_text(s, Inches(6.8), Inches(1.6), Inches(6.0), Inches(5.4), [
        bullet("Research-driven decisions:", 17, ACCENT, True),
        bullet("Extensive EDA before modeling — not optional", 15, TEXT, False, 1),
        bullet("Documented every cleaning rule in the Dataset Alignment Report", 15, TEXT, False, 1),
        bullet("Time-based splits only — random row splits would leak future information", 15, TEXT, False, 1),
        bullet("COVID Q1 2020 held out as a stress test, not part of training", 15, TEXT, False, 1),
        bullet("Outcome:", 17, GOOD, True),
        bullet("A reproducible preprocessing pipeline that turns 2.6 GB of raw OHLCV "
               "into standardized 60-day sequence bundles per job.", 15, GOOD, False, 1),
    ])

    # 5 — Architecture research & target design
    s = add_slide(prs)
    add_title(s, "Challenges: Architecture Research & Target Design",
              "Matching ML architectures to learnable financial targets")
    add_text(s, Inches(0.55), Inches(1.6), Inches(12.2), Inches(2.0), [
        bullet("Not every architecture fits every target — I researched and tested:", 17, ACCENT, True),
        bullet("Sequence models (GRU, LSTM, TCN, 1D-CNN) for temporal patterns in 60-day windows",
               15, TEXT, False, 1),
        bullet("Tree ensembles (XGBoost / HistGBR) on flattened sequences for tabular signal",
               15, TEXT, False, 1),
        bullet("Linear baselines (Ridge / logistic) as honest, trainable references", 15, TEXT, False, 1),
    ])
    add_table(s, Inches(0.55), Inches(3.5), Inches(12.2), [
        ["Target type", "Best fit", "Why"],
        ["Next-day direction (v1)", "BiGRU, TCN", "Sequence classification — but signal was too weak"],
        ["Next-day |return| (J01)", "XGBoost", "Volatility persistence; trees beat deep nets here"],
        ["21d vol regime (J02)", "LSTM", "Longer-horizon regime memory; crisis stress-test critical"],
        ["21d excess vs SPY (J03–J04)", "TBD", "Ranking task — scaffolded for future work"],
    ], col_widths=[0.28, 0.22, 0.50], font_size=12)
    add_text(s, Inches(0.55), Inches(5.8), Inches(12.2), Inches(1.2), [
        bullet("Key research insight: the target matters more than the architecture. "
               "Pivoting from direction to volatility unlocked real predictive skill.", 16, GOOD, True),
    ])

    # 6 — Preprocessing pipeline & training scale
    s = add_slide(prs)
    add_title(s, "Challenges: Preprocessing Pipeline & Training 20+ Models",
              "Building infrastructure to compare architectures fairly at scale")
    add_table(s, Inches(0.55), Inches(1.6), Inches(5.8), [
        ["Pipeline stage", "Output"],
        ["Universe selection", "eligible_symbols.json"],
        ["Feature engineering", "15 features (plus5 profile)"],
        ["Sequence builder", "60-day windows → .npz bundles"],
        ["Scaler + config", "JSON artifacts per job/universe"],
        ["Notebook bake-offs", "1 baseline + 5 candidates each"],
    ], col_widths=[0.45, 0.55], font_size=12)
    add_text(s, Inches(6.7), Inches(1.6), Inches(6.1), Inches(5.2), [
        bullet("Training scale:", 17, ACCENT, True),
        bullet("20+ distinct model runs across jobs and universes", 15, TEXT, False, 1),
        bullet("J01: 12 models (6 stocks + 6 ETFs) — complete", 15, TEXT, False, 1),
        bullet("J02: 9+ models (stocks + market partial) — in progress", 15, TEXT, False, 1),
        bullet("Each run exports checkpoints + predictions JSON for fair comparison", 15, TEXT, False, 1),
        bullet("Engineering overhead:", 17, ACCENT, True),
        bullet("Scaffold generator for consistent notebook structure", 15, TEXT, False, 1),
        bullet("GPU setup on WSL2 (RTX 5090), mixed float16, kernel management", 15, TEXT, False, 1),
        bullet("compare_job_models.py + replay_notebook_outputs.py for reproducibility", 15, TEXT, False, 1),
    ])

    # 7 — Iteration 1 initial plan + failure
    s = add_slide(prs)
    add_title(s, "Initial Plan (Iteration 1): Predict AAPL Direction",
              "Next-day UP/DOWN · 60-day lookback · 7 features · BiGRU and TCN")
    add_text(s, Inches(0.55), Inches(1.7), Inches(12.2), Inches(2.2), [
        bullet("v1 failed in two distinct ways:", 18, ACCENT, True),
        bullet("BiGRU — severe overfitting: train AUC ~0.89 vs validation ~0.49 (gap of 0.40)",
               16, WARN, False, 1),
        bullet("TCN — model collapse: predicted P(UP) ≈ 0.52 for every sample (std 0.004), "
               "equivalent to an 'always UP' strategy", 16, WARN, False, 1),
    ])
    add_table(s, Inches(0.55), Inches(4.0), Inches(12.2), [
        ["Model", "v1 diagnosis", "Val AUC", "Verdict"],
        ["BiGRU", "Memorized training data, zero generalization", "0.493", "Worse than 'always UP'"],
        ["TCN", "Constant output — no sample discrimination", "0.499", "Random ranking"],
    ], col_widths=[0.14, 0.44, 0.14, 0.28], font_size=13)
    add_text(s, Inches(0.55), Inches(5.6), Inches(12.2), Inches(1.2), [
        bullet("Key insight: high accuracy / F1 were illusions of class imbalance (~55% UP days) — "
               "not real predictive skill.", 16, ACCENT, True),
    ])

    # 5 — Iteration 1 v2
    s = add_slide(prs)
    add_title(s, "Iteration 1 v2: Fixes Worked, But the Ceiling Was Low",
              "Training pathologies solved — yet direction remained barely learnable")
    add_table(s, Inches(0.55), Inches(1.7), Inches(12.2), [
        ["Fix", "Change", "Result"],
        ["GRU anti-overfit", "61k → 4.5k params, recurrent dropout, LR 3e-4",
         "Train–val AUC gap 0.40 → 0.005"],
        ["TCN anti-collapse", "Last-timestep readout (was global pooling), AdamW",
         "Val AUC 0.499 → 0.559; pred std 0.004 → 0.051"],
        ["Honest thresholds", "Tuned on validation, applied unchanged to test",
         "No more baseline-mimicking F1"],
    ], col_widths=[0.20, 0.42, 0.38], font_size=13)
    add_text(s, Inches(0.55), Inches(4.4), Inches(12.2), Inches(2.2), [
        bullet("Verdict: even after fixes, best validation AUC was 0.559 — a marginal edge, "
               "not production-ready for directional trading.", 17, WARN, True),
        bullet("Conclusion that drove the pivot: next-day direction is nearly random at daily "
               "horizon. Volatility, however, is persistent and learnable.", 17, GOOD, True),
    ])

    # 6 — The pivot
    s = add_slide(prs)
    add_title(s, "The Pivot: Predict What Is Learnable",
              "ML Iteration 2 — a portfolio of jobs, each mapped to a product feature")
    add_table(s, Inches(0.55), Inches(1.7), Inches(12.2), [
        ["Job", "Prediction target", "Product feature", "Status"],
        ["J01 vol_risk", "Next-day |log return|", "Turbulence alert", "Complete"],
        ["J02 vol_regime", "21-day high-volatility regime", "Market weather", "Active"],
        ["J03 rel_strength", "21d excess return vs SPY", "Stock watchlist", "Planned"],
        ["J04 etf_opportunity", "21d excess return vs SPY (ETFs)", "ETF allocation hints", "Planned"],
    ], col_widths=[0.18, 0.32, 0.28, 0.22], font_size=13)
    add_text(s, Inches(0.55), Inches(4.6), Inches(12.2), Inches(2.0), [
        bullet("Separate models per job — not one mega-network. Shared preprocessing feeds "
               "each job; an orchestration layer merges champion scores.", 16),
        bullet("Every job serves the same product story: risk control and benchmark-relative "
               "selection for novice investors.", 16),
    ])

    # 7 — Methodology
    s = add_slide(prs)
    add_title(s, "Methodology: Bake-Offs With Honest Baselines",
              "1 trainable baseline + 5 architectures per job/universe → champion")
    add_text(s, Inches(0.55), Inches(1.7), Inches(6.1), Inches(4.6), [
        bullet("Candidates per bake-off:", 17, ACCENT, True),
        bullet("Baseline: Ridge / logistic on last-day features", 15, TEXT, False, 1),
        bullet("GRU · TCN · LSTM · 1D-CNN · XGBoost", 15, TEXT, False, 1),
        bullet("Champion rubric:", 17, ACCENT, True),
        bullet("1. Must beat trained baseline on validation", 15, TEXT, False, 1),
        bullet("2. Best COVID-crisis performance", 15, TEXT, False, 1),
        bullet("3. Tie-breakers per job (e.g. top-10% vol AUC)", 15, TEXT, False, 1),
    ])
    add_table(s, Inches(6.9), Inches(1.7), Inches(5.9), [
        ["Split", "Period"],
        ["Train", "≤ 2017-12-31"],
        ["Validation", "2018 – 2019"],
        ["Stress test", "2020 Q1 COVID crash"],
    ], col_widths=[0.4, 0.6], font_size=14)
    add_text(s, Inches(6.9), Inches(3.6), Inches(5.9), Inches(2.6), [
        bullet("Every model exports reproducible artifacts (predictions JSON + checkpoints) "
               "for cross-model comparison and replay.", 15),
        bullet("Naive references also reported: persistence (today's |return| → tomorrow) "
               "and train-mean.", 15),
    ])

    # 8 — Training setup
    s = add_slide(prs)
    add_title(s, "Training Setup")
    add_table(s, Inches(0.55), Inches(1.6), Inches(6.2), [
        ["Setting", "Value"],
        ["Features", "15 engineered (plus5 profile)"],
        ["Lookback", "60 trading days"],
        ["Universe", "dev200 (200 symbols, pooled)"],
        ["fast_dev", "30 epochs (bake-offs)"],
        ["full", "100 epochs (champions)"],
    ], col_widths=[0.35, 0.65], font_size=13)
    add_text(s, Inches(7.1), Inches(1.6), Inches(5.7), Inches(4.8), [
        bullet("Feature groups:", 16, ACCENT, True),
        bullet("Returns (log, absolute, squared)", 14, TEXT, False, 1),
        bullet("Volatility windows (5d, 21d, ratios)", 14, TEXT, False, 1),
        bullet("Volume (log volume, surge)", 14, TEXT, False, 1),
        bullet("Technicals (RSI-14, intraday range)", 14, TEXT, False, 1),
        bullet("Market factors (SPY return, SPY vol)", 14, TEXT, False, 1),
        bullet("Hardware & stack:", 16, ACCENT, True),
        bullet("RTX 5090 GPU on WSL2, TensorFlow 2.21, mixed float16", 14, TEXT, False, 1),
        bullet("XGBoost / HistGBR on CPU with flattened sequences", 14, TEXT, False, 1),
    ])

    # 9 — J01 results
    s = add_slide(prs)
    add_title(s, "Results: J01 Vol/Risk — Complete",
              "Champion: XGBoost in both universes")
    s.shapes.add_picture(str(j01_chart), Inches(0.55), Inches(1.5), width=Inches(12.2))
    add_table(s, Inches(0.55), Inches(5.35), Inches(12.2), [
        ["Universe", "Champion val MAE", "Baseline val MAE", "Beats baseline (val / crisis)"],
        ["ETFs", "0.0043", "0.0055", "Yes / Yes"],
        ["Stocks", "0.0143", "0.0155", "Yes / slightly worse"],
    ], col_widths=[0.18, 0.27, 0.27, 0.28], font_size=13)
    add_text(s, Inches(0.55), Inches(6.7), Inches(12.2), Inches(0.7), [
        bullet("Honest finding: deep sequence models (LSTM 0.0144, GRU 0.0145) barely beat XGB — "
               "architecture complexity did not buy better volatility forecasts.", 14, MUTED),
    ])

    # 10 — J02 results
    s = add_slide(prs)
    add_title(s, "Results: J02 Vol Regime — In Progress",
              "Classifying 21-day high-volatility regimes (macro-F1)")
    s.shapes.add_picture(str(j02_chart), Inches(0.55), Inches(1.5), width=Inches(12.2))
    add_text(s, Inches(0.55), Inches(5.5), Inches(12.2), Inches(1.9), [
        bullet("Stocks (5/6 trained): LSTM leads — val F1 0.802 vs baseline 0.759; crisis F1 0.660.",
               15, GOOD),
        bullet("Market/SPY (4/6 trained): baseline still leads validation (0.660); LSTM best on "
               "crisis (0.354).", 15),
        bullet("Cautionary tale: market TCN scored 0.54 on validation but collapsed to 0.15 on the "
               "COVID window — exactly why every candidate is stress-tested.", 15, WARN),
    ])

    # 14 — What I learned
    s = add_slide(prs)
    add_title(s, "What I Learned",
              "Insights that changed how I think about financial ML")
    add_text(s, Inches(0.55), Inches(1.6), Inches(6.1), Inches(5.4), [
        bullet("Next-day price direction is difficult and unreliable", 17, WARN, True),
        bullet("Daily returns are close to random walk at 1-day horizon", 15, TEXT, False, 1),
        bullet("High accuracy/F1 can be an illusion of class imbalance (~55% UP days)", 15, TEXT, False, 1),
        bullet("Even after fixing overfitting, best val AUC was only 0.559 — not tradeable", 15, TEXT, False, 1),
        bullet("Without strong exogenous signals (news, order flow), direction models fail", 15, TEXT, False, 1),
        bullet("Volatility / risk prediction is more useful", 17, GOOD, True),
        bullet("Volatility clusters and persists — models can beat naive baselines", 15, TEXT, False, 1),
        bullet("Actionable for investors: 'expect turbulence tomorrow' vs 'buy/sell AAPL'", 15, TEXT, False, 1),
        bullet("Better business fit: position sizing, risk alerts, allocation discipline", 15, TEXT, False, 1),
    ])
    add_text(s, Inches(6.8), Inches(1.6), Inches(6.0), Inches(5.4), [
        bullet("Markets behave differently in crisis periods", 17, ACCENT, True),
        bullet("2020 Q1 COVID crash broke patterns learned on 2018–2019 validation", 15, TEXT, False, 1),
        bullet("Models that look strong in calm markets can collapse in crises "
               "(e.g. market TCN: val F1 0.54 → crisis F1 0.15)", 15, WARN, False, 1),
        bullet("Stress-testing is not optional — it must be part of model selection", 15, TEXT, False, 1),
        bullet("Regime detection (J02) directly addresses this structural break", 15, TEXT, False, 1),
        bullet("Broader lessons:", 17, ACCENT, True),
        bullet("Simple baselines (Ridge, persistence) are genuinely hard to beat", 15, TEXT, False, 1),
        bullet("Define success before training: baseline-relative, crisis-tested, product-mapped", 15, GOOD, False, 1),
    ])

    # 15 — Applications & business value
    s = add_slide(prs)
    add_title(s, "Applications & Potential Use",
              "From research project to novice-investor risk intelligence")
    add_table(s, Inches(0.55), Inches(1.6), Inches(12.2), [
        ["Application", "Model layer", "User value"],
        ["Turbulence alerts", "J01 vol/risk", "Warn before high-volatility days — reduce surprise losses"],
        ["Market weather dashboard", "J02 vol/regime", "Shift to defensive posture during crisis-like regimes"],
        ["Stock watchlist", "J03 rel strength", "Highlight names outperforming SPY over ~21 days"],
        ["ETF allocation hints", "J04 ETF opportunity", "Core vs satellite ETF suggestions vs benchmark"],
        ["Portfolio coach", "Rule engine + ML scores", "Position limits, diversification guardrails"],
    ], col_widths=[0.24, 0.26, 0.50], font_size=12)
    add_text(s, Inches(0.55), Inches(4.5), Inches(12.2), Inches(2.5), [
        bullet("Who benefits:", 17, ACCENT, True),
        bullet("Novice investors — calmer decisions, not day-trading on noisy signals", 15, TEXT, False, 1),
        bullet("Portfolio apps / fintech — embed risk scores as a product feature", 15, TEXT, False, 1),
        bullet("Researchers — reproducible bake-off framework for financial time-series ML", 15, TEXT, False, 1),
        bullet("Deployment path:", 17, ACCENT, True),
        bullet("Champion models → orchestration layer → dashboard / API → alerts & watchlists", 15, TEXT, False, 1),
        bullet("GitHub repo ships dev200 artifacts so training can resume on any machine", 15, TEXT, False, 1),
    ])

    # 16 — Future roadmap
    s = add_slide(prs)
    add_title(s, "Future Roadmap & Conclusion")
    add_text(s, Inches(0.55), Inches(1.6), Inches(12.2), Inches(4.2), [
        bullet("Near-term", 18, ACCENT, True),
        bullet("Finish J02 (ETF universe + missing architectures), select regime champions", 15, TEXT, False, 1),
        bullet("Build J03 relative-strength and J04 ETF-opportunity rankers", 15, TEXT, False, 1),
        bullet("Scale-up", 18, ACCENT, True),
        bullet("Orchestration layer: merge champion scores into unified product signals", 15, TEXT, False, 1),
        bullet("Full-universe champion retraining (beyond dev200) + walk-forward validation", 15, TEXT, False, 1),
        bullet("Product delivery", 18, ACCENT, True),
        bullet("Dashboard demo → turbulence alerts, market weather, watchlist, ETF hints", 15, TEXT, False, 1),
    ])
    add_text(s, Inches(0.55), Inches(6.0), Inches(12.2), Inches(1.1), [
        bullet("Takeaway: pivoting from \u201cpredict direction\u201d to \u201cpredict risk\u201d "
               "turned a failing model into a working model portfolio.", 18, GOOD, True),
    ])

    out = OUT_DIR / "stock_prediction_project.pptx"
    prs.save(out)
    return out


if __name__ == "__main__":
    path = build()
    print(f"Saved: {path}")
