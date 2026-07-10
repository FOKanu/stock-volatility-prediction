"""Discover and load ML Iteration 2 artifacts for the dashboard.

Reads only saved outputs: data/models/*_tf_predictions.json and
ML_Iteration_2/J*/champions.json. Never touches notebooks or training code.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"
ML_DIR = ROOT / "ML_Iteration_2"

sys.path.insert(0, str(ROOT / "scripts"))
from ml_iteration2_jobs import (  # noqa: E402
    ARCHITECTURE_LABELS,
    JOB_PRIMARY_METRIC,
    JOB_TASK_KIND,
)

JOB_TITLES = {
    "vol_risk": "J01 · Vol / Risk Regression",
    "vol_regime": "J02 · Vol Regime Classification",
    "rel_strength": "J03 · Relative Strength vs SPY",
    "etf_opportunity": "J04 · ETF Opportunity Ranker",
}

JOB_TARGETS = {
    "vol_risk": "Next-day |log return|",
    "vol_regime": "21-day high-volatility regime (binary)",
    "rel_strength": "21-day excess return vs SPY",
    "etf_opportunity": "21-day ETF excess return vs SPY",
}

ARCH_ORDER = ["baseline", "gru", "tcn", "lstm", "cnn1d", "xgb"]

ARTIFACT_RE = re.compile(
    r"^(?P<arch>baseline|gru|tcn|lstm|cnn1d|xgb)_"
    r"(?P<job_slug>vol_risk|vol_regime|rel_strength|etf_opportunity)_"
    r"(?P<universe>stocks|etfs|market)_"
    r"(?P<dataset_tag>[a-z0-9]+)_tf_predictions\.json$"
)


def discover_artifacts(dataset_tag: str = "dev200") -> pd.DataFrame:
    """One row per trained model artifact found on disk."""
    rows = []
    for path in sorted(MODELS_DIR.glob("*_tf_predictions.json")):
        m = ARTIFACT_RE.match(path.name)
        if not m or m.group("dataset_tag") != dataset_tag:
            continue
        rows.append({**m.groupdict(), "path": str(path)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_summary(path: str, mtime: float) -> dict:
    """Metrics + history without the big prediction arrays."""
    with open(path) as f:
        payload = json.load(f)
    payload.pop("predictions", None)
    return payload


@st.cache_data(show_spinner="Loading prediction arrays…", max_entries=6)
def load_predictions(path: str, mtime: float) -> dict:
    """Full val/test arrays as numpy, keyed by split."""
    with open(path) as f:
        payload = json.load(f)
    out = {}
    for split, arrs in payload.get("predictions", {}).items():
        out[split] = {k: np.asarray(v, dtype=np.float32) for k, v in arrs.items()}
    return out


def summary_for(path: Path | str) -> dict:
    p = Path(path)
    return load_summary(str(p), p.stat().st_mtime)


def predictions_for(path: Path | str) -> dict:
    p = Path(path)
    return load_predictions(str(p), p.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_champions() -> dict:
    out = {}
    for path in sorted(ML_DIR.glob("J*_*/champions.json")):
        with open(path) as f:
            data = json.load(f)
        out[data["job_slug"]] = data
    return out


def metrics_table(dataset_tag: str = "dev200") -> pd.DataFrame:
    """Leaderboard-ready frame: one row per artifact with headline metrics."""
    artifacts = discover_artifacts(dataset_tag)
    if artifacts.empty:
        return artifacts
    rows = []
    for _, art in artifacts.iterrows():
        s = summary_for(art["path"])
        task = JOB_TASK_KIND.get(art["job_slug"], "regression")
        bc = s.get("baseline_comparison") or {}
        ref = s.get("baselines") or s.get("reference_baselines") or {}
        rows.append(
            {
                "job_slug": art["job_slug"],
                "universe": art["universe"],
                "architecture": art["arch"],
                "is_baseline": bool(s.get("is_baseline", False)),
                "task": task,
                "val_mae": s.get("val_mae"),
                "crisis_mae": s.get("crisis_mae"),
                "val_rmse": s.get("val_rmse"),
                "val_macro_f1": s.get("val_macro_f1"),
                "crisis_macro_f1": s.get("crisis_macro_f1"),
                "val_accuracy": s.get("val_accuracy"),
                "crisis_accuracy": s.get("crisis_accuracy"),
                "beats_baseline_val": bc.get("beats_baseline_val_mae")
                if task == "regression"
                else bc.get("beats_baseline_val_macro_f1"),
                "beats_baseline_crisis": bc.get("beats_baseline_crisis_mae")
                if task == "regression"
                else bc.get("beats_baseline_crisis_macro_f1"),
                "persistence_mae_val": ref.get("persistence_mae_val"),
                "persistence_mae_test": ref.get("persistence_mae_test"),
                "n_symbols": s.get("n_symbols"),
                "best_epoch": s.get("best_epoch"),
                "n_epochs": len(s.get("history") or []),
                "trained_at": s.get("trained_at"),
                "path": art["path"],
            }
        )
    df = pd.DataFrame(rows)
    df["architecture"] = pd.Categorical(df["architecture"], ARCH_ORDER, ordered=True)
    return df.sort_values(["job_slug", "universe", "architecture"]).reset_index(drop=True)


def primary_metric(job_slug: str) -> tuple[str, str, bool]:
    """(val column, crisis column, higher_is_better) for a job."""
    if JOB_TASK_KIND.get(job_slug) == "classification":
        return "val_macro_f1", "crisis_macro_f1", True
    return "val_mae", "crisis_mae", False


def current_leader(df: pd.DataFrame, job_slug: str, universe: str) -> pd.Series | None:
    """Best non-baseline model on the job's validation metric."""
    val_col, _, higher = primary_metric(job_slug)
    sub = df[
        (df["job_slug"] == job_slug)
        & (df["universe"] == universe)
        & (~df["is_baseline"])
        & df[val_col].notna()
    ]
    if sub.empty:
        return None
    return sub.loc[sub[val_col].idxmax() if higher else sub[val_col].idxmin()]


def market_dates(n_val: int, n_test: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Approximate trading-day axes for the market (SPY) universe.

    Validation ends 2019-12-31; the COVID stress window starts 2020-01-02.
    Exchange holidays are not modelled, so dates are approximate.
    """
    val = pd.bdate_range(end="2019-12-31", periods=n_val)
    test = pd.bdate_range(start="2020-01-02", periods=n_test)
    return val, test


__all__ = [
    "ARCH_ORDER",
    "ARCHITECTURE_LABELS",
    "JOB_PRIMARY_METRIC",
    "JOB_TASK_KIND",
    "JOB_TARGETS",
    "JOB_TITLES",
    "ROOT",
    "current_leader",
    "discover_artifacts",
    "load_champions",
    "market_dates",
    "metrics_table",
    "predictions_for",
    "primary_metric",
    "summary_for",
]
