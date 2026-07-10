"""Plotly figure builders with a consistent visual language."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, roc_curve

ARCH_COLORS = {
    "baseline": "#8B93A7",
    "gru": "#4F8BF9",
    "tcn": "#F5A623",
    "lstm": "#2ECC8F",
    "cnn1d": "#B07CF7",
    "xgb": "#FF6B6B",
}
ACCENT = "#4F8BF9"
GRID = "rgba(139, 147, 167, 0.15)"

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    margin=dict(l=10, r=10, t=48, b=10),
    hoverlabel=dict(font_size=12),
)


def _base(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(**LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def arch_color(arch: str) -> str:
    return ARCH_COLORS.get(str(arch), "#8B93A7")


def leaderboard_bars(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    *,
    higher_is_better: bool,
    fmt: str = ".4f",
    champion: str | None = None,
) -> go.Figure:
    """Horizontal bars per architecture with the baseline as a reference line."""
    sub = df.dropna(subset=[metric_col]).copy()
    sub = sub.sort_values(metric_col, ascending=not higher_is_better)
    baseline_val = None
    base_rows = sub[sub["is_baseline"]]
    if not base_rows.empty:
        baseline_val = float(base_rows.iloc[0][metric_col])

    labels, colors, widths = [], [], []
    for _, r in sub.iterrows():
        arch = str(r["architecture"])
        name = arch.upper() if arch != "baseline" else "Baseline"
        if champion and arch == champion:
            name = f"👑 {name}"
        labels.append(name)
        colors.append(arch_color(arch))
        widths.append(float(r[metric_col]))

    fig = go.Figure(
        go.Bar(
            x=widths,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:{fmt}}" for v in widths],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>",
        )
    )
    if baseline_val is not None:
        fig.add_vline(
            x=baseline_val,
            line=dict(color="#E4E7F0", dash="dash", width=1.4),
            annotation_text="baseline",
            annotation_font=dict(size=11, color="#E4E7F0"),
        )
    fig.update_yaxes(autorange="reversed")
    arrow = "higher is better →" if higher_is_better else "← lower is better"
    fig.update_xaxes(
        title=dict(text=arrow, font=dict(size=11, color="#8B93A7")),
        range=[0, max(widths) * 1.22],  # room for outside value labels
    )
    return _base(fig, title, height=320)


def training_curves(summaries: list[dict], metric_label: str) -> go.Figure:
    """Validation-loss curves overlaid across architectures."""
    fig = go.Figure()
    for s in summaries:
        hist = s.get("history") or []
        if len(hist) < 2:
            continue
        arch = s.get("architecture", "?")
        epochs = [h["epoch"] for h in hist]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[h["val_loss"] for h in hist],
                name=str(arch).upper(),
                mode="lines",
                line=dict(color=arch_color(arch), width=2.2),
                hovertemplate=f"{str(arch).upper()} · epoch %{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title=f"Validation {metric_label}")
    ys = [y for tr in fig.data for y in tr.y if y and y > 0]
    if ys and max(ys) / min(ys) > 20:  # curves span magnitudes → log scale
        fig.update_yaxes(type="log", title=f"Validation {metric_label} (log)")
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    return _base(fig, "", height=360)


def pred_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, max_points: int = 6000
) -> go.Figure:
    n = len(y_true)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    lim = float(max(y_true.max(), y_pred.max())) * 1.05
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(size=4, color=ACCENT, opacity=0.35),
            hovertemplate="actual %{x:.4f} · predicted %{y:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, lim],
            y=[0, lim],
            mode="lines",
            line=dict(color="#FF6B6B", dash="dash", width=1.4),
            name="perfect prediction",
        )
    )
    fig.update_xaxes(title="Actual |log return|", range=[0, lim])
    fig.update_yaxes(title="Predicted |log return|", range=[0, lim])
    fig.update_layout(legend=dict(orientation="h", y=1.1), showlegend=True)
    return _base(fig, title, height=400)


def residual_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    resid = y_pred - y_true
    fig = go.Figure(
        go.Histogram(
            x=resid,
            nbinsx=60,
            marker=dict(color=ACCENT, opacity=0.75),
            hovertemplate="residual %{x:.4f}: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#FF6B6B", dash="dash", width=1.4))
    fig.update_xaxes(title="Residual (predicted − actual)")
    fig.update_yaxes(title="Count")
    return _base(fig, title, height=400)


def calibration_by_decile(
    y_true: np.ndarray, y_pred: np.ndarray, title: str
) -> go.Figure:
    """Mean predicted vs mean actual within actual-volatility deciles."""
    order = np.argsort(y_true)
    deciles = np.array_split(order, 10)
    mean_true = [float(y_true[d].mean()) for d in deciles]
    mean_pred = [float(y_pred[d].mean()) for d in deciles]
    x = list(range(1, 11))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=mean_true, name="Actual", mode="lines+markers",
            line=dict(color="#E4E7F0", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=mean_pred, name="Predicted", mode="lines+markers",
            line=dict(color=ACCENT, width=2.2),
        )
    )
    fig.update_xaxes(title="Actual-volatility decile (1 = calmest, 10 = wildest)", dtick=1)
    fig.update_yaxes(title="Mean |log return|")
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    return _base(fig, title, height=400)


def confusion_heatmap(
    y_true: np.ndarray, y_pred: np.ndarray, title: str
) -> go.Figure:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    pct = cm / cm.sum() * 100
    labels = ["Low vol", "High vol"]
    text = [
        [f"{cm[i][j]:,}<br>{pct[i][j]:.1f}%" for j in range(2)] for i in range(2)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="#E4E7F0", size=14),
            colorscale=[[0, "rgba(79,139,249,0.08)"], [1, "rgba(79,139,249,0.9)"]],
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _base(fig, title, height=360)


def roc_chart(y_true: np.ndarray, y_prob: np.ndarray, title: str) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob)
    score = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {score:.3f})",
            line=dict(color=ACCENT, width=2.4),
            fill="tozeroy", fillcolor="rgba(79,139,249,0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="chance",
            line=dict(color="#8B93A7", dash="dash", width=1.2),
        )
    )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    return _base(fig, title, height=400)


def probability_by_class(
    y_true: np.ndarray, y_prob: np.ndarray, title: str
) -> go.Figure:
    fig = go.Figure()
    for cls, name, color in ((0, "Actual low vol", "#2ECC8F"), (1, "Actual high vol", "#FF6B6B")):
        mask = y_true.astype(int) == cls
        fig.add_trace(
            go.Histogram(
                x=y_prob[mask],
                name=name,
                nbinsx=40,
                opacity=0.65,
                marker=dict(color=color),
                histnorm="probability",
            )
        )
    fig.add_vline(x=0.5, line=dict(color="#E4E7F0", dash="dash", width=1.2))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(title="Predicted P(high-vol regime)", range=[0, 1])
    fig.update_yaxes(title="Share of split")
    return _base(fig, title, height=400)


def regime_timeline(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
) -> go.Figure:
    """P(high vol) over time with actual high-vol periods shaded."""
    fig = go.Figure()

    in_regime = y_true.astype(int) == 1
    start = None
    for i, flag in enumerate(in_regime):
        if flag and start is None:
            start = dates[i]
        if start is not None and (not flag or i == len(in_regime) - 1):
            end = dates[i] if flag else dates[max(i - 1, 0)]
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(255,107,107,0.14)", line_width=0,
            )
            start = None

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_prob,
            mode="lines",
            name="P(high-vol regime)",
            line=dict(color=ACCENT, width=2.2),
            hovertemplate="%{x|%b %d, %Y}: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line=dict(color="#E4E7F0", dash="dash", width=1.2))
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="rgba(255,107,107,0.5)", symbol="square", size=10),
            name="actual high-vol period",
        )
    )
    fig.update_yaxes(title="P(high-vol regime)", range=[-0.02, 1.05])
    fig.update_xaxes(title="Approximate trading date")
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return _base(fig, title, height=420)


def stress_scatter(
    df: pd.DataFrame,
    val_col: str,
    crisis_col: str,
    title: str,
    *,
    higher_is_better: bool,
    fmt: str = ".4f",
) -> go.Figure:
    """Validation vs COVID-crisis metric — the generalization-gap view."""
    sub = df.dropna(subset=[val_col, crisis_col])
    fig = go.Figure()
    for universe, symbol in (("stocks", "circle"), ("etfs", "diamond"), ("market", "square")):
        u = sub[sub["universe"] == universe]
        if u.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=u[val_col],
                y=u[crisis_col],
                mode="markers+text",
                name=universe,
                text=[str(a).upper() if a != "baseline" else "BASE" for a in u["architecture"]],
                textposition="top center",
                textfont=dict(size=10, color="#8B93A7"),
                marker=dict(
                    size=13,
                    symbol=symbol,
                    color=[arch_color(a) for a in u["architecture"]],
                    line=dict(width=1, color="rgba(228,231,240,0.4)"),
                ),
                hovertemplate=(
                    "%{text} (" + universe + ")<br>val %{x:" + fmt + "}"
                    "<br>crisis %{y:" + fmt + "}<extra></extra>"
                ),
            )
        )
    lo = min(float(sub[val_col].min()), float(sub[crisis_col].min()))
    hi = max(float(sub[val_col].max()), float(sub[crisis_col].max()))
    pad = (hi - lo) * 0.08 or 0.01
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
            line=dict(color="#8B93A7", dash="dot", width=1),
            name="no degradation", hoverinfo="skip",
        )
    )
    better = "top-right is better" if higher_is_better else "bottom-left is better"
    fig.update_xaxes(title=f"Validation 2018-19 ({better})")
    fig.update_yaxes(title="COVID 2020 Q1 stress")
    fig.update_layout(legend=dict(orientation="h", y=-0.22))
    return _base(fig, title, height=460)
