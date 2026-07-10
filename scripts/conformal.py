"""Split-conformal prediction intervals for volatility forecasts (SPCI-style).

Works post-hoc on saved ``*_predictions.json`` artifacts: the validation
partition (2018-2019) is the calibration set, the Q1-2020 COVID window is the
test set. Financial series violate exchangeability, so quantiles are estimated
with a non-overlapping block bootstrap over the calibration sequence (the SPCI
recipe) instead of a plain pooled quantile.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ConformalResult:
    alpha: float
    target_coverage: float
    q_lower: float
    q_upper: float
    empirical_coverage: float
    mean_interval_width: float
    undercoverage_rate: float  # share of test points below the lower bound (missed risk)
    n_calibration: int
    n_test: int
    n_blocks: int
    block_size: int


def _block_bootstrap_quantile(
    scores: np.ndarray,
    q: float,
    block_size: int,
    n_boot: int,
    seed: int,
) -> float:
    """Quantile of non-conformity scores under non-overlapping block resampling."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    n_blocks = max(n // block_size, 1)
    blocks = scores[: n_blocks * block_size].reshape(n_blocks, block_size)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sample = blocks[idx].ravel()
        estimates[b] = np.quantile(sample, q)
    # Conservative choice: mean of bootstrap quantile estimates.
    return float(estimates.mean())


def conformal_intervals(
    y_cal_true: np.ndarray,
    y_cal_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    alpha: float = 0.1,
    block_size: int = 21,
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, ConformalResult]:
    """Asymmetric split-conformal intervals with block-bootstrap calibration.

    Residuals are signed (pred - true), and lower/upper quantiles are calibrated
    separately: volatility under-prediction (the risky side for a risk product)
    gets its own tail estimate rather than assuming symmetric errors.

    Returns (lower_bounds, upper_bounds, ConformalResult).
    """
    y_cal_true = np.asarray(y_cal_true, dtype=np.float64)
    y_cal_pred = np.asarray(y_cal_pred, dtype=np.float64)
    y_test_true = np.asarray(y_test_true, dtype=np.float64)
    y_test_pred = np.asarray(y_test_pred, dtype=np.float64)

    residuals = y_cal_true - y_cal_pred  # positive => model underpredicted vol

    # Finite-sample corrected tail levels for a (1 - alpha) two-sided interval.
    n = len(residuals)
    lo_level = alpha / 2
    hi_level = 1 - alpha / 2
    correction = min((n + 1) / n, 1.0 + 1e-9) if n else 1.0
    hi_level = min(hi_level * correction, 1.0)

    q_lower = _block_bootstrap_quantile(residuals, lo_level, block_size, n_boot, seed)
    q_upper = _block_bootstrap_quantile(residuals, hi_level, block_size, n_boot, seed + 1)

    lower = np.maximum(y_test_pred + q_lower, 0.0)  # vol targets are non-negative
    upper = y_test_pred + q_upper

    covered = (y_test_true >= lower) & (y_test_true <= upper)
    under = y_test_true > upper  # true vol above upper bound: risk was underestimated

    n_blocks = max(len(residuals) // block_size, 1)
    result = ConformalResult(
        alpha=alpha,
        target_coverage=1 - alpha,
        q_lower=float(q_lower),
        q_upper=float(q_upper),
        empirical_coverage=float(covered.mean()),
        mean_interval_width=float((upper - lower).mean()),
        undercoverage_rate=float(under.mean()),
        n_calibration=int(n),
        n_test=int(len(y_test_true)),
        n_blocks=int(n_blocks),
        block_size=int(block_size),
    )
    return lower, upper, result


def load_predictions_payload(models_dir: Path, artifact_name: str) -> dict:
    with open(models_dir / artifact_name) as f:
        return json.load(f)


def conformal_from_artifact(
    models_dir: Path,
    artifact_name: str,
    alpha: float = 0.1,
    block_size: int = 21,
    n_boot: int = 200,
    seed: int = 42,
) -> dict:
    """Run conformal calibration on a saved predictions JSON. Returns a payload
    with intervals + coverage diagnostics, ready to be written next to the model
    artifacts."""
    payload = load_predictions_payload(models_dir, artifact_name)
    preds = payload["predictions"]
    y_cal_true = np.array(preds["val"]["y_true"])
    y_cal_pred = np.array(preds["val"]["y_pred"])
    y_test_true = np.array(preds["test"]["y_true"])
    y_test_pred = np.array(preds["test"]["y_pred"])

    lower, upper, result = conformal_intervals(
        y_cal_true,
        y_cal_pred,
        y_test_true,
        y_test_pred,
        alpha=alpha,
        block_size=block_size,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "source_artifact": artifact_name,
        "job_slug": payload.get("job_slug"),
        "universe": payload.get("universe"),
        "architecture": payload.get("architecture"),
        "dataset_tag": payload.get("dataset_tag"),
        "conformal": asdict(result),
        "test_intervals": {
            "y_true": y_test_true.tolist(),
            "y_pred": y_test_pred.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
    }
