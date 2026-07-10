"""Maximum Mean Discrepancy (MMD) drift monitor for feature distributions.

Compares a recent window of feature vectors against a training-baseline sample.
When the MMD statistic exceeds a threshold calibrated on in-distribution data,
the incoming distribution has shifted — the product should treat model outputs
with caution (widen conformal bands, prefer conservative coach rules).

Pure NumPy; cheap enough to run as a daily production health-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _median_heuristic_bandwidth(X: np.ndarray, Y: np.ndarray, max_points: int = 500, seed: int = 42) -> float:
    """Median pairwise distance across a subsample of both sets (standard heuristic)."""
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y])
    if len(Z) > max_points:
        Z = Z[rng.choice(len(Z), max_points, replace=False)]
    sq = np.sum(Z**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    d2 = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(np.maximum(d2, 0.0)))
    return np.sqrt(med / 2) if med > 0 else 1.0


def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = None) -> float:
    """Unbiased MMD^2 estimate with an RBF kernel between samples X and Y."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if bandwidth is None:
        bandwidth = _median_heuristic_bandwidth(X, Y)
    gamma = 1.0 / (2 * bandwidth**2)

    def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq_a = np.sum(A**2, axis=1)
        sq_b = np.sum(B**2, axis=1)
        return np.exp(-gamma * np.maximum(sq_a[:, None] + sq_b[None, :] - 2 * A @ B.T, 0.0))

    m, n = len(X), len(Y)
    Kxx = kernel(X, X)
    Kyy = kernel(Y, Y)
    Kxy = kernel(X, Y)
    # Unbiased: exclude diagonal terms.
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sum_xx + sum_yy - 2 * Kxy.mean())


@dataclass
class DriftReport:
    mmd2: float
    threshold: float
    drifted: bool
    bandwidth: float
    n_baseline: int
    n_recent: int


class MMDDriftMonitor:
    """Calibrate an MMD threshold on in-distribution splits, then score new windows.

    Usage:
        monitor = MMDDriftMonitor(train_features)          # (n, d) last-day features
        monitor.calibrate(n_permutations=100, quantile=0.99)
        report = monitor.check(recent_features)            # (m, d) recent window
    """

    def __init__(
        self,
        baseline: np.ndarray,
        sample_size: int = 500,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        baseline = np.asarray(baseline, dtype=np.float64)
        if len(baseline) > sample_size * 4:
            idx = self.rng.choice(len(baseline), sample_size * 4, replace=False)
            baseline = baseline[idx]
        self.baseline = baseline
        self.sample_size = sample_size
        self.bandwidth = _median_heuristic_bandwidth(baseline, baseline)
        self.threshold: float | None = None

    def calibrate(self, n_permutations: int = 100, quantile: float = 0.99) -> float:
        """Null distribution: MMD between random splits of the baseline itself."""
        stats = []
        n = len(self.baseline)
        half = min(self.sample_size, n // 2)
        for _ in range(n_permutations):
            idx = self.rng.permutation(n)
            a = self.baseline[idx[:half]]
            b = self.baseline[idx[half : 2 * half]]
            stats.append(mmd_rbf(a, b, bandwidth=self.bandwidth))
        self.threshold = float(np.quantile(stats, quantile))
        return self.threshold

    def check(self, recent: np.ndarray) -> DriftReport:
        if self.threshold is None:
            self.calibrate()
        recent = np.asarray(recent, dtype=np.float64)
        if len(recent) > self.sample_size:
            idx = self.rng.choice(len(recent), self.sample_size, replace=False)
            recent = recent[idx]
        base = self.baseline
        if len(base) > self.sample_size:
            idx = self.rng.choice(len(base), self.sample_size, replace=False)
            base = base[idx]
        stat = mmd_rbf(base, recent, bandwidth=self.bandwidth)
        return DriftReport(
            mmd2=stat,
            threshold=self.threshold,
            drifted=stat > self.threshold,
            bandwidth=self.bandwidth,
            n_baseline=len(base),
            n_recent=len(recent),
        )
