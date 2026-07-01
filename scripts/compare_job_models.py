#!/usr/bin/env python3
"""Compare all trained models in a job folder against the trainable baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_iteration2_jobs import ML_DIR, JOB_PRIMARY_METRIC


def load_metrics(models_dir: Path, pattern: str, primary_metric: str) -> list[dict]:
    rows = []
    for path in sorted(models_dir.glob(pattern)):
        with open(path) as f:
            payload = json.load(f)
        rows.append(
            {
                "artifact": path.name,
                "architecture": payload.get("architecture"),
                "is_baseline": payload.get("is_baseline", False),
                "val_mae": payload.get("val_mae"),
                "crisis_mae": payload.get("crisis_mae"),
                "val_macro_f1": payload.get("val_macro_f1"),
                "crisis_macro_f1": payload.get("crisis_macro_f1"),
                "val_rmse": payload.get("val_rmse"),
                "primary_metric_value": payload.get(primary_metric),
                "baseline_comparison": payload.get("baseline_comparison"),
            }
        )
    return rows


def compare_job(job_slug: str, universe: str, dataset_tag: str = "dev200") -> None:
    models_dir = Path(__file__).resolve().parents[1] / "data" / "models"
    glob = f"*_{job_slug}_{universe}_{dataset_tag}_tf_predictions.json"
    metric = JOB_PRIMARY_METRIC.get(job_slug, "val_mae")
    rows = load_metrics(models_dir, glob, metric)
    if not rows:
        print(f"No prediction artifacts found for pattern: {glob}")
        return

    baseline = next((r for r in rows if r["is_baseline"]), None)

    print(f"Job: {job_slug} | Universe: {universe} | Metric: {metric}")
    print("-" * 88)
    if metric == "macro_f1":
        print(
            f"{'architecture':<12} {'val_f1':>10} {'crisis_f1':>12} "
            f"{'vs baseline val':>16} {'vs baseline crisis':>18}"
        )
        print("-" * 88)
        baseline_val = baseline["val_macro_f1"] if baseline else None
        baseline_crisis = baseline["crisis_macro_f1"] if baseline else None
        sort_key = lambda r: (-(r.get("val_macro_f1") or -1), not r["is_baseline"])
        for row in sorted(rows, key=sort_key):
            arch = row["architecture"] or "?"
            val = row["val_macro_f1"]
            crisis = row["crisis_macro_f1"]
            if baseline and not row["is_baseline"]:
                dv = val - baseline_val if val is not None and baseline_val is not None else None
                dc = crisis - baseline_crisis if crisis is not None and baseline_crisis is not None else None
                vs_v = f"{dv:+.4f}" if dv is not None else "n/a"
                vs_c = f"{dc:+.4f}" if dc is not None else "n/a"
            elif row["is_baseline"]:
                vs_v = "BASELINE"
                vs_c = "BASELINE"
            else:
                vs_v = "no baseline"
                vs_c = "no baseline"
            tag = " *" if row["is_baseline"] else ""
            print(f"{arch + tag:<12} {val:>10.4f} {crisis:>12.4f} {vs_v:>16} {vs_c:>18}")
        print("-" * 88)
        print("Positive delta = beats baseline. Train 00a_baseline_* before comparing.")
        return

    print(f"{'architecture':<12} {'val_mae':>10} {'crisis_mae':>12} {'vs baseline val':>16} {'vs baseline crisis':>18}")
    print("-" * 88)

    baseline_val = baseline["val_mae"] if baseline else None
    baseline_crisis = baseline["crisis_mae"] if baseline else None

    for row in sorted(rows, key=lambda r: (not r["is_baseline"], r.get("val_mae") or 999)):
        arch = row["architecture"] or "?"
        val = row["val_mae"]
        crisis = row["crisis_mae"]
        if baseline and not row["is_baseline"]:
            dv = val - baseline_val if val is not None and baseline_val is not None else None
            dc = crisis - baseline_crisis if crisis is not None and baseline_crisis is not None else None
            vs_v = f"{dv:+.6f}" if dv is not None else "n/a"
            vs_c = f"{dc:+.6f}" if dc is not None else "n/a"
        elif row["is_baseline"]:
            vs_v = "BASELINE"
            vs_c = "BASELINE"
        else:
            vs_v = "no baseline"
            vs_c = "no baseline"
        tag = " *" if row["is_baseline"] else ""
        print(f"{arch + tag:<12} {val:>10.6f} {crisis:>12.6f} {vs_v:>16} {vs_c:>18}")

    print("-" * 88)
    print("Negative delta = beats baseline. Train 00a_baseline_* before comparing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare job models vs baseline")
    parser.add_argument("--job-slug", default="vol_risk")
    parser.add_argument("--universe", default="stocks")
    parser.add_argument("--dataset-tag", default="dev200")
    args = parser.parse_args()
    compare_job(args.job_slug, args.universe, args.dataset_tag)


if __name__ == "__main__":
    main()
