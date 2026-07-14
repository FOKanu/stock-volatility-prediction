"""ML Iteration 2 job registry: folders, naming, and artifact conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ML_Iteration_2"

ARCHITECTURES: dict[str, str] = {
    "a": "gru",
    "b": "tcn",
    "c": "lstm",
    "d": "cnn1d",
    "e": "xgb",
}

# Extra candidates only available for regression jobs (no classifier builder).
REGRESSION_ONLY_ARCHITECTURES: dict[str, str] = {
    "f": "garchgru",
}

ARCHITECTURE_LABELS: dict[str, str] = {
    "baseline": "Ridge on last-day features (trainable linear baseline)",
    "gru": "2-layer GRU (64 units, dropout 0.2)",
    "tcn": "TCN (dilated causal conv blocks)",
    "lstm": "2-layer LSTM (64 units, dropout 0.2)",
    "cnn1d": "1D CNN (Conv1D stack + GAP)",
    "xgb": "Histogram gradient boosting on flattened sequences",
    "garchgru": "GARCH(1,1)-GRU hybrid cell (embedded conditional-variance recursion)",
}

# Primary metric used to compare models against the baseline within a job.
JOB_TASK_KIND: dict[str, str] = {
    "vol_risk": "regression",
    "vol_regime": "classification",
    "vol_regime3": "classification",
    "rel_strength": "regression",
    "etf_opportunity": "regression",
}

JOB_PRIMARY_METRIC: dict[str, str] = {
    "vol_risk": "val_mae",
    "vol_regime": "macro_f1",
    "vol_regime3": "macro_f1",
    "rel_strength": "val_mae",
    "etf_opportunity": "val_mae",
}

JOB_N_CLASSES: dict[str, int] = {"vol_regime3": 3}
JOB_CLASS_LABELS: dict[str, list[str]] = {
    "vol_regime": ["low vol", "high vol"],
    "vol_regime3": ["bull", "bear", "crisis"],
}


def job_n_classes(job_slug: str) -> int:
    return JOB_N_CLASSES.get(job_slug, 2)


@dataclass(frozen=True)
class NotebookSpec:
    job_id: str
    job_slug: str
    job_title: str
    job_num: str
    arch_letter: str
    architecture: str
    universe: str
    asset_type: str
    phase: str
    status: str  # "active" | "scaffold"
    target_type: str
    primary_metric: str
    is_baseline: bool = False


def baseline_notebook_filename(spec: NotebookSpec) -> str:
    return f"00a_baseline_{spec.job_slug}_{spec.universe}.ipynb"


def notebook_filename(spec: NotebookSpec) -> str:
    if spec.is_baseline:
        return baseline_notebook_filename(spec)
    return (
        f"{spec.job_num}{spec.arch_letter}_{spec.architecture}_"
        f"{spec.job_slug}_{spec.universe}.ipynb"
    )


def notebook_path(spec: NotebookSpec) -> Path:
    folder = ML_DIR / f"J{spec.job_id}_{spec.job_slug}" / spec.universe
    return folder / notebook_filename(spec)


def artifact_stem(spec: NotebookSpec, model_tag: str = "") -> str:
    """e.g. gru_vol_risk_stocks_dev200_tf"""
    tag = model_tag or ""
    return f"{spec.architecture}_{spec.job_slug}_{spec.universe}{tag}_tf"


def universe_prefix(universe: str) -> str:
    if universe == "market":
        return "spy"
    if universe == "etfs":
        return "etfs_universe"
    return "stocks_universe"


def target_artifact_suffix(target_type: str) -> str:
    """Non-default targets get a filename suffix so J01 artifacts are not overwritten."""
    return "" if target_type == "abs_return" else f"_{target_type}"


def sequences_names(
    universe: str, dataset_tag: str, target_type: str = "abs_return"
) -> tuple[str, str, str]:
    prefix = universe_prefix(universe)
    target_suffix = target_artifact_suffix(target_type)
    if dataset_tag == "full":
        stem = f"{prefix}{target_suffix}"
        return (
            f"{stem}_sequences_plus5.npz",
            f"pipeline_config_{stem}_plus5.json",
            f"scaler_{stem}_plus5.json",
        )
    stem = f"{prefix}_{dataset_tag}{target_suffix}"
    return (
        f"{stem}_sequences_plus5.npz",
        f"pipeline_config_{stem}_plus5.json",
        f"scaler_{stem}_plus5.json",
    )


def all_notebook_specs() -> list[NotebookSpec]:
    specs: list[NotebookSpec] = []

    def add_job(
        job_id: str,
        job_slug: str,
        job_title: str,
        job_num: str,
        universes: list[str],
        asset_types: dict[str, str],
        phase: str,
        status: str,
        target_type: str,
        primary_metric: str,
    ) -> None:
        arch_map = dict(ARCHITECTURES)
        # GARCH hybrids only make sense for volatility regression targets.
        if job_slug == "vol_risk":
            arch_map.update(REGRESSION_ONLY_ARCHITECTURES)
        for letter, arch in arch_map.items():
            for universe in universes:
                specs.append(
                    NotebookSpec(
                        job_id=job_id,
                        job_slug=job_slug,
                        job_title=job_title,
                        job_num=job_num,
                        arch_letter=letter,
                        architecture=arch,
                        universe=universe,
                        asset_type=asset_types[universe],
                        phase=phase,
                        status=status,
                        target_type=target_type,
                        primary_metric=primary_metric,
                    )
                )

    add_job(
        "01",
        "vol_risk",
        "Vol / Risk Regression",
        "01",
        ["stocks", "etfs"],
        {"stocks": "stock", "etfs": "etf"},
        "P1",
        "active",
        "abs_return",
        "val_mae",
    )
    add_job(
        "02",
        "vol_regime",
        "Vol Regime Classification",
        "02",
        ["stocks", "etfs", "market"],
        {"stocks": "stock", "etfs": "etf", "market": "etf"},
        "P3",
        "active",
        "vol_regime_21",
        "macro_f1",
    )
    add_job(
        "02v2",
        "vol_regime3",
        "3-Class Vol Regime (bull/bear/crisis)",
        "02",
        ["stocks", "etfs", "market"],
        {"stocks": "stock", "etfs": "etf", "market": "etf"},
        "P3",
        "active",
        "vol_regime3_21",
        "macro_f1",
    )
    add_job(
        "03",
        "rel_strength",
        "Relative Strength vs SPY",
        "03",
        ["stocks"],
        {"stocks": "stock"},
        "P4",
        "scaffold",
        "excess_return_21",
        "spearman_excess",
    )
    add_job(
        "04",
        "etf_opportunity",
        "ETF Opportunity Ranker",
        "04",
        ["etfs"],
        {"etfs": "etf"},
        "P5",
        "scaffold",
        "excess_return_21",
        "spearman_excess",
    )
    return specs


def all_baseline_specs() -> list[NotebookSpec]:
    """One trainable baseline notebook per job × universe (00a_baseline_*)."""
    baselines: list[NotebookSpec] = []
    for spec in all_notebook_specs():
        if spec.arch_letter != "a":
            continue
        baselines.append(
            NotebookSpec(
                job_id=spec.job_id,
                job_slug=spec.job_slug,
                job_title=spec.job_title,
                job_num="00",
                arch_letter="a",
                architecture="baseline",
                universe=spec.universe,
                asset_type=spec.asset_type,
                phase=spec.phase,
                status=spec.status,
                target_type=spec.target_type,
                primary_metric=JOB_PRIMARY_METRIC[spec.job_slug],
                is_baseline=True,
            )
        )
    return baselines


def all_notebook_specs_including_baselines() -> list[NotebookSpec]:
    return all_baseline_specs() + all_notebook_specs()


def filter_notebook_specs(
    specs: list[NotebookSpec],
    *,
    job_id: str | None = None,
    job_slug: str | None = None,
    universe: str | None = None,
    status: str | None = None,
    architecture: str | None = None,
) -> list[NotebookSpec]:
    """Narrow notebook specs for targeted scaffold runs."""
    out = specs
    if job_id is not None:
        out = [s for s in out if s.job_id == job_id]
    if job_slug is not None:
        out = [s for s in out if s.job_slug == job_slug]
    if universe is not None:
        out = [s for s in out if s.universe == universe]
    if status is not None:
        out = [s for s in out if s.status == status]
    if architecture is not None:
        out = [s for s in out if s.architecture == architecture]
    return out


def champions_path(job_id: str, job_slug: str) -> Path:
    return ML_DIR / f"J{job_id}_{job_slug}" / "champions.json"


def baseline_artifact_stem(job_slug: str, universe: str, model_tag: str = "") -> str:
    return f"baseline_{job_slug}_{universe}{model_tag}_tf"
