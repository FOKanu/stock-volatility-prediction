"""Project root and raw dataset path resolution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_MARKERS = (
    Path("scripts/run_preprocessing.py"),
    Path("scripts/tf_data.py"),
    Path("ML_Iteration_2/09a_gru_15_fts_all_stock.ipynb"),
    Path("data/processed"),
    Path("Stock Market Dataset"),
)

DEFAULT_DATASET_DIR = Path(
    r"C:\Users\friyekekanu\Datasets\Project - Stock Performance Prediction\Stock Market Dataset"
)
WSL_DATASET_DIR = Path(
    "/home/friyekekanu/ai-workspace/experiments/stock-prediction/Stock Market Dataset"
)


def find_project_root() -> Path:
    candidates: list[Path] = [Path.cwd()]
    if Path.cwd().name in (
        "scripts",
        "notebooks",
        "04_ML_Model_Iteration-1",
        "04_ML_Model_Iteration-2",
        "ML_Iteration_2",
    ):
        candidates.append(Path.cwd().parent)
    candidates.extend(Path.cwd().parents)

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for marker in PROJECT_MARKERS:
            if (resolved / marker).exists():
                return resolved

    raise FileNotFoundError(
        "Could not find project root. Run from the Notebooks folder or set STOCK_DATASET_DIR."
    )


def resolve_dataset_dir(project_root: Path | None = None) -> Path:
    env_path = os.environ.get("STOCK_DATASET_DIR")
    if env_path:
        dataset_dir = Path(env_path).expanduser().resolve()
        if dataset_dir.exists():
            return dataset_dir
        raise FileNotFoundError(f"STOCK_DATASET_DIR does not exist: {dataset_dir}")

    root = project_root or find_project_root()
    local = root / "Stock Market Dataset"
    if local.exists():
        return local.resolve()

    if DEFAULT_DATASET_DIR.exists():
        return DEFAULT_DATASET_DIR.resolve()

    if WSL_DATASET_DIR.exists():
        return WSL_DATASET_DIR.resolve()

    raise FileNotFoundError(
        "Stock Market Dataset not found. Set STOCK_DATASET_DIR or create a junction at "
        f"{local}"
    )
