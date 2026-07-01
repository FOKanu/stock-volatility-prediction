"""Resolve stock-prediction project root and put scripts/ on sys.path (Jupyter cells)."""

from __future__ import annotations

import sys
from pathlib import Path

_NOTEBOOK_ROOT_NAMES = frozenset(
    {"04_ML_Model_Iteration-2", "ML_Iteration_2", "best model", "scripts", "stocks", "etfs", "market"}
)


def ensure_scripts_path() -> Path:
    """Find project root from cwd and prepend scripts/ to sys.path."""
    root = Path.cwd().resolve()
    if root.name in _NOTEBOOK_ROOT_NAMES:
        root = root.parent
    marker = root / "scripts" / "tf_data.py"
    if not marker.exists():
        for candidate in [Path.cwd(), *Path.cwd().parents]:
            if (candidate / "scripts" / "tf_data.py").exists():
                root = candidate.resolve()
                break
    scripts = root / "scripts"
    if not (scripts / "tf_data.py").exists():
        raise ModuleNotFoundError(
            f"Could not find scripts/tf_data.py (cwd={Path.cwd()}). "
            "Run from experiments/stock-prediction or run the Setup cell first."
        )
    scripts_str = str(scripts)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    return root


def activate_scripts_path() -> Path:
    """Walk parents to import this module, then ensure scripts/ is on sys.path."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        scripts = candidate / "scripts"
        if (scripts / "notebook_bootstrap.py").exists():
            scripts_str = str(scripts)
            if scripts_str not in sys.path:
                sys.path.insert(0, scripts_str)
            break
    return ensure_scripts_path()
