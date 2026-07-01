#!/usr/bin/env python3
"""Prepend notebook_bootstrap to train/setup cells that import tf_notebook_utils."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ML_Iteration_2"

BOOTSTRAP_LINES = [
    "import sys\n",
    "from pathlib import Path\n",
    "for _c in [Path.cwd(), *Path.cwd().parents]:\n",
    "    _s = _c / \"scripts\"\n",
    "    if (_s / \"notebook_bootstrap.py\").exists():\n",
    "        sys.path.insert(0, str(_s))\n",
    "        break\n",
    "from notebook_bootstrap import ensure_scripts_path\n",
    "ensure_scripts_path()\n",
    "\n",
]

MARKERS = (
    "from tf_notebook_utils import",
    "from tf_data import",
)


def needs_bootstrap(source: str) -> bool:
    if "notebook_bootstrap import" in source:
        return False
    return any(m in source for m in MARKERS)


def fix_cell(source: list[str]) -> tuple[list[str], bool]:
    text = "".join(source)
    if not needs_bootstrap(text):
        return source, False
    return BOOTSTRAP_LINES + source, True


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text())
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        new_source, cell_changed = fix_cell(cell.get("source", []))
        if cell_changed:
            cell["source"] = new_source
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1))
    return changed


def main() -> int:
    targets = [p for p in ML_DIR.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts]
    fixed = sum(1 for p in targets if fix_notebook(p))
    print(f"Fixed import bootstrap in {fixed}/{len(targets)} notebooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
