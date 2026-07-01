#!/usr/bin/env python3
"""Pin all project notebooks to the canonical stock-prediction Jupyter kernel."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from kernel_config import KERNEL_DISPLAY_NAME, KERNEL_METADATA, KERNEL_PICKER_HINT

ROOT = Path(__file__).resolve().parents[1]

# Directories scanned for notebooks (relative to project root).
NOTEBOOK_ROOTS = (
    ROOT / "ML_Iteration_2",
    ROOT / "04_ML_Model_Iteration-1",
    ROOT / "Analytics notebooks",
)

KERNEL_LINE_PATTERNS: tuple[re.Pattern[str], ...] = ()


CANONICAL_KERNEL_MARKER = "**Kernel:** select **`Python (stock-prediction)`**"


def _fix_markdown_kernel_lines(source: str) -> tuple[str, bool]:
    if CANONICAL_KERNEL_MARKER in source:
        return source, False
    if "**Kernel:**" not in source:
        return source, False
    new_source = re.sub(
        r"\*\*Kernel:\*\*[^\n]*\n?",
        KERNEL_PICKER_HINT + "\n",
        source,
        count=1,
    )
    return new_source, new_source != source


def fix_notebook(path: Path, *, dry_run: bool = False) -> bool:
    nb = json.loads(path.read_text())
    changed = False

    ks = nb.get("metadata", {}).get("kernelspec", {})
    if (
        ks.get("name") != KERNEL_METADATA["kernelspec"]["name"]
        or ks.get("display_name") != KERNEL_METADATA["kernelspec"]["display_name"]
    ):
        nb.setdefault("metadata", {})
        nb["metadata"]["kernelspec"] = dict(KERNEL_METADATA["kernelspec"])
        nb["metadata"]["language_info"] = {
            **nb["metadata"].get("language_info", {}),
            **KERNEL_METADATA["language_info"],
        }
        changed = True

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        new_source, cell_changed = _fix_markdown_kernel_lines(source)
        if cell_changed:
            cell["source"] = new_source.splitlines(keepends=True)
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"
            changed = True

    if changed and not dry_run:
        path.write_text(json.dumps(nb, indent=1))
    return changed


def iter_notebooks() -> list[Path]:
    paths: list[Path] = []
    for root in NOTEBOOK_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in path.parts:
                continue
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any notebook is not pinned to the canonical kernel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report notebooks that would change without writing.",
    )
    args = parser.parse_args()

    targets = iter_notebooks()
    if not targets:
        print("No notebooks found.", file=sys.stderr)
        return 1

    outdated: list[Path] = []
    for path in targets:
        if fix_notebook(path, dry_run=True):
            outdated.append(path)
            rel = path.relative_to(ROOT)
            if args.dry_run:
                print(f"would fix {rel}")
            elif not args.check:
                fix_notebook(path)
                print(f"fixed {rel}")

    if args.check:
        if outdated:
            print(
                f"{len(outdated)} notebook(s) need kernel pin → "
                f"run: python scripts/fix_notebook_kernels.py",
                file=sys.stderr,
            )
            for path in outdated:
                print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
            return 1
        print(f"OK: all {len(targets)} notebooks use {KERNEL_DISPLAY_NAME}")
        return 0

    fixed = len(outdated)
    print(f"Done: {fixed}/{len(targets)} notebooks updated → {KERNEL_DISPLAY_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
