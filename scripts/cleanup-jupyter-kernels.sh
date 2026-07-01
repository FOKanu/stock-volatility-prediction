#!/usr/bin/env bash
# Remove stale stock-prediction Jupyter kernels that clutter the picker.
set -euo pipefail

USER_KERNEL_ROOT="${HOME}/.local/share/jupyter/kernels"

# Kernels that pointed at this project or older TF experiments (safe to remove).
STALE_NAMES=(
  stock-prediction-wsl
)

echo "Cleaning stale user-level Jupyter kernels under ${USER_KERNEL_ROOT}"
for name in "${STALE_NAMES[@]}"; do
  dir="${USER_KERNEL_ROOT}/${name}"
  if [[ -d "$dir" ]]; then
    rm -rf "$dir"
    echo "  removed ${name}"
  fi
done

echo "Done. Re-run: bash scripts/install-jupyter-kernel.sh"
