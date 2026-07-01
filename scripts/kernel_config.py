"""Single source of truth for the stock-prediction Jupyter kernel."""

from __future__ import annotations

KERNEL_NAME = "stock-prediction"
KERNEL_DISPLAY_NAME = "Python (stock-prediction)"

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": KERNEL_DISPLAY_NAME,
        "language": "python",
        "name": KERNEL_NAME,
    },
    "language_info": {
        "name": "python",
        "version": "3.12.3",
    },
}

# Legacy kernelspec names replaced by fix_notebook_kernels.py
LEGACY_KERNEL_NAMES = frozenset(
    {
        "python3",
        "stock-prediction-wsl",
        "stock-tf-5090",
        "tf-gpu-wsl",
        "tf-wsl-gpu",
        "template-ml",
    }
)

KERNEL_PICKER_HINT = (
    f"**Kernel:** select **`{KERNEL_DISPLAY_NAME}`** in the kernel picker "
    f"(register with `bash scripts/install-jupyter-kernel.sh` if missing)."
)
