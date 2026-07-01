#!/usr/bin/env bash
# Register the ONE Jupyter kernel for stock-prediction (TensorFlow GPU / WSL).
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT}/.venv"
KERNEL_NAME="stock-prediction"
DISPLAY_NAME="Python (stock-prediction)"
KERNEL_DIR="${VENV}/share/jupyter/kernels/${KERNEL_NAME}"
USER_KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}"
SP="${VENV}/lib/python3.12/site-packages/nvidia"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing ${VENV}/bin/python — run scripts/setup-wsl-venv.sh first." >&2
  exit 1
fi

LD_PATHS=(
  "${SP}/cudnn/lib"
  "${SP}/cublas/lib"
  "${SP}/cuda_runtime/lib"
  "${SP}/cusolver/lib"
  "${SP}/cusparse/lib"
  "${SP}/curand/lib"
  "${SP}/nvjitlink/lib"
  "${SP}/cufft/lib"
  "${SP}/nccl/lib"
  "/usr/lib/wsl/lib"
)
LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PATHS[*]}"):"

write_kernel() {
  local target_dir="$1"
  mkdir -p "$target_dir"
  cat > "${target_dir}/kernel.json" <<EOF
{
  "argv": [
    "${VENV}/bin/python",
    "-Xfrozen_modules=off",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "${DISPLAY_NAME}",
  "language": "python",
  "metadata": {
    "debugger": true,
    "supported_encryption": "curve"
  },
  "kernel_protocol_version": "5.5",
  "env": {
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "XLA_FLAGS": "--xla_gpu_cuda_data_dir=${SP}/cuda_nvcc",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "PIP_USER": ""
  }
}
EOF
  echo "Wrote ${target_dir}/kernel.json"
}

write_kernel "$KERNEL_DIR"
write_kernel "$USER_KERNEL_DIR"

# Remove duplicate / legacy kernels for this project (project venv).
LEGACY_PROJECT_KERNELS=(python3 stock-prediction-wsl)
for legacy in "${LEGACY_PROJECT_KERNELS[@]}"; do
  if [[ -d "${VENV}/share/jupyter/kernels/${legacy}" ]]; then
    rm -rf "${VENV}/share/jupyter/kernels/${legacy}"
    echo "Removed legacy project kernel: ${legacy}"
  fi
done

echo ""
echo "Registered kernel: ${DISPLAY_NAME} (${KERNEL_NAME})"
echo "Python: ${VENV}/bin/python"
echo ""
echo "Next: python scripts/fix_notebook_kernels.py"
